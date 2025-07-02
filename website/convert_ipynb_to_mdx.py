import json
import re
import shutil
import uuid
from os import PathLike
from pathlib import Path
from typing import Dict, Tuple, Union
from io import StringIO

import os
import json

import mdformat  
import nbformat
import pandas as pd
from nbformat.notebooknode import NotebookNode

try:
    from libfb.py.fbcode_root import get_fbcode_dir
except ImportError:
    SCRIPTS_DIR = Path(__file__).parent.resolve()
    # If SCRIPTS_DIR is `website`, then just use its parent.
    LIB_DIR = SCRIPTS_DIR.parent.resolve()
else:
    # If the try block is successful, you only need the root directory.
    LIB_DIR = Path(get_fbcode_dir()).resolve()

WEBSITE_DIR = LIB_DIR.joinpath("website")
DOCS_DIR = WEBSITE_DIR.joinpath("docs")
OVERVIEW_DIR = DOCS_DIR
TUTORIALS_DIR = DOCS_DIR.joinpath("tutorials")

def load_nbs_to_convert() -> Dict[str, Dict[str, str]]:
    """Load the metadata and list of notebooks to convert to mdx.

    Args:
        None

    Returns:
        Dict[str, Dict[str, str]]: A dictionary of metadata needed to convert notebooks
        to mdx.

    """
    tutorials_json_path = WEBSITE_DIR.joinpath("tutorials.json")
    with open(str(tutorials_json_path), "r") as f:
        tutorials_data = json.load(f)

    return tutorials_data


def load_notebook(path: Union[PathLike, str]) -> NotebookNode:
    """Load the given notebook into memory.

    Args:
        path (Union[PathLike, str]): Path to the notebook.

    Returns:
        NotebookNode: `nbformat` object.

    """
    if isinstance(path, PathLike):
        path = str(path)
    with open(path, "r") as f:
        nb_str = f.read()
        nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

    return nb


def transform_markdown_cell(
    cell: NotebookNode,
    img_folder: Union[PathLike, str],
) -> str:
    """Transform the given Jupyter markdown cell.

    Args:
        cell (NotebookNode): Jupyter markdown cell object.
        img_folder (Union[PathLike, str]): Image folder path.

    Returns:
        str: Transformed markdown object suitable for inclusion in mdx files.

    """
    img_folder = Path(img_folder)
    cell_source = cell["source"]

    # Check if the cell is displaying an image.
    if cell_source[0] == "!":
        # Change the path to always be `assets/img/...`
        start = cell_source.find("(") + 1
        stop = cell_source.find(")")
        image_path = cell_source[start:stop]
        
        # Check if the image path is a URL (starts with http:// or https://)
        if image_path.startswith(('http://', 'https://')):
            # For remote images, just use the URL directly
            # No need to copy the file locally
            pass
        else:
            # Handle local files as before
            old_img_path = (LIB_DIR / "website" / "docs" / "tutorials" / Path(image_path)).resolve()
            name = old_img_path.name
            img_path_str = f"assets/img/{name}"
            cell_source = cell_source[:start] + img_path_str + cell_source[stop:]
            # Copy the image to the folder where the markdown can access it.
            new_img_path = str(img_folder.joinpath(name))
            shutil.copy(str(old_img_path), new_img_path)

    # Wrap lines using black's default of 88 characters.
    new_cell_source = mdformat.text(
        cell_source,
        options={"wrap": 88},
        extensions={"myst"},
    )

    # We will attempt to handle inline style attributes written in HTML by converting
    # them to something React can consume.
    token = "style="
    pattern = re.compile(f'{token}"([^"]*)"')
    found_patterns = re.findall(pattern, new_cell_source)
    for found_pattern in found_patterns:
        react_style_string = json.dumps(
            dict(
                [
                    [t.strip() for t in token.strip().split(":")]
                    for token in found_pattern.split(";")
                    if token
                ]
            )
        )
        react_style_string = f"{{{react_style_string}}}"
        new_cell_source = new_cell_source.replace(found_pattern, react_style_string)
        new_cell_source = new_cell_source.replace('"{{', "{{").replace('}}"', "}}")
    return f"{new_cell_source}\n\n"


def transform_code_cell(
    cell: NotebookNode,
    plot_data_folder: Union[PathLike, str],
    filename: Union[PathLike, str],
) -> Dict[str, Union[str, bool]]:
    """Transform the given Jupyter code cell.

    Args:
        cell (NotebookNode): Jupyter code cell object.
        plot_data_folder (Union[PathLike, str]): Path to the `plot_data` folder for the
            tutorial.
        filename (str): File name to use for the mdx and jsx output.

    Returns:
        Dict[str, Union[str, bool]]: Dictionary containing mdx output, jsx output, 
        components, and flags for different plot types.

    """
    plot_data_folder = Path(plot_data_folder).resolve()
    # Data display priority.
    priorities = [
        "text/markdown",
        "application/javascript",
        "image/png",
        "image/jpeg",
        "image/svg+xml",
        "image/gif",
        "image/bmp",
        "text/latex",
        "text/html",
        "application/vnd.jupyter.widget-view+json",  # tqdm progress bars
        "text/plain",
    ]

    bokeh_flag = False
    plotly_flag = False
    altair_flag = False
    d3_html_flag = False

    mdx_output = ""
    jsx_output = ""
    link_btn = "../../src/components/LinkButtons.jsx"
    cell_out = "../../src/components/CellOutput.jsx"
    plot_out = "../../src/components/Plotting.jsx"
    components_output = f'import LinkButtons from "{link_btn}";\n'
    components_output += f'import CellOutput from "{cell_out}";\n'

    # Handle cell input.
    cell_source = cell.get("source", "")
    
    # Check if this is a %%html cell (D3/HTML visualization)
    is_html_cell = cell_source.strip().startswith("%%html")
    
    if is_html_cell:
        # For %%html cells, extract the HTML content and save it as a file
        html_content = cell_source[cell_source.find("%%html") + 6:].strip()
        
        # Generate a unique filename for the HTML file
        file_name = f"D3Visualization_{uuid.uuid4().hex[:8]}.html"
        
        # Save to the static/plot_data directory for serving
        static_plot_data_folder = WEBSITE_DIR.joinpath("static").joinpath("plot_data")
        static_plot_data_folder.mkdir(parents=True, exist_ok=True)
        
        file_path = static_plot_data_folder.joinpath(file_name)
        with open(file_path, "w") as f:
            f.write(html_content)
        
        # Add the code block showing the %%html command
        mdx_output += f"```python\n{cell_source}\n```\n\n"
        
        # Add the iframe to display the HTML content
        path_to_html = f"/plot_data/{file_name}"
        mdx_output += (
            f"<iframe src='{path_to_html}' width='100%' height='600' "
            f"style={{{{border: '1px solid #ccc', borderRadius: '4px'}}}}></iframe>\n\n"
        )
        
        d3_html_flag = True
        
        return {
            "mdx": mdx_output,
            "jsx": jsx_output,
            "components": components_output,
            "bokeh": bokeh_flag,
            "plotly": plotly_flag,
            "altair": altair_flag,
            "d3_html": d3_html_flag,
        }
    else:
        # Regular code cell handling
        mdx_output += f"```python\n{cell_source}\n```\n\n"

    # Handle cell outputs.
    cell_outputs = cell.get("outputs", [])
    if cell_outputs:
        # Create a list of all the data types in the outputs of the cell. These values
        # are similar to the ones in the priorities variable.
        cell_output_dtypes = [
            list(cell_output.get("data", {}).keys()) for cell_output in cell_outputs
        ]

        # Order the output of the cell's data types using the priorities list.
        ordered_cell_output_dtypes = [
            sorted(
                set(dtypes).intersection(set(priorities)),
                key=lambda dtype: priorities.index(dtype),
            )
            for dtypes in cell_output_dtypes
        ]

        # Create a list of the cell output types. We will handle each one differently
        # for inclusion in the mdx string. Types include:
        # - "display_data"
        # - "execute_result"
        # - "stream"
        # - "error"
        cell_output_types = [cell_output["output_type"] for cell_output in cell_outputs]

        # We handle bokeh and plotly figures differently, so check to see if the output
        # contains on of these plot types.
        if "plotly" in str(cell_output_dtypes):
            plotly_flag = True
        if "bokeh" in str(cell_output_dtypes):
            bokeh_flag = True

        # Cycle through the cell outputs and transform them for inclusion in the mdx
        # string.
        display_data_outputs = []
        for i, cell_output in enumerate(cell_outputs):
            data_object = (
                ordered_cell_output_dtypes[i][0]
                if ordered_cell_output_dtypes[i]
                # Handle "stream" cell output type.
                else "text/plain"
            )
            data_category, data_type = data_object.split("/")
            cell_output_data = cell_output.get("data", {}).get(data_object, "")
            cell_output_type = cell_output_types[i]

            # Handle "display_data".
            if cell_output_type == "display_data":
                # Handle HTML content from display_data (for cases where HTML is in output)
                if data_category == "text" and data_type == "html":
                    # Check if this looks like a D3/custom HTML visualization
                    if any(keyword in cell_output_data.lower() for keyword in ['d3.', '<script', '<svg', 'visualization']):
                        file_name = f"HTMLVisualization_{uuid.uuid4().hex[:8]}.html"
                        
                        # Save to the static/plot_data directory
                        static_plot_data_folder = WEBSITE_DIR.joinpath("static").joinpath("plot_data")
                        static_plot_data_folder.mkdir(parents=True, exist_ok=True)
                        
                        file_path = static_plot_data_folder.joinpath(file_name)
                        with open(file_path, "w") as f:
                            f.write(cell_output_data)
                        
                        # Add iframe to display the HTML
                        path_to_html = f"/plot_data/{file_name}"
                        mdx_output += (
                            f"<iframe src='{path_to_html}' width='100%' height='600' "
                            f"style={{{{border: '1px solid #ccc', borderRadius: '4px'}}}}></iframe>\n\n"
                        )
                        d3_html_flag = True
                        continue

                if not bokeh_flag and not plotly_flag:
                    # Handle binary images.
                    if data_category == "image":
                        if data_type in ["png", "jpeg", "gif", "bmp"]:
                            mdx_output += (
                                f"![](data:{data_object};base64,{cell_output_data})\n\n"
                            )
                    # TODO: Handle svg images.

                    # Handle tqdm progress bars.
                    if data_type == "vnd.jupyter.widget-view+json":
                        cell_output_data = cell_output["data"]["text/plain"]
                        display_data_outputs.append(cell_output_data)

                # Handle plotly images.
                if plotly_flag:
                    components_output += f'import {{PlotlyFigure}} from "{plot_out}";\n'
                    cell_output_data = cell_output["data"]
                    for key, value in cell_output_data.items():
                        if key == "application/vnd.plotly.v1+json":
                            # Save the plotly JSON data.
                            file_name = "PlotlyFigure" + str(uuid.uuid4())
                            file_path = str(
                                plot_data_folder.joinpath(f"{file_name}.json")
                            )
                            with open(file_path, "w") as f:
                                json.dump(value, f, indent=2)

                            # Add the Plotly figure to the MDX output.
                            path_to_data = f"./assets/plot_data/{file_name}.json"
                            mdx_output += (
                                f"<PlotlyFigure data={{require('{path_to_data}')}} "
                                "/>\n\n"
                            )

                # Handle bokeh images.
                if bokeh_flag:
                    components_output += f'import {{BokehFigure}} from "{plot_out}";\n'
                    # Ignore any HTML data objects. The bokeh object we want is a
                    # `application/javascript` object. We will also ignore the first
                    # bokeh output, which is an image indicating that bokeh is loading.
                    bokeh_ignore = (
                        data_object == "text/html"
                        or "HTML_MIME_TYPE" in cell_output_data
                    )
                    if bokeh_ignore:
                        continue
                    if data_object == "application/javascript":
                        # Parse the cell source to create a name for the component. This
                        # will be used as the id for the div as well as it being added
                        # to the JSON data.
                        plot_name = cell_source.split("\n")[-1]
                        token = "show("
                        plot_name = plot_name[plot_name.find(token) + len(token) : -1]
                        div_name = plot_name.replace("_", "-")
                        # Parse the javascript for the bokeh JSON data.
                        flag = "const docs_json = "
                        json_string = list(
                            filter(
                                lambda line: line.startswith(flag),
                                [
                                    line.strip()
                                    for line in cell_output_data.splitlines()
                                ],
                            )
                        )[0]
                        # Ignore the const definition and the ending ; from the line.
                        json_string = json_string[len(flag) : -1]
                        json_data = json.loads(json_string)
                        # The js from bokeh in the notebook is nested in a single key,
                        # hence the reason why we do this.
                        json_data = json_data[list(json_data.keys())[0]]
                        js = {}
                        js["target_id"] = div_name
                        js["root_id"] = json_data["roots"]["root_ids"][0]
                        js["doc"] = {
                            "defs": json_data["defs"],
                            "roots": json_data["roots"],
                            "title": json_data["title"],
                            "version": json_data["version"],
                        }
                        js["version"] = json_data["version"]
                        # Save the bokeh JSON data.
                        file_path = str(plot_data_folder.joinpath(f"{div_name}.json"))
                        with open(file_path, "w") as f:
                            json.dump(js, f, indent=2)

                            # Add the Bokeh figure to the MDX output.
                        path_to_data = f"./assets/plot_data/{div_name}.json"
                        mdx_output += (
                            f"<BokehFigure data={{require('{path_to_data}')}} />\n\n"
                        )

            # Handle "execute_result".
            if cell_output_type == "execute_result":
                # Handle binary images.
                if data_category == "image":
                    if data_type in ["png", "jpeg", "gif", "bmp"]:
                        mdx_output += (
                            f"![](data:{data_object};base64,{cell_output_data})\n\n"
                        )
                    # TODO: Handle svg images.

                if data_category == "text":
                    # Handle HTML.
                    if data_type == "html":
                        # Handle pandas DataFrames. There is a scoped style tag in the
                        # DataFrame output that uses the class name `dataframe` to style
                        # the output. We will use this token to determine if a pandas
                        # DataFrame is being displayed.
                        if "dataframe" in cell_output_data:
                            df = pd.read_html(StringIO(cell_output_data), flavor="lxml")
                            # NOTE: The return is a list of dataframes and we only care
                            #       about the first one.
                            md_df = df[0]

                            # Check and handle MultiIndex in DataFrame columns
                            if isinstance(md_df.columns, pd.MultiIndex):
                                md_df.columns = ['_'.join(col).strip() for col in md_df.columns.values]

                            # Iterate over columns to rename 'Unnamed' columns
                            for column in md_df.columns:
                                if isinstance(column, tuple):
                                    # Handle tuple case (if still present after flattening)
                                    if column[0].startswith("Unnamed"):
                                        md_df.rename(columns={column: ""}, inplace=True)
                                else:
                                    # Normal case where column is a string
                                    if column.startswith("Unnamed"):
                                        md_df.rename(columns={column: ""}, inplace=True)

                            # Remove the index if it is just a range, and output to markdown.
                            md = ""
                            if isinstance(md_df.index, pd.RangeIndex):
                                md = md_df.to_markdown(index=False)
                            else:
                                md = md_df.to_markdown()
                            mdx_output += f"\n{md}\n\n"

                    # Handle plain text.
                    if data_type == "plain":
                        cell_output_data = "\n".join(
                            [line for line in cell_output_data.splitlines() if line]
                        )
                        display_data_outputs.append(cell_output_data)
                    # Handle markdown.
                    if data_type == "markdown":
                        mdx_output += f"{cell_output_data}\n\n"

            # Handle "stream".
            if cell_output_type == "stream":
                # Ignore if the output is an error.
                if cell_output["name"] == "stderr":
                    continue
                cell_output_data = cell_output.get("text", None)
                if cell_output_data is not None:
                    cell_output_data = "\n".join(
                        [line for line in cell_output_data.splitlines() if line]
                    )
                    display_data_outputs.append(cell_output_data)

            if cell_output_type == "execute_result":
                data = cell_output.get("data", {})
                if "text/html" in data and "alt.HConcatChart" in data.get("text/plain", ""):
                    html_content = data["text/html"]
                    file_name = f"AltairPlot_{uuid.uuid4()}.html"

                    # Assuming 'static' directory is at the root of your Docusaurus project
                    file_path = WEBSITE_DIR.joinpath("static").joinpath("plot_data").joinpath(file_name)
                    
                    # Ensure the plot_data directory exists
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(file_path, "w") as f:
                        f.write(html_content)

                    # The path where the file is served, relative to your site's base URL
                    path_to_html = f"/plot_data/{file_name}"
                    mdx_output += (
                        f"<iframe src='{path_to_html}' width='100%' height='520'></iframe>\n\n"
                    )
                    altair_flag = True

        if display_data_outputs:
            cell_output_data = "\n".join(display_data_outputs)
            mdx_output += f"<CellOutput>\n{{`{cell_output_data}`}}\n</CellOutput>\n\n"

    return {
        "mdx": mdx_output,
        "jsx": jsx_output,
        "components": components_output,
        "bokeh": bokeh_flag,
        "plotly": plotly_flag,
        "altair": altair_flag,
        "d3_html": d3_html_flag,
    }

def find_frontmatter_ending(mdx: str, stop_looking_after: int = 10) -> int:
    """Find the line number where the mdx frontmatter ends.

    Args:
        mdx (str): String representation of the mdx file.
        stop_looking_after (int): Optional, default is 10. Number of lines to stop
        looking for the end of the frontmatter.

    Returns:
        int: The next line where the frontmatter ending is found.

    Raises:
        IndexError: No markdown frontmatter was found.

    """
    indices = []
    still_looking = 0
    lines = mdx.splitlines()
    for i, line in enumerate(lines):
        still_looking += 1
        if still_looking >= stop_looking_after:
            break
        if line == "---":
            indices.append(i)
            still_looking = 0
        if i == len(line) - 1:
            break

    if not indices:
        msg = "No markdown frontmatter found in the tutorial."
        raise IndexError(msg)

    return max(indices) + 1


def transform_notebook(path: Union[str, PathLike]) -> Tuple[str, str]:
    """Transform the given Jupyter notebook into strings suitable for mdx and jsx files.

    Args:
        path (Union[str, PathLike]): Path to the Jupyter notebook.

    Returns:
        Tuple[str, str]: mdx string, jsx string

    """
    # Ensure the given path is a pathlib.PosixPath object.
    path = Path(path).resolve()

    # Load all metadata for notebooks that should be included in the documentation.
    nb_metadata = load_nbs_to_convert()

    # Extract title and generate the key consistently
    title = extract_title(path)
    tutorial_folder_name = title.replace(' ', '_').lower()
    filename = title  # Since 'filename' is used for output

    #Use file path for tutorials directory
    tutorial_folder = TUTORIALS_DIR
    assets_folder = tutorial_folder.joinpath("assets")
    img_folder = assets_folder.joinpath("img")
    plot_data_folder = assets_folder.joinpath("plot_data")
    if not tutorial_folder.exists():
        tutorial_folder.mkdir(parents=True, exist_ok=True)
    if not img_folder.exists():
        img_folder.mkdir(parents=True, exist_ok=True)
    if not plot_data_folder.exists():
        plot_data_folder.mkdir(parents=True, exist_ok=True)

    # Load the notebook.
    nb = load_notebook(path)
    print("The tutorial folder name is", tutorial_folder_name)

    # Begin to build the mdx string.
    mdx = ""
    # Add the frontmatter to the mdx string. This is the part between the `---` lines
    # that define the tutorial sidebar_label information.
    frontmatter = "\n".join(
        ["---"]
        + [
            f"{key}: {value}"
            for key, value in nb_metadata.get(
                tutorial_folder_name,
                {
                    "title": "",
                    "sidebar_label": "",
                    "path": "",
                    "nb_path": "",
                    "github": "",
                    "colab": "",
                },
            ).items()
        ]
        + ["---"]
    )
    frontmatter_line = len(frontmatter.splitlines())
    mdx += f"{frontmatter}\n"

    # Create the JSX and components strings.
    jsx = ""
    components = set()

    # Cycle through each cell in the notebook.
    bokeh_flags = []
    plotly_flags = []
    d3_html_flags = []
    for cell in nb["cells"]:
        cell_type = cell["cell_type"]

        # Handle markdown cell objects.
        if cell_type == "markdown":
            mdx += transform_markdown_cell(cell, img_folder)

        # Handle code cell objects.
        if cell_type == "code":
            tx = transform_code_cell(cell, plot_data_folder, filename)
            mdx += str(tx["mdx"])
            jsx += str(tx["jsx"])
            bokeh_flags.append(tx["bokeh"])
            plotly_flags.append(tx["plotly"])
            d3_html_flags.append(tx.get("d3_html", False))
            for component in str(tx["components"]).splitlines():
                components.add(component)

    # Add the JSX template object to the jsx string. Only include the plotting component
    # that is needed.
    bokeh_flag = any(bokeh_flags)
    plotly_flag = any(plotly_flags)
    d3_html_flag = any(d3_html_flags)
    plotting_fp = "./website/src/components/Plotting.jsx"
    JSX_TEMPLATE = ["import React from 'react';"]
    if bokeh_flag:
        JSX_TEMPLATE.append(f"import {{ BokehFigure }} from '{plotting_fp}';")
    if plotly_flag:
        JSX_TEMPLATE.append(f"import {{ PlotlyFigure }} from '{plotting_fp}';")

    jsx = "\n".join([item for item in JSX_TEMPLATE if item]) + "\n\n" + jsx
    # Remove the last line since it is blank.
    jsx = "\n".join(jsx.splitlines()[:-1])

    # Add the react components needed to display bokeh objects in the mdx string.
    mdx = mdx.splitlines()
    mdx[frontmatter_line:frontmatter_line] = list(components) + [""]
    # Add the react components needed to display links to GitHub and Colab.
    idx = frontmatter_line + len(components) + 1

    glk = nb_metadata[tutorial_folder_name]["github"]
    clk = nb_metadata[tutorial_folder_name]["colab"]
    mdx[idx:idx] = (
        f'<LinkButtons\n  githubUrl="{glk}"\n  colabUrl="{clk}"\n/>\n\n'
    ).splitlines()
    mdx = "\n".join(mdx)

    # Write the mdx file to disk.
    mdx_filename = str(TUTORIALS_DIR.joinpath(f"{filename}.mdx"))
    with open(mdx_filename, "w") as f:
        f.write(mdx)

    # Write the jsx file to disk.
    jsx_filename = str(TUTORIALS_DIR.joinpath(f"{filename}.jsx"))
    with open(jsx_filename, "w") as f:
        f.write(jsx)

    # Return the mdx and jsx strings for debugging purposes.
    return mdx, jsx

def extract_title(notebook_path):
    # Extract the title from the notebook filename or file content
    return notebook_path.stem.replace('_', ' ').title()

def generate_tutorials_json(examples_dir):
    tutorials_json = {}
    for nb_file in os.listdir(examples_dir):
        if nb_file.endswith(".ipynb"):
            title = extract_title(Path(nb_file))
            nb_path = f"website/docs/tutorials/{nb_file}"
            file_name = title.replace(' ', '_')
            tutorials_json[file_name.lower()] = {
                "title": title,
                "sidebar_label": title,
                "path": f"website/docs/tutorials",
                "nb_path": nb_path,
                "github": f"https://github.com/xplainable/xplainable/blob/main/examples/{file_name}.ipynb",
                "colab": f"https://colab.research.google.com/github/xplainable/xplainable/blob/main/examples/{file_name}.ipynb"
            }
    return tutorials_json


def clear_plot_data_files(plot_data_folder: Path, extension: str = ".html"):
    """Remove all files with the given extension in the specified directory."""
    for file in plot_data_folder.glob(f'*{extension}'):
        file.unlink()

if __name__ == "__main__":

    print("--------------------------------------------")
    print("Delete current html files served in static")
    print("--------------------------------------------")

    # Define the path to your 'plot_data' directory
    plot_data_dir = WEBSITE_DIR.joinpath("static").joinpath("plot_data")

    # Ensure the plot_data directory exists
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    # Clear all existing HTML files in the plot_data directory
    for html_file in plot_data_dir.glob("*.html"):
        html_file.unlink()

    print("--------------------------------------------")
    print("Creating the tutorials.json file output     ")
    print("--------------------------------------------")
    # Generate tutorials.json
    # examples_dir = LIB_DIR.joinpath('website/docs/tutorials')  # Adjust the path as needed
    examples_dir = TUTORIALS_DIR
    print(examples_dir)
    tutorials_json = generate_tutorials_json(examples_dir)

    # Write tutorials.json to disk
    tutorials_json_path = WEBSITE_DIR.joinpath("tutorials.json")  # Adjust the path as needed
    with open(tutorials_json_path, 'w') as f:
        json.dump(tutorials_json, f, indent=4)

    #Load the tutorials metadata
    tutorials_metadata = load_nbs_to_convert()

    print("--------------------------------------------")
    print("Converting tutorial notebooks into mdx files")
    print("--------------------------------------------")
    for _, value in tutorials_metadata.items():
        path = (LIB_DIR / value["nb_path"]).resolve()
        print(f"{path.stem}")
        mdx, jsx = transform_notebook(path)
    print("")
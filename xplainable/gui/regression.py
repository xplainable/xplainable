from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.models.regression import XRegressor
from xplainable.utils import TrainButton, ping_server
from pandas.api.types import is_numeric_dtype
from xplainable.exceptions import *
from xplainable.quality import XScan
import xplainable

def regressor(df, model_name, model_description=''):
    """ Trains an xplainable regressor via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target
        model_name (str): A unique name for the model

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XRegressor: The trained model
    """

    nan_values = df.isna().sum().sum()
    if nan_values > 0:
        raise MissingValueError("Ensure there are no missing values before training")

    # This allows widgets to show full label text
    style = {'description_width': 'initial'}

    # Instantiate widget output
    output = widgets.Output()
    model = XRegressor(
        model_name=model_name,
        model_description=model_description)

    # HEADER
    #logo = open('../_img/logo.png', 'rb').read()
    #logo_display = widgets.Image(
    #    value=logo, format='png', width=50, height=50)
    #logo_display.layout = widgets.Layout(margin='15px 25px 15px 15px')

    header_title = widgets.HTML(f"<h3>Model: {model_name}&nbsp&nbsp</h3>")
    header_title.layout = widgets.Layout(margin='10px 0 0 0')

    connection_status = widgets.HTML(f"<h5><font color='red'>[offline]</h5>")
    connection_status.layout = widgets.Layout(margin='10px 0 0 0')

    connection_status_button = widgets.Button(description="offline")
    connection_status_button.layout = widgets.Layout(margin='10px 0 0 0')
    connection_status_button.style = {
            "button_color": 'red',
            "text_color": 'white'
            }

    header = widgets.VBox(
        [widgets.HBox([
            #widgets.VBox([logo_display]),
            header_title, connection_status_button])])

    # TAB 1
    # Column A
    title_a1 = widgets.HTML(
        f"<h5>Target</h5>", layout=widgets.Layout(height='auto'))

    id_title = widgets.HTML(f"<h5>ID Column (0 selected)</h5>")

    numeric_columns = df.select_dtypes('number').columns
    if len(numeric_columns) == 0:
        raise TypeError("You must have at least one numeric column to set as the regressor target")
    
    # get all cols with cardinality of 1
    potential_id_cols = [None] + [col for col in df.columns if XScan._cardinality(df[col]) == 1]

    col_a_layout = widgets.Layout(width='200px')
    id_columns = widgets.SelectMultiple(options=potential_id_cols, style=style)
    id_columns.layout = col_a_layout
    
    numeric_columns = list(reversed([i for i in list(numeric_columns) if i not in potential_id_cols]))
    numeric_columns = [None] + numeric_columns
    
    target = widgets.Dropdown(options=numeric_columns)
    target.layout = col_a_layout

    # Hide ID selector if no ID cols
    if len(potential_id_cols) == 0:
        id_title.layout.visibility = 'hidden'
        id_columns.layout.visibility = 'hidden'

    colA = widgets.VBox([title_a1, target, id_title, id_columns])
    colA.layout = widgets.Layout(margin='0 25px 0 0')

    # Column B
    title_b = widgets.HTML(
        f"<h5>Hyperparameters</h5>",
        layout=widgets.Layout(height='auto'))

    max_depth = widgets.IntSlider(
        value=25, min=2, max=100, step=1, description='max_depth:',
        style=style)

    min_leaf_size = widgets.FloatSlider(
        value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_leaf_size:', style=style)

    min_info_gain = widgets.FloatSlider(value=0.015, min=0.001, max=0.2,\
        step=0.001, readout_format='.3f', description='min_info_gain:',
        style=style)

    bin_alpha = widgets.FloatSlider(value=0.2, min=0.01, max=0.5, step=0.01,
    description='bin_alpha:', style=style)

    tail_sensitivity = widgets.FloatSlider(value=1.1, min=1.0, max=1.5,
    step=0.01, description='tail_sensitivity:', style=style)

    range_header = widgets.HTML(f"<h5>Prediction Range</h5>")
    range_header.layout.visibility = 'hidden'

    prediction_range_slider = widgets.FloatRangeSlider()
    prediction_range_slider.layout.visibility = 'hidden'

    validation_size_header = widgets.HTML(f"<h5>Validation Size</h5>")
    validation_size = widgets.FloatSlider(value=0.2, min=0.05, max=0.5, step=0.01)

    colBParams = widgets.VBox([
        title_b, max_depth, min_leaf_size, min_info_gain,
        bin_alpha, tail_sensitivity])

    colBSettings = widgets.VBox([validation_size_header, validation_size, range_header, prediction_range_slider])

    colB = widgets.Tab([colBParams, colBSettings])
    colB.set_title(0, 'Hyperparameters')
    colB.set_title(1, 'Settings')

    # Build Tab
    tab1 = widgets.HBox([colA, colB])

    # TAB 2
    # Add Evolve stage
    def add_button_evolve_clicked(_):

        vals = [
            mutations.value,
            generations.value,
            max_generation_depth.value,
            max_severity.value,
            max_leaves.value
        ]

        value_string = ','.join([str(i) for i in vals])

        if len(pipeline.options) == 0:
            new_option = f'1: Tighten({value_string})'

        elif pipeline.options[0] == "":
            new_option = f'{len(pipeline.options)}: Evolve({value_string})'
        else:
            new_option = f'{len(pipeline.options)+1}: Evolve({value_string})'

        pipeline.options = (*pipeline.options, new_option)
        pipeline.options = (o for o in pipeline.options if o != "")

    # Add Tighten stage
    def add_button_tighten_clicked(_):
        vals = [
            iterations.value,
            learning_rate.value,
            early_stopping.value
        ]

        value_string = ','.join([str(i) for i in vals])

        if len(pipeline.options) == 0:
            new_option = f'1: Tighten({value_string})'

        elif pipeline.options[0] == "":
            new_option = f'{len(pipeline.options)}: Tighten({value_string})'

        else:
            new_option = f'{len(pipeline.options)+1}: Tighten({value_string})'

        pipeline.options = (*pipeline.options, new_option)
        pipeline.options = (o for o in pipeline.options if o != "")

    # Drop selected stage
    def drop_button_clicked(b):
        val = pipeline.value

        if not val:
            return

        elif len(pipeline.options) == 1:
            pipeline.options = ('')

        elif val and val[0] != '':    
            pipeline.options = (o for o in pipeline.options if o not in val)

            pipeline.options = (
                f'{i+1}:{o.split(":")[1]}' for i, o in enumerate(pipeline.options))

    def id_cols_changed(_):
        id_vals = [i for i in list(id_columns.value) if i is not None]
        id_title.value = f"<h5>ID Column ({len(id_vals)} selected)</h5>"

    def target_changed(_):

        if target.value is None:
            prediction_range_slider.layout.visibility = 'hidden'
            range_header.layout.visibility = 'hidden'
            train_button.disabled = True
            return

        ser = df[target.value]

        if not is_numeric_dtype(ser):
            prediction_range_slider.layout.visibility = 'hidden'
            range_header.layout.visibility = 'hidden'
            
        else:
            prediction_range_slider.layout.visibility = 'visible'
            range_header.layout.visibility = 'visible'
            mn = ser.min()
            mx = ser.max()
            rng = mx - mn

            prediction_range_slider.min = round(mn - (rng*0.2), 4)
            prediction_range_slider.max = round(mx + (rng*0.2), 4)
            prediction_range_slider.value = [mn, mx]

        train_button.disabled = False

    def _check_connection(_):
        try:
            if ping_server(xplainable.__client__.compute_hostname):
                connection_status_button.description = "Connected"
                connection_status_button.style.button_color = 'green'
            else:
                connection_status_button.description = "Offline"
                connection_status_button.style.button_color = 'red'
        except:
            pass

    connection_status_button.on_click(_check_connection)

    id_columns.observe(id_cols_changed, names=['value'])
    target.observe(target_changed, names=['value'])

    # Column 1
    html_style = widgets.Layout(flex_flow = 'row wrap', max_width = '300px')
    title_1 = widgets.HTML(
        f"<h5>Evolve</h5><p>Optimises weights with evolutionary neural network</p>")
    title_1.layout = html_style

    mutations = widgets.IntSlider(
        value=100, min=5, max=200, step=5, description='mutations:', style=style)

    generations = widgets.IntSlider(
        value=20, min=5, max=50, step=5, description='generations:', style=style)

    max_generation_depth = widgets.IntSlider(
        value=10, min=2, max=100, step=1,
        description='max_generation_depth:', style=style)

    max_severity = widgets.FloatSlider(
        value=0.25, min=0.01, max=0.5, step=0.01, description='max_severity:',
        style=style)

    max_leaves = widgets.IntSlider(
        value=20, min=1, max=100, step=5, description='max_leaves:', style=style)

    add_button_evolve = widgets.Button(description="Add Stage",icon='plus')
    add_button_evolve.style.button_color = '#12b980'
    add_button_evolve.on_click(add_button_evolve_clicked)

    col1 = widgets.VBox([
        title_1, mutations, generations, max_generation_depth, max_severity,
        max_leaves, add_button_evolve])
    #col1.layout = widgets.Layout(max_width = '200px')

    # Column 2
    title_2 = widgets.HTML(f"<h5>Tighten</h5><p>Optimises weights with iterative error correction</p>")
    title_2.layout = html_style
    add_button_tighten = widgets.Button(description="Add Stage", icon='plus')
    add_button_tighten.style.button_color = '#12b980'
    add_button_tighten.on_click(add_button_tighten_clicked)

    iterations = widgets.IntSlider(
        value=100, min=10, max=500, step=10, description='iterations:', style=style)

    learning_rate = widgets.FloatSlider(
        value=0.05, min=0.005, max=0.2, step=0.005, description='learning_rate:',
        style=style, readout_format='.3f')

    early_stopping = widgets.IntSlider(
        value=100, min=5, max=200, step=5, description='early_stopping:', style=style)

    col2 = widgets.VBox(
        [title_2, iterations, learning_rate, early_stopping, add_button_tighten])

    # Build tab
    tab2 = widgets.HBox([col1, col2])

    # Join tabs
    tabs = widgets.Tab([tab1, tab2])
    tabs.set_title(0, 'Parameters')
    tabs.set_title(1, 'Optimisation')
    tabs.layout = widgets.Layout(margin='0 0 0 15px')

    # PIPELINE
    title_3 = widgets.HTML(
        f"<h5>Optimisation Pipeline</h5><p>Select from optimisation tab</p>",
        layout=widgets.Layout(height='auto'))

    pipeline = widgets.SelectMultiple(
        index=(0,),
        options=[''],
        rows=10,
        disabled=False,
        layout=widgets.Layout(width='300px'))

    # Create drop button
    drop_button = widgets.Button(description="Drop Stage(s)",icon='times')
    drop_button.style.button_color = '#e21c47'
    drop_button.on_click(drop_button_clicked)

    # Build column 3
    col3 = widgets.Box([widgets.VBox([title_3, pipeline, drop_button])])
    col3.layout = widgets.Layout(margin='0 0 0 25px')

    # Build body
    body = widgets.HBox([tabs, col3])

    # FOOTER
    # Clear output on click
    def close_button_clicked(_):
        clear_output()

    # Train model on click
    def train_button_clicked(b):

        prediction_range = None if prediction_range_slider.layout.visibility == 'hidden' else list(prediction_range_slider.value)
        
        model = b.model
        model.max_depth = max_depth.value
        model.min_leaf_size = min_leaf_size.value
        model.min_info_gain = min_info_gain.value
        model.bin_alpha = bin_alpha.value
        model.tail_sensitivity = tail_sensitivity.value
        model.prediction_range = prediction_range
        model.validation_size = validation_size.value
        model._layers = [
            i.split(":")[1].strip() for i in list(pipeline.options) if i != '']

        with output:

            body.close()
            footer.close()

            try:
                X, y = df.drop(
                    columns=[target.value]), df[target.value]
                id_vals = [i for i in list(id_columns.value) if i is not None]
                model.fit(X, y, id_columns=id_vals)

            except Exception as e:
                clear_output()
                raise RuntimeError(e)

    divider = widgets.HTML(f'<hr class="solid">')

    #  Create buttons and listen for clicks
    train_button = TrainButton(description="Train", icon='bolt', model=model, disabled=True)
    train_button.style.button_color = '#0080ea'
    train_button.on_click(train_button_clicked)

    close_button = widgets.Button(description='Close')
    close_button.on_click(close_button_clicked)

    train_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')
    close_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')

    # Build footer
    footer = widgets.VBox([divider, widgets.HBox([train_button, close_button])])

    # Build final screen
    screen = widgets.VBox([header, body, footer, output])

    # Display the GUI
    display(screen)

    # Ping server to check for connection
    _check_connection(None)

    # Return blank model await training
    return model

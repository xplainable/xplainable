from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.utils import Loader
from xplainable.models.classification import XClassifier
from xplainable.models.regression import XRegressor
from xplainable.utils import TrainButton


def train_classifier(df, model_name, hostname):
    """ Trains an xplainable classifier via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target
        model_name (str): A unique name for the model
        hostname (str): Base url of host machine

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XClassifier: The trained model
    """

    # This allows widgets to show full label text
    style = {'description_width': 'initial'}

    # Instantiate widget output
    outt = widgets.Output()

    # Instantiate the model
    model = XClassifier(model_name=model_name, hostname=hostname)

    # HEADER
    logo = open('xplainable/_img/white_bg.png', 'rb').read()
    logo_display = widgets.Image(
        value=logo, format='png', width=50, height=50)

    header_title = widgets.HTML(
        f"<h2>\u00a0\u00a0\u00a0Model: {model_name}</h2>",
        layout=widgets.Layout(height='auto'))

    divider = widgets.HTML(
        f'<hr class="solid">', layout=widgets.Layout(height='auto'))

    header = widgets.VBox(
        [widgets.HBox([widgets.VBox([logo_display]), header_title])])

    # COLUMN 1
    col1a = widgets.HTML(
        f"<h4>Target</h4>", layout=widgets.Layout(height='auto'))

    col1b = widgets.HTML(
        f"<h4>ID Column(s)</h4>", layout=widgets.Layout(height='auto'))

    target = widgets.Dropdown(options=df.columns)
    id_columns = widgets.SelectMultiple(options=df.columns, style=style)

    colA = widgets.VBox([col1a, target, col1b, id_columns])

    # COLUMN 2

    # Options to toggle param/opt view
    options = {
        True: 'flex',
        False: 'none'
    }

    # Change view on optimisation button change
    def on_change(_):
        vala = optimise.value
        valb = not vala

        n_trials.layout.display = options[vala]
        early_stopping.layout.display = options[vala]

        max_depth.layout.display = options[valb]
        min_leaf_size.layout.display = options[valb]
        min_info_gain.layout.display = options[valb]
        bin_alpha.layout.display = options[valb]

    # Select display from optimise dropdown
    col2 = widgets.HTML(
        f"<h4>Parameters</h4>", layout=widgets.Layout(height='auto'))

    optimise = widgets.Dropdown(
        value=False, options=[True, False],
        description='optimise:', style=style)

    optimise.observe(on_change)

    # Param pickers
    max_depth = widgets.IntSlider(
        value=12, min=2, max=100, step=1, description='max_depth:',
        style=style)

    min_leaf_size = widgets.FloatSlider(
        value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_leaf_size:', style=style)

    min_info_gain = widgets.FloatSlider(
        value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_info_gain:', style=style)
    
    bin_alpha = widgets.FloatSlider(
        value=0.05, min=0.01, max=0.5, step=0.01, description='bin_alpha:',
        style=style)
    
    # Optimise param pickers
    n_trials = widgets.IntSlider(
        value=30, min=5, max=150, step=5, description='n_trials:',
        style=style)

    early_stopping = widgets.IntSlider(
        value=15, min=5, max=50, step=5, description='early_stopping:',
        style=style)

    # Set initial optimise widgets to no display
    n_trials.layout.display = 'none'
    early_stopping.layout.display = 'none'

    colB = widgets.VBox(
        [col2, optimise, max_depth, min_leaf_size, min_info_gain,
        bin_alpha, n_trials, early_stopping])
    
    body = widgets.HBox([colA, colB])

    # FOOTER
    train_button = TrainButton(description='Train Model', model=model)
    close_button = widgets.Button(description='Close')
    footer = widgets.VBox(
        [divider, widgets.HBox([train_button, close_button])])

    # SCREEN – Build final screen
    screen = widgets.VBox([header, body, footer, outt])
    
    # Close screen
    def close():
        header.close()
        body.close()
        footer.close()

    # Close and clear
    def close_button_click(_):
        close()
        clear_output()

    # Train model on click
    def train_button_clicked(b):
        with outt:
        
            model = b.model         
            model.max_depth = max_depth.value,
            model.min_leaf_size = min_leaf_size.value,
            model.min_info_gain = min_info_gain.value,
            model.bin_alpha = bin_alpha.value,
            model.optimise = optimise.value,
            model.n_trials = n_trials.value,
            model.early_stopping = early_stopping.value

            try:
                close()
                X, y = df.drop(
                    columns=[target.value]), df[target.value]

                model.fit(X, y, list(id_columns.value))

            except Exception as e:
                close()
                clear_output()
                raise RuntimeError(e)

    # Listen for clicks
    train_button.on_click(train_button_clicked)
    train_button.style.button_color = '#0080ea'
    close_button.on_click(close_button_click)

    # Display screen
    display(screen)

    # Need to return empty model first
    return model


def train_regressor(df, model_name, hostname):
    """ Trains an xplainable classifier via a simple GUI.

    Args:
        df (pandas.DataFrame): Training data including target
        model_name (str): A unique name for the model
        hostname (str): Base url of host machine

    Raises:
        RuntimeError: When model fails to fit

    Returns:
        xplainale.models.XClassifier: The trained model
    """

    # This allows widgets to show full label text
    style = {'description_width': 'initial'}

    # Instantiate widget output
    output = widgets.Output()
    model = XRegressor(model_name=model_name, hostname=hostname)

    # HEADER
    logo = open('xplainable/_img/white_bg.png', 'rb').read()
    logo_display = widgets.Image(
        value=logo, format='png', width=50, height=50)

    header_title = widgets.HTML(
        f"<h2>\u00a0\u00a0\u00a0Model: {model_name}</h2>",
        layout=widgets.Layout(height='auto'))

    header = widgets.VBox(
        [widgets.HBox([widgets.VBox([logo_display]), header_title])])

    # TAB 1
    # Column A
    title_a1 = widgets.HTML(
        f"<h4>Target</h4>", layout=widgets.Layout(height='auto'))

    title_a2 = widgets.HTML(
        f"<h4>ID Column(s)</h4>", layout=widgets.Layout(height='auto'))

    target = widgets.Dropdown(options=df.columns)
    id_columns = widgets.SelectMultiple(options=df.columns, style=style)

    colA = widgets.VBox([title_a1, target, title_a2, id_columns])

    # Column B
    title_b = widgets.HTML(
        f"<h4>Parameters (for initial fit)</h4>",
        layout=widgets.Layout(height='auto'))

    max_depth = widgets.IntSlider(
        value=25, min=2, max=100, step=1, description='max_depth:',
        style=style)

    min_leaf_size = widgets.FloatSlider(
        value=0.01, min=0.001, max=0.2, step=0.001, readout_format='.3f',
        description='min_leaf_size:', style=style)

    min_info_gain = widgets.FloatSlider(value=0.015, min=0.001, max=0.2,\
        step=0.001, readout_format='.3f', description='min_info_gain:',
        style=style)

    bin_alpha = widgets.FloatSlider(value=0.2, min=0.01, max=0.5, step=0.01,
    description='bin_alpha:', style=style)

    tail_sensitivity = widgets.FloatSlider(value=1.1, min=1.0, max=1.5,
    step=0.01, description='tail_sensitivity:', style=style)

    colB = widgets.VBox([
        title_b, max_depth, min_leaf_size, min_info_gain,
        bin_alpha, tail_sensitivity])

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


    # Column 1
    title_1 = widgets.HTML(
        f"<h4>Evolve</h4>", layout=widgets.Layout(height='auto'))

    mutations = widgets.IntSlider(
        value=100, min=5, max=200, step=5, description='mutations:', style=style)

    generations = widgets.IntSlider(
        value=50, min=5, max=200, step=5, description='generations:', style=style)

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

    # Column 2
    title_2 = widgets.HTML(f"<h4>Tighten</h4>", layout=widgets.Layout(height='auto'))
    add_button_tighten = widgets.Button(description="Add Stage",icon='plus')
    add_button_tighten.style.button_color = '#12b980'
    add_button_tighten.on_click(add_button_tighten_clicked)

    iterations = widgets.IntSlider(
        value=100, min=5, max=200, step=5, description='iterations:', style=style)

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

    # PIPELINE
    title_3 = widgets.HTML(
        f"<h4>Optimisation Pipeline</h4><p>Select from optimisation tab</p>",
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
    col3 = widgets.VBox([title_3, pipeline, drop_button])

    # Build body
    body = widgets.HBox([tabs, col3])

    # FOOTER
    # Clear output on click
    def close_button_clicked(_):
        clear_output()

    # Train model on click
    def train_button_clicked(b):
        
        model = b.model
        model.max_depth = max_depth.value
        model.min_leaf_size = min_leaf_size.value
        model.min_info_gain = min_info_gain.value
        model.bin_alpha = bin_alpha.value
        model.tail_sensitivity = tail_sensitivity.value
        #model.prediction_range = prediction_range.value
        model._layers = [
            i.split(":")[1].strip() for i in list(pipeline.options) if i != '']

        with output:

            header.close()
            body.close()
            footer.close()

            try:
                clear_output()
                X, y = df.drop(
                    columns=[target.value]), df[target.value]

                model.fit(X, y, list(id_columns.value))

            except Exception as e:
                clear_output()
                raise RuntimeError(e)

    divider = widgets.HTML(
        f'<hr class="solid">', layout=widgets.Layout(height='auto'))

    #  Create buttons and listen for clicks
    train_button = TrainButton(description="Train", icon='bolt', model=model)
    train_button.style.button_color = '#0080ea'
    train_button.on_click(train_button_clicked)

    close_button = widgets.Button(description='Close', icon='window-close')
    close_button.on_click(close_button_clicked)

    # Build footer
    footer = widgets.VBox([divider, widgets.HBox([train_button, close_button])])

    # Build final screen
    screen = widgets.VBox([header, body, footer, output])

    # Display the GUI
    display(screen)

    # Return blank model await training
    return model

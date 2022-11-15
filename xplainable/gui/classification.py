from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.models.classification import XClassifier
from xplainable.utils import TrainButton, ping_server
from xplainable.quality import XScan
import xplainable


def classifier(df, model_name, model_description=''):
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
    model = XClassifier(
        model_name=model_name,
        model_description=model_description)

    # HEADER
    logo = open('xplainable/_img/logo.png', 'rb').read()
    logo_display = widgets.Image(
        value=logo, format='png', width=50, height=50)
    logo_display.layout = widgets.Layout(margin='15px 25px 15px 15px')

    header_title = widgets.HTML(f"<h3>Model: {model_name}&nbsp&nbsp</h3>")
    header_title.layout = widgets.Layout(margin='10px 0 0 0')

    divider = widgets.HTML(
        f'<hr class="solid">', layout=widgets.Layout(height='auto'))

    connection_status = widgets.HTML(f"<h5><font color='red'>[offline]</h5>")
    connection_status.layout = widgets.Layout(margin='10px 0 0 0')

    connection_status_button = widgets.Button(description="offline")
    connection_status_button.layout = widgets.Layout(margin='10px 0 0 0')
    connection_status_button.style = {
            "button_color": 'red',
            "text_color": 'white'
            }

    header = widgets.VBox(
        [widgets.HBox([widgets.VBox([logo_display]), header_title, connection_status_button])])

    # COLUMN 1
    col1a = widgets.HTML(
        f"<h5>Target</h5>", layout=widgets.Layout(height='auto'))

    id_title = widgets.HTML(f"<h5>ID Column (0 selected)</h5>")

    possible_targets = [None] + [i for i in df.columns if df[i].nunique() < 20]
    
    target = widgets.Dropdown(
        options=possible_targets,
        layout = widgets.Layout(width='200px')
        )

    # get all cols with cardinality of 1
    potential_id_cols = [None] + [col for col in df.columns if XScan._cardinality(df[col]) == 1]
    id_columns = widgets.SelectMultiple(
        options=potential_id_cols,
        style=style,
        layout = widgets.Layout(width='200px')
        )

    colA = widgets.VBox([col1a, target, id_title, id_columns])
    colA.layout = widgets.Layout(margin='0 0 0 15px')

    # COLUMN 2

    # Options to toggle param/opt view
    options = {
        True: 'flex',
        False: 'none'
    }

    # Change view on optimisation button change
    def optimise_changed(_):
        vala = optimise.value
        valb = not vala

        opt_params.layout.display = options[vala]
        std_params.layout.display = options[valb]
        optimise_metric.layout.display = options[vala]

    def target_changed(_):

        if target.value is None:
            train_button.disabled = True
        else:
            train_button.disabled = False

    optimise = widgets.Dropdown(
        value=False, options=[True, False],
        description='optimise:', style=style,
        layout = widgets.Layout(max_width='200px'))

    optimise_metrics = [
        'weighted-f1',
        'macro-f1',
        'accuracy',
        'recall',
        'precision'
    ]

    optimise_metric = widgets.Dropdown(
        value='weighted-f1',
        options=optimise_metrics,
        description='metric:',
        style=style,
        layout = widgets.Layout(max_width='200px', margin='0 0 0 10px'))

    # Hide on instantiation
    optimise_metric.layout.display = 'none'

    optimise_display = widgets.HBox([
        optimise,
        optimise_metric
    ])

    optimise.observe(optimise_changed, names=['value'])
    target.observe(target_changed, names=['value'])

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
    
    # Optimise param pickers
    n_trials = widgets.IntSlider(
        value=30, min=5, max=150, step=5, description='n_trials:',
        style=style)

    early_stopping = widgets.IntSlider(
        value=15, min=5, max=50, step=5, description='early_stopping:',
        style=style)

    # SEARCH SPACE – MAX_DEPTH
    max_depth_space = widgets.IntRangeSlider(
        value=[4, 22],
        min=2,
        max=100,
        step=1,
        description="max_depth:",
        style={'description_width': 'initial'},
        layout = widgets.Layout(min_width='350px')
    )

    max_depth_step = widgets.Dropdown(
        options=[1, 2, 5],
        layout = widgets.Layout(max_width='75px')
    )

    max_depth_space_display = widgets.HBox([max_depth_space, max_depth_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_leaf_size_space = widgets.FloatRangeSlider(
        value=[0.005, 0.08],
        min=0.005,
        max=0.2,
        step=0.005,
        description="min_leaf_size:",
        style={'description_width': 'initial'},
        readout_format='.3f',
        layout = widgets.Layout(min_width='350px')
    )

    min_leaf_size_step = widgets.Dropdown(
        options=[0.005, 0.01, 0.02],
        layout = widgets.Layout(max_width='75px')
    )

    min_leaf_size_display = widgets.HBox([min_leaf_size_space, min_leaf_size_step])

    # SEARCH SPACE – MIN_LEAF_SIZE
    min_info_gain_space = widgets.FloatRangeSlider(
        value=[0.005, 0.08],
        min=0.005,
        max=0.2,
        step=0.005,
        description="min_info_gain:",
        style={'description_width': 'initial'},
        readout_format='.3f',
        layout = widgets.Layout(min_width='350px')
    )

    min_info_gain_step = widgets.Dropdown(
        options=[0.005, 0.01, 0.02],
        layout = widgets.Layout(max_width='75px')
    )

    min_info_gain_display = widgets.HBox([min_info_gain_space, min_info_gain_step])

    std_params = widgets.VBox([
        widgets.HTML(f"<h5>Hyperparameters</h5>"),
        max_depth,
        min_leaf_size,
        min_info_gain
    ])

    opt_params = widgets.VBox([
        widgets.HTML(f"<h5>Trials</h5>"),
        n_trials,
        early_stopping,
        widgets.HTML(f"<h5>Search Space</h5>"),
        max_depth_space_display,
        min_leaf_size_display,
        min_info_gain_display
    ])

     # Set initial optimise widgets to no display
    opt_params.layout.display = 'none'

    colBParams = widgets.VBox([
        optimise_display,
        std_params,
        opt_params
        ])

    bin_alpha_header = widgets.HTML(f"<h5>Bin Alpha</h5>")
    bin_alpha = widgets.FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01)

    validation_size_header = widgets.HTML(f"<h5>Validation Size</h5>")
    validation_size = widgets.FloatSlider(value=0.2, min=0.05, max=0.5, step=0.01)

    colBSettings = widgets.VBox([bin_alpha_header, bin_alpha, validation_size_header, validation_size])

    colB = widgets.Tab([colBParams, colBSettings])
    colB.set_title(0, 'Parameters')
    colB.set_title(1, 'Settings')
    colB.layout = widgets.Layout(margin='0 0 0 15px', min_width='400px')
    
    body = widgets.HBox([colA, colB])

    # FOOTER
    train_button = TrainButton(description='Train Model', model=model, icon='bolt', disabled=True)
    close_button = widgets.Button(description='Close')

    train_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')
    close_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')

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

    def id_cols_changed(_):
        id_vals = [i for i in list(id_columns.value) if i is not None]
        id_title.value = f"<h5>ID Column ({len(id_vals)} selected)</h5>"

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

    # Train model on click
    def train_button_clicked(b):
        with outt:
        
            model = b.model         
            model.max_depth = max_depth.value
            model.min_leaf_size = min_leaf_size.value
            model.min_info_gain = min_info_gain.value
            model.bin_alpha = bin_alpha.value
            model.optimise = optimise.value
            model.n_trials = n_trials.value
            model.early_stopping = early_stopping.value
            model.validation_size = validation_size.value
            model.max_depth_space = list(max_depth_space.value) + [max_depth_step.value]
            model.min_leaf_size_space = list(min_leaf_size_space.value) + [min_leaf_size_step.value]
            model.min_info_gain_space = list(min_info_gain_space.value) + [min_info_gain_step.value]
            model.opt_metric = optimise_metric.value

            try:
                body.close()
                footer.close()
                
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
    connection_status_button.on_click(_check_connection)

    # Listen for changes
    id_columns.observe(id_cols_changed, names=['value'])

    # Display screen
    display(screen)

    # Ping server to check for connection
    _check_connection(None)

    # Need to return empty model first
    return model

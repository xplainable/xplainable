from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.models.classification import XClassifier
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
    train_button = TrainButton(description='Train Model', model=model, icon='bolt')
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

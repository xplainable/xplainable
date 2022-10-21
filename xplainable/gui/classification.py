from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.models.classification import XClassifier
from xplainable.utils import TrainButton
from xplainable.quality import XScan


def train_classifier(df, model_name, model_description=''):
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

    header_title = widgets.HTML(f"<h2>Model: {model_name}</h2>")
    header_title.layout = widgets.Layout(margin='10px 0 0 0')

    divider = widgets.HTML(
        f'<hr class="solid">', layout=widgets.Layout(height='auto'))

    header = widgets.VBox(
        [widgets.HBox([widgets.VBox([logo_display]), header_title])])

    # COLUMN 1
    col1a = widgets.HTML(
        f"<h4>Target</h4>", layout=widgets.Layout(height='auto'))

    id_title = widgets.HTML(f"<h4>ID Column (0 selected)</h4>")

    possible_targets = [None] + [i for i in df.columns if df[i].nunique() < 20]
    
    target = widgets.Dropdown(options=possible_targets)

    # get all cols with cardinality of 1
    potential_id_cols = [None] + [col for col in df.columns if XScan._cardinality(df[col]) == 1]
    id_columns = widgets.SelectMultiple(options=potential_id_cols, style=style)

    colA = widgets.VBox([col1a, target, id_title, id_columns])
    colA.layout = widgets.Layout(margin='0 0 0 15px')

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

    def target_changed(_):

        if target.value is None:
            train_button.disabled = True
        else:
            train_button.disabled = False

    # Select display from optimise dropdown
    col2 = widgets.HTML(
        f"<h4>Hyperparameters</h4>", layout=widgets.Layout(height='auto'))

    optimise = widgets.Dropdown(
        value=False, options=[True, False],
        description='optimise:', style=style)

    optimise.observe(on_change)
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

    colBParams = widgets.VBox(
        [col2, optimise, max_depth, min_leaf_size, min_info_gain,
        bin_alpha, n_trials, early_stopping])


    validation_size_header = widgets.HTML(f"<h4>Validation Size</h4>")
    validation_size = widgets.FloatSlider(value=0.2, min=0.05, max=0.5, step=0.01)

    colBSettings = widgets.VBox([validation_size_header, validation_size])

    colB = widgets.Tab([colBParams, colBSettings])
    colB.set_title(0, 'Parameters')
    colB.set_title(1, 'Settings')
    colB.layout = widgets.Layout(margin='0 0 0 15px')
    
    body = widgets.HBox([colA, colB])

    # FOOTER
    train_button = TrainButton(description='Train Model', model=model, icon='bolt', disabled=True)
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

    def id_cols_changed(_):
        id_vals = [i for i in list(id_columns.value) if i is not None]
        id_title.value = f"<h4>ID Column ({len(id_vals)} selected)</h4>"

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
            model.validation_size = validation_size.value

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

    # Listen for changes
    id_columns.observe(id_cols_changed, names=['value'])

    # Display screen
    display(screen)

    # Need to return empty model first
    return model

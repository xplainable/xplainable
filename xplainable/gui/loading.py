from datetime import datetime
from .preprocessor import Preprocessor
import xplainable
import ipywidgets as widgets
from IPython.display import display, clear_output
from xplainable.utils import *
from xplainable.preprocessing import transformers as tf

def get_time_string(dt):
        """Get time since object created"""

        delt = (datetime.utcnow() - dt)
        seconds = delt.seconds

        if delt.days > 1:
            return f'{delt.days} days ago'
        elif delt.days == 1:
            return '1 day ago'
        elif seconds == 1:
            return f'{seconds} second ago'
        elif seconds < 60:
            return f'{seconds} seconds ago'
        elif seconds < 3600:
            return f'{int(seconds / 60)} minutes ago'
        elif seconds < 7200:
            return f'{int(seconds / 60 / 60)} hour ago'
        else:
            return f'{int(seconds / 60 / 60)} hours ago'

def load_preprocessor():

    def build_transformer(stage):
        """Build transformer from metadata"""

        params = str(stage['params'])
        trans = eval(f'tf.{stage["name"]}(**{params})')

        return trans

    def on_preprocessor_change(_):
        """Loads versions when preprocessor is selected"""

        if preprocessors.value is None:
            load_button.disabled = True
            versions.options = []
            stages.options = []
            return

        matches = [i for i in preprocessor_content if i['preprocessor_name'] == preprocessors.value]
        if len(matches) > 0:
            preprocessors.preprocessor = matches[0]['preprocessor_id']

            versions_response = xplainable.client.__session__.get(
                f"{xplainable.__client__.api_hostname}/preprocessors/{preprocessors.preprocessor}/versions"
            )
            
            preprocessors.metadata = get_response_content(versions_response)

            dts = [datetime.strptime(i['created'], '%Y-%m-%dT%H:%M:%S.%f')for i in preprocessors.metadata]
            time_strings = [f'{get_time_string(dt)}' for dt in dts]
            version_options = [f"Version {i['version_id']} ({time_string})" for i, time_string in zip(preprocessors.metadata, time_strings)]
            versions.options = [None]+version_options
            versions.metadata = preprocessors.metadata

    def on_version_select(_):
        """Displays version stages on version select"""
        if versions.value is None:
            stages.options = []
            return
        selected_version = preprocessors.metadata[versions.index-1]['version_id']
        versions_response = xplainable.client.__session__.get(
                f"{xplainable.__client__.api_hostname}/preprocessors/{preprocessors.preprocessor}/versions/{selected_version}"
            )
        preprocessors.version_data = get_response_content(versions_response)

        stages.options = [f'{i}: {v["feature"]} --> {v["name"]} --> {v["params"]}' for i, v in enumerate(preprocessors.version_data['stages'])]
        load_button.disabled = False
    
    def close_button_clicked(_):
        """Clears all output"""
        clear_output()

    def load_button_clicked(_):
        """Updates preprocessor object with loaded metadata"""
        xp.preprocessor_name = preprocessors.value
        xp.pipeline.stages = [{"feature": i["feature"], "name": i["name"], "transformer": build_transformer(i)} for i in preprocessors.version_data['stages']]
        xp.state = len(xp.pipeline.stages)
        clear_output()

    # --- HEADER ---
    #logo = open('xplainable/_img/logo.png', 'rb').read()
    #logo_display = widgets.Image(
    #    value=logo, format='png', width=50, height=50)
    
    #label = open('xplainable/_img/label_load_preprocessor.png', 'rb').read()
    #label_display = widgets.Image(value=label, format='png')

    #header = widgets.HBox([logo_display, label_display])
    header = widgets.HBox([])
    header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    preprocessors_response = xplainable.client.__session__.get(
        f'{xplainable.__client__.api_hostname}/preprocessors'
    )

    selector_heading = widgets.HTML("<h4>Select Preprocessor</h4>")

    preprocessor_content = get_response_content(preprocessors_response)

    preprocessor_options = [i['preprocessor_name'] for i in preprocessor_content]

    preprocessors = PreprocessorDropdown(options=[None]+preprocessor_options)
    preprocessors.observe(on_preprocessor_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, preprocessors, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px')

    # PIPELINE
    pipeline_heading = widgets.HTML("<h4>Pipeline</h4>")

    stages = widgets.Select(options=[''])
    stages.layout = widgets.Layout(height='122px')

    pipeline = widgets.VBox([pipeline_heading, stages])

    body = widgets.HBox([selector, pipeline])

    # --- FOOTER ---
    load_button = widgets.Button(description='Load', disabled=True)
    load_button.style.button_color = '#0080ea'
    load_button.layout = widgets.Layout(margin=' 0 0 10px 25px')
    load_button.on_click(load_button_clicked)

    close_button = widgets.Button(description='Close')
    close_button.layout = widgets.Layout(margin=' 0 0 10px 10px')
    close_button.on_click(close_button_clicked)

    footer = widgets.HBox([load_button, close_button])

    # --- SCREEN ---
    screen = widgets.VBox([header, body, footer])

    display(screen)

    # Init preprocessor
    xp = Preprocessor(preprocessor_name=None)

    return xp


def load_model():

    def on_model_change(_):
        """Loads versions when model is selected"""

        if models.value is None:
            load_button.disabled = True
            versions.options = []
            stages.options = []
            return

        match = model_content[models.index-1]
        models.model = match['model_id']
        versions_response = xplainable.client.__session__.get(
            f"{xplainable.__client__.api_hostname}/models/{models.model}/versions"
        )
        models.metadata = get_response_content(versions_response)
        dts = [datetime.strptime(i['train_timestamp'], '%Y-%m-%dT%H:%M:%S.%f')for i in models.metadata]
        time_strings = [f'{get_time_string(dt)}' for dt in dts]
        version_options = [f"Version {i['version_id']} ({time_string})" for i, time_string in zip(models.metadata, time_strings)]
        versions.options = [None]+version_options
        versions.metadata = models.metadata
        model.model_type = match['model_type']

    def on_version_select(_):
        """Displays version stages on version select"""
        stages.options = []
        if versions.value is None:
            load_button.disabled = True
            return

        selected_version = models.metadata[versions.index-1]['version_id']
        summary_data = [f"VALIDATION RESULTS"]
        if model.model_type == 'binary_classification':
            for metric in ['Accuracy', 'F1', 'Recall', 'Precision']:
                val = models.metadata[versions.index-1]['evaluation']['prob_eval'][metric]
                summary_data.append(f"{metric}: {round(val*100, 2)}%")
        elif model.model_type == 'regression':
            for i, v in models.metadata[versions.index-1]['evaluation'].items():
                summary_data.append(f"{i}: {v}")

        load_button.disabled = False
        stages.options = summary_data
    
    def close_button_clicked(_):
        """Clears all output"""
        clear_output()

    def load_button_clicked(b):
        """Updates model object with loaded metadata"""
        selected_version = models.metadata[versions.index-1]['version_id']
        model.model = xplainable.__client__.load_model(models.model, selected_version)

        clear_output()

    # --- HEADER ---
    logo = open('xplainable/_img/logo.png', 'rb').read()
    logo_display = widgets.Image(
        value=logo, format='png', width=50, height=50)
    
    label = open('xplainable/_img/label_load_model.png', 'rb').read()
    label_display = widgets.Image(value=label, format='png')

    header = widgets.HBox([logo_display, label_display])
    header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    models_response = xplainable.client.__session__.get(
        f'{xplainable.__client__.api_hostname}/models'
    )

    selector_heading = widgets.HTML("<h4>Select Model</h4>")

    model_content = get_response_content(models_response)
    model_options = [f"{i['model_name']} [{i['model_type']}]" for i in model_content]

    models = widgets.Dropdown(options=[None]+model_options)
    models.observe(on_model_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, models, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px')

    # SUMMARY
    summary_heading = widgets.HTML("<h4>Summary</h4>")

    stages = widgets.Select(options=[''])
    stages.layout = widgets.Layout(height='122px')

    summary = widgets.VBox([summary_heading, stages])

    body = widgets.HBox([selector, summary])

    # --- FOOTER ---
    class Model:
        model=None
        model_type=None
        
    model = Model()
    load_button = TrainButton(description='Load', model=model, disabled=True)
    load_button.style.button_color = '#0080ea'
    load_button.layout = widgets.Layout(margin=' 0 0 10px 25px')
    load_button.on_click(load_button_clicked)

    close_button = widgets.Button(description='Close')
    close_button.layout = widgets.Layout(margin=' 0 0 10px 10px')
    close_button.on_click(close_button_clicked)

    footer = widgets.HBox([load_button, close_button])

    # --- SCREEN ---
    screen = widgets.VBox([header, body, footer])

    display(screen)

    return model

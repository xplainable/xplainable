from datetime import datetime
from .preprocessor import Preprocessor
import xplainable
import ipywidgets as widgets
from IPython.display import display, clear_output
from xplainable.utils import *
from xplainable.preprocessing import transformers as tf
from xplainable.gui.displays import BarGroup

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

        matches = [
            i[0] for i in preprocessor_content if i[1] == preprocessors.value]
            
        if len(matches) > 0:
            preprocessors.preprocessor = matches[0]

            versions_response = xplainable.client.__session__.get(
                f"{xplainable.client.hostname}/v1/preprocessors/\
                    {preprocessors.preprocessor}/versions")
            
            preprocessors.metadata = get_response_content(versions_response)

            dts = [datetime.strptime(i[1], '%Y-%m-%dT%H:%M:%S.%f') \
                for i in preprocessors.metadata]

            time_strings = [f'{get_time_string(dt)}' for dt in dts]

            version_options = [
                f"Version {i[0]} ({time_string})" for i, time_string in \
                    zip(preprocessors.metadata, time_strings)]

            versions.options = [None]+version_options
            versions.metadata = preprocessors.metadata

    def on_version_select(_):
        """Displays version stages on version select"""
        if versions.value is None:
            stages.options = []
            return
        selected_version = preprocessors.metadata[versions.index-1][0]
        versions_response = xplainable.client.__session__.get(
                f"{xplainable.client.hostname}/v1/preprocessors/\
                    {preprocessors.preprocessor}/versions/{selected_version}"
            )
        preprocessors.version_data = get_response_content(versions_response)

        stages.options = [
            f'{i}: {v["feature"]} --> {v["name"]} --> {v["params"]}' for \
                i, v in enumerate(preprocessors.version_data['stages'])]

        load_button.disabled = False
    
    def close_button_clicked(_):
        """Clears all output"""
        clear_output()

    def load_button_clicked(_):
        """Updates preprocessor object with loaded metadata"""
        xp.preprocessor_name = preprocessors.value
        xp.pipeline.stages = [{"feature": i["feature"], "name": i["name"], \
            "transformer": build_transformer(i)} for i \
                in preprocessors.version_data['stages']]

        xp.state = len(xp.pipeline.stages)
        clear_output()

    # --- HEADER ---
    header = widgets.HBox([])
    header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    preprocessors_response = xplainable.client.__session__.get(
        f'{xplainable.client.hostname}/v1/preprocessors'
    )

    selector_heading = widgets.HTML("<h4>Select Preprocessor</h4>")

    preprocessor_content = get_response_content(preprocessors_response)

    preprocessor_options = [i[1] for i in preprocessor_content]

    preprocessors = PreprocessorDropdown(options=[None]+preprocessor_options)
    preprocessors.observe(on_preprocessor_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, preprocessors, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px')

    # PIPELINE
    pipeline_heading = widgets.HTML("<h4>Pipeline</h4>")

    stages = widgets.Select(options=[''])
    stages.layout = widgets.Layout(height='122px', width='500px')

    pipeline = widgets.VBox([pipeline_heading, stages])

    body = widgets.HBox([selector, pipeline])

    # --- FOOTER ---
    load_button = widgets.Button(description='Load', disabled=True)
    load_button.style = {'button_color': '#0080ea', "text_color": 'white'}
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
    xp = Preprocessor(name=None)

    return xp

def load_model():

    def on_model_change(_):
        """Loads versions when model is selected"""

        if models.value is None:
            load_button.disabled = True
            versions.options = []
            return

        match = model_content[models.index-1]
        models.model = match['model_id']
        versions_response = xplainable.client.__session__.get(
            f"{xplainable.client.hostname}/v1/models/{models.model}/versions"
        )
        models.metadata = get_response_content(versions_response)
        dts = [datetime.strptime(i['created'], '%Y-%m-%dT%H:%M:%S.%f') for \
            i in models.metadata]

        time_strings = [f'{get_time_string(dt)}' for dt in dts]
        
        npartitions = [i['partitions'] for i in models.metadata]

        version_options = [
            f"Version {i['version_id']} | Partitions: {n} ({time_string})" for \
                i, time_string, n, in zip(
                    models.metadata, time_strings, npartitions)]

        versions.options = [None]+version_options
        versions.metadata = models.metadata
        model.model_type = match['model_type']

    def on_version_select(_):
        """Displays version stages on version select"""

        if versions.value is None:
            load_button.disabled = True
            partitions_dropdown_display.layout.visibility = 'hidden'
            metrics.collapse_items()
            return
        
        selected_version = models.metadata[versions.index-1]['version_id']
        partition_on = models.metadata[versions.index-1]['partition_on']
        
        partitions_response = xplainable.client.__session__.get(
            f"{xplainable.client.hostname}/v1/models/{models.model}/versions/\
                {selected_version}")
        
        model.partitions = get_response_content(partitions_response)
        partition_eval = [(i['partition'], i['evaluation']['prob_eval']) \
            for i in model.partitions]

        partitions = [f"{partition_on} == {i[0]}" if i[0] != "__dataset__" \
            else i[0] for i in partition_eval ]

        partitions_dropdown.options = partitions

        if "__dataset__" in partitions:
            partitions_dropdown.value = "__dataset__"

        update_metric_values()
        partitions_dropdown_display.layout.visibility = 'visible'
        metrics.expand_items()
        load_button.disabled = False
    
    def update_metric_values():
        if model.model_type == 'binary_classification':
            for i in model.partitions:
                v = partitions_dropdown.value
                v = v.split(" == ")[1] if " == " in v else v
                if i['partition'] == v:
                    for metric in metrics.items:
                        val = round(i['evaluation']['prob_eval'][metric]*100, 2)
                        metrics.set_value(metric, val)
                
        #elif model.model_type == 'regression':
        #    for i, v in models.metadata[versions.index-1]['evaluation'].items():
        #        summary_data.append(f"{i}: {v}")
    
    def on_partition_select(_):
        """Displays partition stages on partition select"""
        update_metric_values()
        
    def close_button_clicked(_):
        """Clears all output"""
        clear_output()

    def load_button_clicked(b):
        """Updates model object with loaded metadata"""
        selected_version = models.metadata[versions.index-1]['version_id']

        model.model = xplainable.client.load_model(
            models.model, selected_version)

        clear_output()

    header = widgets.HBox([])
    header.layout = widgets.Layout(margin = ' 5px 0 15px 25px ')

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    models_response = xplainable.client.__session__.get(
        f'{xplainable.client.hostname}/v1/models'
    )

    selector_heading = widgets.HTML("<h4>Select Model</h4>")

    model_content = get_response_content(models_response)
    model_options = [
        f"{i['model_name']} [{i['model_type']}]" for i in model_content]

    models = widgets.Dropdown(options=[None]+model_options)
    models.observe(on_model_change, names=['value'])
    
    partitions_dropdown = widgets.Dropdown(options=[])
    partitions_dropdown.observe(on_partition_select, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, models, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px')

    # SUMMARY
    summary_heading = widgets.HTML("<h4>Summary</h4>")

    metrics = BarGroup(
        items=['Accuracy', 'F1', 'Recall', 'Precision'],
        suffix='%')
        
    metrics.bar_layout.width = '175px'
    metrics.window_layout.margin = '20px 0 0 0'
    metrics.collapse_items()
    metrics.set_bar_color(color="#0080ea")
    
    partitions_dropdown_display = widgets.VBox(
        [summary_heading, partitions_dropdown],
        layout=widgets.Layout(visibility='hidden'))

    summary = widgets.VBox([partitions_dropdown_display, metrics.show()])

    body = widgets.HBox([selector, summary])

    # --- FOOTER ---
    class Model:
        model=None
        model_type=None
        partitions=None
        partition_on=None
        
    model = Model()
    load_button = TrainButton(description='Load', model=model, disabled=True)
    load_button.style = {'button_color': '#0080ea', "text_color": 'white'}
    load_button.layout = widgets.Layout(margin=' 0 0 10px 25px')
    load_button.on_click(load_button_clicked)

    close_button = widgets.Button(description='Close')
    close_button.layout = widgets.Layout(margin=' 0 0 10px 10px')
    close_button.on_click(close_button_clicked)

    footer = widgets.HBox([load_button, close_button])

    # --- SCREEN ---
    divider = widgets.HTML(f'<hr class="solid">')
    screen = widgets.VBox([header, body, divider, footer])

    display(screen)

    return model
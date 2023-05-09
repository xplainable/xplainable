from datetime import datetime
from .preprocessor import Preprocessor
import xplainable
import ipywidgets as widgets
from IPython.display import display, clear_output
from ...utils import *
from ...preprocessing import transformers as tf
from ..components import Header


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

def load_preprocessor(preprocessor_id=None, version_id=None):

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
                f"{xplainable.client.hostname}/v1/preprocessors/{preprocessors.preprocessor}/versions")
            
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
                    {preprocessors.preprocessor}/versions/{selected_version}/pipeline"
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
        xp.df_delta = preprocessors.version_data['deltas']

        xp.state = len(xp.pipeline.stages)
        clear_output()

    if preprocessor_id is not None:
        version_id = 'latest' if version_id is None else version_id
        return xplainable.client.load_preprocessor(preprocessor_id, version_id)

    # --- HEADER ---
    header = Header(title='Load Preprocessor', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    preprocessors_response = xplainable.client.__session__.get(
        f'{xplainable.client.hostname}/v1/preprocessors'
    )

    selector_heading = widgets.HTML("<h5>Select</h5>")

    preprocessor_content = get_response_content(preprocessors_response)

    preprocessor_options = [i[1] for i in preprocessor_content]

    preprocessors = PreprocessorDropdown(options=[None]+preprocessor_options)
    preprocessors.observe(on_preprocessor_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, preprocessors, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px')

    # PIPELINE
    pipeline_heading = widgets.HTML("<h5>Pipeline</h5>")

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
    screen = widgets.VBox([header.show(), body, footer])

    display(screen)

    # Init preprocessor
    xp = Preprocessor()

    return xp
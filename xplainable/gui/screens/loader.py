""" Copyright Xplainable Pty Ltd, 2023"""

from datetime import datetime
import xplainable
from ...utils import *
from ...preprocessing import transformers as xtf
from ...core.models import PartitionedRegressor, PartitionedClassifier


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

    if xplainable.client is None:
        print("visit https://www.xplainable.io/sign-up to access this service")
        return
    
    if preprocessor_id is not None:
        if version_id is None:
            raise ValueError(
                "version_id must be specified if model_id is specified")
        return xplainable.client.load_preprocessor(preprocessor_id, version_id)
    
    # Check if optional dependencies are installed
    xplainable._check_optional_dependencies_gui()
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from ..components import Header
    from .preprocessor import Preprocessor
    
    def build_transformer(stage):
        """Build transformer from metadata"""

        if not hasattr(xtf, stage["name"]):
            raise ValueError(
                f"{stage['name']} does not exist in the transformers module")

        # Get transformer function
        func = getattr(xtf, stage["name"])

        return func(**stage['params'])

    def on_preprocessor_change(_):
        """Loads versions when preprocessor is selected"""

        if preprocessors.value is None:
            load_button.disabled = True
            versions.options = []
            stages.options = []
            return

        idx = preprocessors.index-1

        preprocessors.preprocessor = preprocessor_content[idx]['preprocessor_id']

        preprocessors.metadata = xplainable.client.list_preprocessor_versions(
            preprocessors.preprocessor)

        dts = [datetime.strptime(i['created'], '%Y-%m-%dT%H:%M:%S.%f') \
            for i in preprocessors.metadata]

        time_strings = [f'{get_time_string(dt)}' for dt in dts]

        version_options = [
            f"Version {i['version_number']} ({time_string})" for i, time_string in \
                zip(preprocessors.metadata, time_strings)]

        versions.options = [None]+version_options
        versions.metadata = preprocessors.metadata

    def on_version_select(_):
        """Displays version stages on version select"""
        if versions.value is None:
            stages.options = []
            return
        selected_version = preprocessors.metadata[versions.index-1]['version_id']

        preprocessors.version_data = xplainable.client.load_preprocessor(
            preprocessors.preprocessor,
            selected_version,
            response_only=True
        )

        stages.options = [
            f'{i}: {v["feature"]} --> {v["name"]} --> {v["params"]}' for \
                i, v in enumerate(preprocessors.version_data['stages'])]

        load_button.disabled = False
    
    def close_button_clicked(_):
        """Clears all output"""
        screen.close()

    def load_button_clicked(_):
        """Updates preprocessor object with loaded metadata"""
        xp.preprocessor_name = preprocessors.value
        xp.pipeline.stages = [{"feature": i["feature"], "name": i["name"], \
            "transformer": build_transformer(i)} for i \
                in preprocessors.version_data['stages']]
        xp.df_delta = preprocessors.version_data['deltas']
        xp.state = len(xp.pipeline.stages)
        p = preprocessors.preprocessor
        v = preprocessors.metadata[versions.index-1]['version_number']
        vid = preprocessors.metadata[versions.index-1]['version_id']
        screen.close()
        print(f"Successfully loaded preprocessor {p} version {v} (version_id: {vid})")

    # --- HEADER ---
    header = Header(title='Load Preprocessor', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    selector_heading = widgets.HTML("<h5>Select</h5>")

    preprocessor_content = xplainable.client.list_preprocessors()
    preprocessor_options = [
        f"ID: {i['preprocessor_id']} | {i['preprocessor_name']}" for \
            i in preprocessor_content]

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

def load_regressor(model_id=None, version_id=None):

    if xplainable.client is None:
        print("visit https://www.xplainable.io/sign-up to access this service")
        return
    
    if model_id is not None:
        if version_id is None:
            raise ValueError(
                "version_id must be specified if model_id is specified")
        
        return xplainable.client.load_regressor(model_id, version_id)
    
    # Check if optional dependencies are installed
    xplainable._check_optional_dependencies_gui()
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from ..components import Header

    def on_model_change(_):
        """Loads versions when model is selected"""

        if models.value is None:
            load_button.disabled = True
            output_box.layout.display = 'none'
            versions.options = [None]
            return

        idx = type_indicies[models.index-1]

        models.model = model_content[idx]['model_id']

        models.metadata = xplainable.client.list_model_versions(
            models.model)

        dts = [datetime.strptime(i['created'], '%Y-%m-%dT%H:%M:%S.%f') \
            for i in models.metadata]

        time_strings = [f'{get_time_string(dt)}' for dt in dts]

        version_options = [
            f"Version {i['version_number']} ({time_string})" for i, time_string in \
                zip(models.metadata, time_strings)]

        versions.options = [None]+version_options
        versions.metadata = models.metadata

        description.value = f'<p>{model_content[idx]["model_description"]}</p>'
        target.value = f'<p>{model_content[idx]["target_name"]}</p>'

        output_box.layout.display = 'flex'

    def on_version_select(_):
        """Displays version stages on version select"""
        if versions.value is None:
            load_button.disabled = True
            return
        models.version_data = models.metadata[versions.index-1]['version_id']

        load_button.disabled = False
    
    def close_button_clicked(_):
        """Clears all output"""
        screen.close()

    def load_button_clicked(_):
        """Updates model object with loaded metadata"""
        m = models.model
        vid = models.version_data
        v = versions.metadata[versions.index-1]['version_number']
        
        xplainable.client.load_regressor(m, vid, model=partitioned_model)
        screen.close()
        print(f"Successfully loaded model {m} version {v} (version_id: {vid})")

    # --- HEADER ---
    header = Header(title='Load Regressor', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    selector_heading = widgets.HTML("<h5>Select</h5>")

    model_content = xplainable.client.list_models()

    type_indicies = [
        i for i, v in enumerate(model_content) if v['model_type'] == 'regression']
    
    model_options = [f"ID: {v['model_id']} | {v['model_name']}" for i, v in \
                     enumerate(model_content) if i in type_indicies]

    models = ModelDropdown(options=[None]+model_options)
    models.observe(on_model_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, models, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px', width='35%')

    # PIPELINE
    details_heading = widgets.HTML("<h5>Details</h5>")
    description = widgets.HTML()
    desc_output = widgets.VBox([details_heading, description])
    desc_output.layout = widgets.Layout(width='80%')
    
    target_heading = widgets.HTML("<h5>Target</h5>")
    target = widgets.HTML()
    target_output = widgets.VBox([target_heading, target])
    target_output.layout = widgets.Layout(width='20%')
    
    output_box = widgets.HBox([target_output, desc_output])
    output_box.layout = widgets.Layout(width='65%', display = 'none')
    
    body = widgets.HBox([selector, output_box])

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

    # Init model
    partitioned_model = PartitionedRegressor()

    return partitioned_model

def load_classifier(model_id=None, version_id=None):

    if xplainable.client is None:
        print("visit https://www.xplainable.io/sign-up to access this service")
        return
    
    if model_id is not None:
        if version_id is None:
            raise ValueError(
                "version_id must be specified if model_id is specified")
        
        return xplainable.client.load_classifier(model_id, version_id)
    
    # Check if optional dependencies are installed
    xplainable._check_optional_dependencies_gui()
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from ..components import Header

    def on_model_change(_):
        """Loads versions when model is selected"""

        if models.value is None:
            load_button.disabled = True
            output_box.layout.display = 'none'
            versions.options = [None]
            return

        idx = type_indicies[models.index-1]

        models.model = model_content[idx]['model_id']

        models.metadata = xplainable.client.list_model_versions(
            models.model)

        dts = [datetime.strptime(i['created'], '%Y-%m-%dT%H:%M:%S.%f') \
            for i in models.metadata]

        time_strings = [f'{get_time_string(dt)}' for dt in dts]

        version_options = [
            f"Version {i['version_number']} ({time_string})" for i, time_string in \
                zip(models.metadata, time_strings)]

        versions.options = [None]+version_options
        versions.metadata = models.metadata

        description.value = f'<p>{model_content[idx]["model_description"]}</p>'
        target.value = f'<p>{model_content[idx]["target_name"]}</p>'

        output_box.layout.display = 'flex'

    def on_version_select(_):
        """Displays version stages on version select"""
        if versions.value is None:
            load_button.disabled = True
            return
        models.version_data = models.metadata[versions.index-1]['version_id']

        load_button.disabled = False
    
    def close_button_clicked(_):
        """Clears all output"""
        screen.close()

    def load_button_clicked(_):
        """Updates model object with loaded metadata"""
        m = models.model
        vid = models.version_data
        v = versions.metadata[versions.index-1]['version_number']
        
        xplainable.client.load_classifier(m, vid, model=partitioned_model)
        screen.close()
        print(f"Successfully loaded model {m} version {v} (version_id: {vid})")

    if model_id is not None:
        return xplainable.client.load_classifier(model_id, version_id)

    # --- HEADER ---
    header = Header(title='Load Classifier', logo_size=40, font_size=18)
    header.title = {'margin': '10px 15px 0 10px'}

    # --- BODY ---
    # SELECTOR
    # Init Dropdown data
    selector_heading = widgets.HTML("<h5>Select</h5>")

    model_content = xplainable.client.list_models()
    type_indicies = [
        i for i, v in enumerate(model_content) if \
            v['model_type'] == 'binary_classification']
    
    model_options = [f"ID: {v['model_id']} | {v['model_name']}" for i, v in \
                     enumerate(model_content) if i in type_indicies]

    models = ModelDropdown(options=[None]+model_options)
    models.observe(on_model_change, names=['value'])

    versions = widgets.Select(options=[None])
    versions.observe(on_version_select, names=['value'])

    selector = widgets.VBox([selector_heading, models, versions])
    selector.layout = widgets.Layout(margin='0 20px 20px 20px', width='35%')

    # PIPELINE
    details_heading = widgets.HTML("<h5>Details</h5>")
    description = widgets.HTML()
    desc_output = widgets.VBox([details_heading, description])
    desc_output.layout = widgets.Layout(width='80%')
    
    target_heading = widgets.HTML("<h5>Target</h5>")
    target = widgets.HTML()
    target_output = widgets.VBox([target_heading, target])
    target_output.layout = widgets.Layout(width='20%')
    
    output_box = widgets.HBox([target_output, desc_output])
    output_box.layout = widgets.Layout(width='65%', display = 'none')
    
    body = widgets.HBox([selector, output_box])

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

    # Init model
    partitioned_model = PartitionedClassifier()

    return partitioned_model
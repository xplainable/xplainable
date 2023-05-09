import xplainable
from ...utils.api import get_response_content
from ...utils.xwidgets import TextInput
from ...core.optimisation.targeting import generate_ruleset
from ...quality.scanner import XScan
from ...utils.encoders import NpEncoder
import json

import ipywidgets as widgets
from IPython.display import display

from xplainable.utils.api import get_response_content
from xplainable.utils.xwidgets import TextInput

import time


class ModelPersist:
    
    def __init__(self, model, model_type,
    eval_objects=None, metadata_objects=None, df=None):

        self.model = model
        self.model_type = model_type
        self.partition_on = model.partition_on
        self.partitions = list(self.model.partitions.keys())
        self.eval_objects = eval_objects
        self.metadata_objects = metadata_objects
        self.df = df
        self.selected_model_id = None

    def save(self):

        model_name = TextInput(
            label="Name: ",
            label_type='h5',
            label_margin='2px 10px 0 0',
            box_width='220px')

        model_description = TextInput(
            label="Description: ",
            label_type='h5',
            label_margin='2px 10px 0 15px',
            box_width='350px'
            )

        loading = widgets.IntProgress(min=0, max=8, value=0)
        loading.layout = widgets.Layout(height='20px', width='100px',
        display='none', margin=' 15px 0 0 10px')
        loading.style = {'bar_color': '#0080ea'}

        loading_status = widgets.HTML(
            f'', layout=widgets.Layout(margin='12px 0 0 15px', width='250px'))

        model_details = widgets.HBox([model_name(), model_description()])
        loader_dropdown = widgets.Dropdown(options=[None])
        description_output = widgets.HTML(
            f'', layout=widgets.Layout(margin='0 0 0 15px'))

        loader = widgets.HBox(
            [loader_dropdown, description_output],
            layout=widgets.Layout(display='none'))

        buttons = widgets.ToggleButtons(options=['New Model', 'Existing Model'])
        buttons.layout = widgets.Layout(margin="0 0 20px 0")

        class Options:
            options = []

        model_options = Options()

        def get_models():

            models = xplainable.client.list_models()

            model_options.options = [
                (i['model_name'], i['model_description'], i['model_id']) for i in models if \
                    i['model_type'] == self.model_type]

            loader_dropdown.options = [None]+[
                f"ID: {i['model_id']} | {i['model_name']}" for i in models if \
                    i['model_type'] == self.model_type]

        def on_select(_):
            if buttons.index == 0:
                loader.layout.display = 'none'
                model_name.value = ''
                model_description.value = ''
                model_details.layout.display = 'flex'
                self.selected_model_id = None

            else:
                loader_dropdown.index = 0
                loader.layout.display = 'flex'
                model_details.layout.display = 'none'
                get_models()
                model_name.value = ''
                model_description.value = ''

        def model_selected(_):
            idx = loader_dropdown.index
            if idx is None:
                model_name.value = ''
                description_output.value = ''
                model_description.value = ''
                self.selected_model_id = None
                
            elif len(model_options.options) > 0:
                self.selected_model_id = model_options.options[idx-1][2]
                model_name.value = model_options.options[idx-1][0]
                desc = model_options.options[idx-1][1]
                description_output.value = f'{desc}'
                model_description.value = desc

        def close(_):
            button_display.close()
            model_details.close()
            loader.close()
            divider.close()
            action_buttons.close()

        def name_change(_):
            if model_name.value is None or model_name.value == '':
                confirm_button.disabled = True
            else:
                confirm_button.disabled = False

        def save_clicked(_):
            action_buttons.layout.display = 'none'
            button_display.layout.display = 'flex'
            apply_buttons.layout.display = 'flex'

            idx = buttons.index
            if idx == 0:
                model_details.layout.display = 'flex'
                loader.layout.display = 'none'
            else:
                model_details.layout.display = 'none'
                loader.layout.display = 'flex'

        def cancel_clicked(_):
            action_buttons.layout.display = 'flex'
            button_display.layout.display = 'none'
            model_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            loader.layout.display = 'none'

        def get_health_data():
            scanner = XScan()
            scanner.scan(self.df)

            results = []
            for i, v in scanner.profile.items():
                feature_info = {
                    "feature": i,
                    "description": None,
                    "health_info": json.dumps(v, cls=NpEncoder)
                }
                results.append(feature_info)

            return results

        def on_confirm(_):

            confirm_button.description = "Saving..."
            confirm_button.disabled = True
            loading.max = len(self.partitions) + 3
            loading.layout.display = 'flex'
            
            if buttons.index == 0:
                model_id = xplainable.client.create_model_id(
                    model_name.value,
                    model_description.value,
                    self.model.partitions['__dataset__'].target,
                    self.model_type
                )

            else:
                model_id = self.selected_model_id
            
            loading_status.value = f'Calculating dataset rules...'
            ruleset = generate_ruleset(
                self.df,
                self.model.partitions['__dataset__'].target,
                self.model.partitions['__dataset__'].id_columns
                )
            loading.value = loading.value + 1

            loading_status.value = f'Scanning data health...'
            health_info = get_health_data()
            loading.value = loading.value + 1

            loading_status.value = f'Creating model version...'
            version_id = xplainable.client.create_model_version(
                model_id, self.partition_on, ruleset, health_info)
            loading.value = loading.value + 1
            
            for part in self.partitions:
                
                loading_status.value = f'logging {part} model'

                try:
                    mdl = self.model.partitions[part]

                    partition_id = xplainable.client.log_partition(
                        self.model_type,
                        part,
                        mdl,
                        model_id,
                        version_id,
                        self.eval_objects[part],
                        self.metadata_objects[part]
                    )
                except Exception as e:
                    print(e)
                    loading_status.value = f'something went wrong'
                    confirm_button.description = "Error"
                    confirm_button.style.button_color = '#e14067'
                    time.sleep(0.5)
                    break


                # Increment loader after logging partition
                loading.value = loading.value + 1

            loading_status.value = ''
            loading.layout.display = 'none'
            loading.value = 0

            if confirm_button.description != "Error":
                confirm_button.description = "Saved"
                confirm_button.style.button_color = '#12b980'

            time.sleep(0.5)

            button_display.layout.display = 'none'
            model_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            loader.layout.display = 'none'
            action_buttons.layout.display = 'flex'

            confirm_button.description = "Confirm"
            confirm_button.style.button_color = '#0080ea'
            confirm_button.disabled = False

        buttons.observe(on_select, names=['value'])
        loader_dropdown.observe(model_selected, names=['value'])
        button_display = widgets.HBox([buttons])

        divider = widgets.HTML(f'<hr class="solid">')
        save_button = widgets.Button(description="Save", disabled=False)
        save_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        save_button.style = {
                "button_color": '#0080ea',
                "text_color": 'white'
                }
        save_button.on_click(save_clicked)

        cancel_button = widgets.Button(description="Cancel", disabled=False)
        cancel_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        cancel_button.style = {
                "button_color": '#e14067',
                "text_color": 'white'
                }
        cancel_button.on_click(cancel_clicked)

        confirm_button = widgets.Button(description="Confirm", disabled=True)
        confirm_button.on_click(on_confirm)
        close_button = widgets.Button(description="Close")
        close_button.on_click(close)

        close_button.layout = widgets.Layout(margin=' 10px 0 10px 0')
        close_button.style = {
                "button_color": '#e14067',
                "text_color": 'white'
                }

        confirm_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        confirm_button.style = {
                "button_color": '#0080ea',
                "text_color": 'white'
                }

        apply_buttons = widgets.HBox([cancel_button, confirm_button, loading, loading_status])

        model_name.w_text_input.observe(name_change, names=['value'])

        action_buttons = widgets.HBox([close_button, save_button])
        
        screen = widgets.VBox([action_buttons, button_display, model_details, loader, apply_buttons])
        
        action_buttons.layout.display = 'flex'
        button_display.layout.display = 'none'
        model_details.layout.display = 'none'
        apply_buttons.layout.display = 'none'

        return screen


def set_preprocessor_details(preprocessor):

    preprocessor_name = TextInput(
        label="Preprocessor name: ",
        label_type='h5',
        label_margin='2px 10px 0 0',
        box_width='220px')

    preprocessor_description = TextInput(
        label="Preprocessor description: ",
        label_type='h5',
        label_margin='2px 10px 0 15px',
        box_width='350px'
        )
    
    preprocessor_details = widgets.HBox(
        [preprocessor_name(), preprocessor_description()]
        )

    loader_dropdown = widgets.Dropdown(options=[None])
    description_output = widgets.HTML(
        f'', layout=widgets.Layout(margin='0 0 0 15px'))

    loader = widgets.HBox(
        [loader_dropdown, description_output],
        layout=widgets.Layout(display='none'))

    buttons = widgets.ToggleButtons(options=['New', 'Existing'])
    buttons.layout = widgets.Layout(margin="0 0 20px 0")

    class Options:
        options = []
    
    preprocessor_options = Options()

    def get_preprocessors():
        preprocessor_response = xplainable.client.__session__.get(
            f'{xplainable.client.hostname}/v1/preprocessors'
        )
        
        preprocessors = get_response_content(preprocessor_response)
        
        preprocessor_options.options = preprocessors

        loader_dropdown.options = [None]+[i[1] for i in preprocessors]

    def on_select(_):
        if buttons.index == 1:
            preprocessor_name.hide()
            preprocessor_description.hide()
            loader_dropdown.index = 0
            loader.layout.display = 'flex'
            get_preprocessors()
            preprocessor_name.value = ''
            preprocessor_description.value = ''
        else:
            loader.layout.display = 'none'
            preprocessor_name.value = ''
            preprocessor_description.value = ''
            preprocessor_name.show()
            preprocessor_description.show()
            
    def _selected(_):
        idx = loader_dropdown.index
        if idx is None:
            preprocessor_name.value = ''
            description_output.value = ''
            preprocessor_description.value = ''
        elif len(preprocessor_options.options) > 0:
            preprocessor_name.value = preprocessor_options.options[idx-1][1]
            desc = preprocessor_options.options[idx-1][2]
            description_output.value = f'{desc}'
            preprocessor_description.value = desc

    def name_change(_):
        preprocessor.preprocessor_name = preprocessor_name.value
        
    def description_change(_):
        preprocessor.description = preprocessor_description.value

    buttons.observe(on_select, names=['value'])
    loader_dropdown.observe(_selected, names=['value'])
    button_display = widgets.HBox([buttons])

    divider = widgets.HTML(f'<hr class="solid">')

    preprocessor_name.w_text_input.observe(name_change, names=['value'])
    preprocessor_description.w_text_input.observe(description_change, names=['value'])

    screen = widgets.VBox([button_display, preprocessor_details, loader, divider])

    display(screen)
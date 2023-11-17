""" Copyright Xplainable Pty Ltd, 2023"""

import xplainable
from ...utils.xwidgets import TextInput
import time

import ipywidgets as widgets
from xplainable.utils.xwidgets import TextInput
from IPython.display import display, clear_output


class ModelPersist:
    
    def __init__(self, model, model_type, X, y):

        self.model = model
        self.model_type = model_type
        self.partition_on = model.partition_on
        self.partitions = list(self.model.partitions.keys())
        self.X = X
        self.y = y
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
        model_loader_dropdown = widgets.Dropdown(options=[None])
        description_output = widgets.HTML(
            f'', layout=widgets.Layout(margin='0 0 0 15px'))

        model_loader = widgets.HBox(
            [model_loader_dropdown, description_output],
            layout=widgets.Layout(display='none'))

        buttons = widgets.ToggleButtons(options=['New Model', 'Existing Model'])
        buttons.layout = widgets.Layout(margin="0 0 20px 0")

        class Options:
            options = []

        model_options = Options()

        def get_models():

            models = xplainable.client.list_models()

            model_options.options = [
                (i['model_name'], i['model_description'], i['model_id']) for i \
                    in models if i['model_type'] == self.model_type]

            model_loader_dropdown.options = [None]+[
                f"ID: {i['model_id']} | {i['model_name']}" for i in models if \
                    i['model_type'] == self.model_type]

        def on_select(_):
            if buttons.index == 0:
                model_loader.layout.display = 'none'
                model_name.value = ''
                model_description.value = ''
                model_details.layout.display = 'flex'
                self.selected_model_id = None

            else:
                model_loader_dropdown.index = 0
                model_loader.layout.display = 'flex'
                model_details.layout.display = 'none'
                get_models()
                model_name.value = ''
                model_description.value = ''

        def model_selected(_):
            idx = model_loader_dropdown.index
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
            screen.close()

        def name_change(_):
            if model_name.value is None or model_name.value == '':
                confirm_button.disabled = True
            else:
                confirm_button.disabled = False

        def save_clicked(_):

            if xplainable.client is None:
                print("visit https://www.xplainable.io/sign-up to access this service")
                save_button.disabled = True
                return

            action_buttons.layout.display = 'none'
            button_display.layout.display = 'flex'
            apply_buttons.layout.display = 'flex'

            idx = buttons.index
            if idx == 0:
                model_details.layout.display = 'flex'
                model_loader.layout.display = 'none'
            else:
                model_details.layout.display = 'none'
                model_loader.layout.display = 'flex'

        def cancel_clicked(_):
            action_buttons.layout.display = 'flex'
            button_display.layout.display = 'none'
            model_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            model_loader.layout.display = 'none'

        def catch_errors(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    with error_output:
                        clear_output(wait=True)
                        print("Failed. \n")
                        print(e)
                        display(clear_error_button)

                    confirm_button.description = "Error"
                    confirm_button.style.button_color = '#e14067'
                    loading.value = 0
                    loading.layout.display = 'none'
                    loading_status.value = ''

                    time.sleep(1)
                    confirm_button.description = "Try again"
                    confirm_button.style.button_color = '#0080ea'
                    confirm_button.disabled = False
                    return 
            return wrapper

        @catch_errors
        def on_confirm(_):

            confirm_button.description = "Saving..."
            confirm_button.disabled = True
            loading.max = 2
            loading.layout.display = 'flex'
            
            if buttons.index == 0:
                model_id = xplainable.client.create_model_id(
                    self.model,
                    model_name.value,
                    model_description.value
                )

            else:
                model_id = self.selected_model_id
            
            loading.value = loading.value + 1

            loading_status.value = f'Creating model version...'
            
            version_id = xplainable.client.create_model_version(
                self.model,
                model_id,
                self.X,
                self.y
                )
            
            loading.value = loading.value + 1
            
            loading_status.value = ''
            loading.layout.display = 'none'
            loading.value = 0

            confirm_button.description = "Saved"
            confirm_button.style.button_color = '#12b980'

            time.sleep(0.5)

            button_display.layout.display = 'none'
            model_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            model_loader.layout.display = 'none'
            action_buttons.layout.display = 'flex'

            confirm_button.description = "Confirm"
            confirm_button.style.button_color = '#0080ea'
            confirm_button.disabled = False

        buttons.observe(on_select, names=['value'])
        model_loader_dropdown.observe(model_selected, names=['value'])
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

        apply_buttons = widgets.HBox(
            [cancel_button, confirm_button, loading, loading_status])

        model_name.w_text_input.observe(name_change, names=['value'])

        action_buttons = widgets.HBox([close_button, save_button])

        error_output = widgets.Output()

        def clear_error(_):
            error_output.clear_output()

        clear_error_button = widgets.Button(description="Clear")
        clear_error_button.on_click(clear_error)
        
        screen = widgets.VBox([
            action_buttons,
            button_display,
            model_details,
            model_loader,
            apply_buttons,
            error_output
            ])
        
        action_buttons.layout.display = 'flex'
        button_display.layout.display = 'none'
        model_details.layout.display = 'none'
        apply_buttons.layout.display = 'none'

        return screen


class PreprocessorPersist:
    
    def __init__(self, preprocessor):

        self.preprocessor = preprocessor
        self.selected_preprocessor_id = None

    def save(self):

        preprocessor_name = TextInput(
            label="Name: ",
            label_type='h5',
            label_margin='2px 10px 0 0',
            box_width='220px')

        preprocessor_description = TextInput(
            label="Description: ",
            label_type='h5',
            label_margin='2px 10px 0 15px',
            box_width='350px'
            )

        preprocessor_details = widgets.HBox(
            [preprocessor_name(), preprocessor_description()])
        
        preprocessor_loader_dropdown = widgets.Dropdown(options=[None])

        description_output = widgets.HTML(
            f'', layout=widgets.Layout(margin='0 0 0 15px'))

        preprocessor_loader = widgets.HBox(
            [preprocessor_loader_dropdown, description_output],
            layout=widgets.Layout(display='none'))

        buttons = widgets.ToggleButtons(
            options=['New preprocessor', 'Existing preprocessor'])
        
        buttons.layout = widgets.Layout(margin="0 0 20px 0")

        class Options:
            options = []

        preprocessor_options = Options()

        def get_preprocessors():

            preprocessors = xplainable.client.list_preprocessors()

            preprocessor_options.options = [
                (i['preprocessor_name'], i['preprocessor_description'], \
                 i['preprocessor_id']) for i in preprocessors]

            preprocessor_loader_dropdown.options = [None]+[
                f"ID: {i['preprocessor_id']} | {i['preprocessor_name']}" for \
                    i in preprocessors]

        def on_select(_):
            if buttons.index == 0:
                preprocessor_loader.layout.display = 'none'
                preprocessor_name.value = ''
                preprocessor_description.value = ''
                preprocessor_details.layout.display = 'flex'
                self.selected_preprocessor_id = None

            else:
                preprocessor_loader_dropdown.index = 0
                preprocessor_loader.layout.display = 'flex'
                preprocessor_details.layout.display = 'none'
                get_preprocessors()
                preprocessor_name.value = ''
                preprocessor_description.value = ''

        def preprocessor_selected(_):
            idx = preprocessor_loader_dropdown.index
            if idx is None:
                preprocessor_name.value = ''
                description_output.value = ''
                preprocessor_description.value = ''
                self.selected_preprocessor_id = None
                
            elif len(preprocessor_options.options) > 0:
                self.selected_preprocessor_id = preprocessor_options.options[
                    idx-1][2]
                
                preprocessor_name.value = preprocessor_options.options[idx-1][0]
                desc = preprocessor_options.options[idx-1][1]
                description_output.value = f'{desc}'
                preprocessor_description.value = desc

        def close(_):
            screen.close()

        def name_change(_):
            if preprocessor_name.value is None or preprocessor_name.value == '':
                confirm_button.disabled = True
            else:
                confirm_button.disabled = False

        def save_clicked(_):

            if xplainable.client is None:
                print("visit https://www.xplainable.io/sign-up to access this service")
                save_button.disabled = True
                return

            if len(self.preprocessor.pipeline.stages) == 0:
                save_button.disabled = True
                save_button.style.button_color = "#e14067"
                save_button.description = "Empty pipeline"
                time.sleep(2)
                save_button.description = "Save"
                save_button.style.button_color = "#0080ea"
                save_button.disabled = False
                return

            action_buttons.layout.display = 'none'
            button_display.layout.display = 'flex'
            apply_buttons.layout.display = 'flex'

            idx = buttons.index
            if idx == 0:
                preprocessor_details.layout.display = 'flex'
                preprocessor_loader.layout.display = 'none'
            else:
                preprocessor_details.layout.display = 'none'
                preprocessor_loader.layout.display = 'flex'

        def done_clicked(_):
            action_buttons.layout.display = 'flex'
            button_display.layout.display = 'none'
            preprocessor_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            preprocessor_loader.layout.display = 'none'
            
        def save_preprocessor_df(_):
            """Saves current state dataframe"""
            save_df_button.description = "Saving..."
            ts = str(int(time.time()))
            
            self.preprocessor._df_trans.to_csv(
                f'preprocessed_{ts}.csv',
                index=False)

            save_df_button.description = "Saved"
            save_df_button.disabled = True
            time.sleep(1)
            save_df_button.disabled = False
            save_df_button.description = "Save DataFrame"

        def catch_errors(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    with error_output:
                        clear_output(wait=True)
                        print("Failed. \n")
                        print(e)
                        display(clear_error_button)

                    confirm_button.description = "Error"
                    confirm_button.style.button_color = '#e14067'
                    time.sleep(1)
                    confirm_button.description = "Try again"
                    confirm_button.style.button_color = '#0080ea'
                    confirm_button.disabled = False
                    return 
            return wrapper

        @catch_errors
        def on_confirm(_):

            confirm_button.description = "Saving..."
            confirm_button.disabled = True
            
            if buttons.index == 0:
                preprocessor_id = xplainable.client.create_preprocessor_id(
                    preprocessor_name.value,
                    preprocessor_description.value
                )

            else:
                preprocessor_id = self.selected_preprocessor_id

            # Create preprocessor version
            preprocessor_id = xplainable.client.create_preprocessor_version(
                preprocessor_id,
                self.preprocessor.pipeline,
                self.preprocessor._df_trans
                )

            confirm_button.description = "Saved"
            confirm_button.style.button_color = '#12b980'

            time.sleep(0.5)

            button_display.layout.display = 'none'
            preprocessor_details.layout.display = 'none'
            apply_buttons.layout.display = 'none'
            preprocessor_loader.layout.display = 'none'
            action_buttons.layout.display = 'flex'

            confirm_button.description = "Confirm"
            confirm_button.style.button_color = '#0080ea'
            confirm_button.disabled = False

        buttons.observe(on_select, names=['value'])

        preprocessor_loader_dropdown.observe(
            preprocessor_selected, names=['value'])
        
        button_display = widgets.HBox([buttons])

        divider = widgets.HTML(f'<hr class="solid">')
        save_button = widgets.Button(description="Save", disabled=False)
        save_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        save_button.style = {
                "button_color": '#0080ea',
                "text_color": 'white'
                }
        save_button.on_click(save_clicked)

        done_button = widgets.Button(description="Cancel", disabled=False)
        done_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        done_button.style = {
                 "button_color": '#e14067',
                 "text_color": 'white'
                 }
        done_button.on_click(done_clicked)
    
        save_df_button = widgets.Button(
            description="Save as csv", disabled=False)
        
        save_df_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        save_df_button.style = {
                "button_color": '#12b980',
                "text_color": 'white'
                }
        save_df_button.on_click(save_preprocessor_df)

        confirm_button = widgets.Button(description="Confirm", disabled=True)
        confirm_button.on_click(on_confirm)
        close_button = widgets.Button(description="Close")
        close_button.on_click(close)

        close_button.layout = widgets.Layout(margin=' 10px 0 10px 0')

        confirm_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        confirm_button.style = {
                "button_color": '#0080ea',
                "text_color": 'white'
                }

        apply_buttons = widgets.HBox([done_button, confirm_button])

        preprocessor_name.w_text_input.observe(name_change, names=['value'])

        action_buttons = widgets.HBox(
            [close_button, save_df_button, save_button])
        
        # output used for more user friendly error messages
        error_output = widgets.Output()

        def clear_error(_):
            error_output.clear_output()

        clear_error_button = widgets.Button(description="Clear")
        clear_error_button.on_click(clear_error)

        screen = widgets.VBox([
            action_buttons,
            button_display,
            preprocessor_details,
            preprocessor_loader,
            apply_buttons,
            error_output
            ])
        
        action_buttons.layout.display = 'flex'
        button_display.layout.display = 'none'
        preprocessor_details.layout.display = 'none'
        apply_buttons.layout.display = 'none'

        return screen
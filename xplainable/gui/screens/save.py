import xplainable
from ...utils.api import get_response_content
from ...utils.xwidgets import TextInput

import ipywidgets as widgets
from IPython.display import display, clear_output

def save_model(model):

    model_name = TextInput(
        label="Model name: ",
        label_type='h5',
        label_margin='2px 10px 0 0',
        box_width='220px')

    model_description = TextInput(
        label="Model description: ",
        label_type='h5',
        label_margin='2px 10px 0 15px',
        box_width='350px'
        )
    
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
        models_response = xplainable.client.__session__.get(
            f'{xplainable.client.hostname}/v1/models'
        )
        
        models = get_response_content(models_response)
        
        model_options.options = [
            (i['model_name'], i['model_description']) for i in models if \
                i['model_type'] == 'binary_classification']

        loader_dropdown.options = [None]+[i['model_name'] for i in models]

    def on_select(_):
        if buttons.index == 1:
            model_name.hide()
            model_description.hide()
            loader_dropdown.index = 0
            loader.layout.display = 'flex'
            get_models()
            model_name.value = ''
            model_description.value = ''
        else:
            loader.layout.display = 'none'
            model_name.value = ''
            model_description.value = ''
            model_name.show()
            model_description.show()
            
    def model_selected(_):
        idx = loader_dropdown.index
        if idx is None:
            model_name.value = ''
            description_output.value = ''
            model_description.value = ''
        elif len(model_options.options) > 0:
            model_name.value = model_options.options[idx-1][0]
            desc = model_options.options[idx-1][1]
            description_output.value = f'{desc}'
            model_description.value = desc

    def on_confirm(_):
        print(model_name.value)
        print(model_description.value)
        
        confirm_button.description = "Saved"
        confirm_button.style.button_color = '#12b980'
        confirm_button.disabled = True

        model_description.disable()
        model_name.disable()
        buttons.disabled = True
        loader_dropdown.disabled = True

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

    buttons.observe(on_select, names=['value'])
    loader_dropdown.observe(model_selected, names=['value'])
    button_display = widgets.HBox([buttons])

    divider = widgets.HTML(f'<hr class="solid">')
    confirm_button = widgets.Button(description="Save", disabled=True)
    confirm_button.on_click(on_confirm)
    close_button = widgets.Button(description="Close")
    close_button.on_click(close)

    close_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')
    close_button.style = {
            "button_color": '#e14067',
            "text_color": 'white'
            }

    confirm_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
    confirm_button.style = {
            "button_color": '#0080ea',
            "text_color": 'white'
            }

    model_name.w_text_input.observe(name_change, names=['value'])

    action_buttons = widgets.HBox([close_button, confirm_button])

    screen = widgets.VBox([button_display, model_details, loader, divider, action_buttons])

    display(screen)

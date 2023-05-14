from traitlets import traitlets
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
import webbrowser

class LiveDF:

    def __init__(self, df):
        self.df = df

    def __call__(self):
        
        def _set_params(visualise=True):
            if visualise:
                display(self.df)

        return interactive(_set_params)


adder_element_layout = widgets.Layout(
    width='260px',
    margin='0 0 10px 0'
)
adder_element_style = {'description_width': 'initial'}

class Dropdown(widgets.Dropdown):
    """This is used to set a default max_width for adder dropdowns"""
    
    def __init__(self, *args, **kwargs):
        super(Dropdown, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class Text(widgets.Text):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(Text, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class IntText(widgets.IntText):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(IntText, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class SelectMultiple(widgets.SelectMultiple):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(SelectMultiple, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style

class TagsInput(widgets.TagsInput):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(TagsInput, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class Checkbox(widgets.Checkbox):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(Checkbox, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class ToggleButtons(widgets.ToggleButtons):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(ToggleButtons, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = {'description_width': 'initial', "button_width": "auto"}


class IntSlider(widgets.IntSlider):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(IntSlider, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class FloatSlider(widgets.FloatSlider):
    """This is used to set a default max_width for adder Text"""
    
    def __init__(self, *args, **kwargs):
        super(FloatSlider, self).__init__(*args, **kwargs)
        self.layout = adder_element_layout
        self.style = adder_element_style


class TrainButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, partitions=None, *args, **kwargs):
        super(TrainButton, self).__init__(*args, **kwargs)
        self.add_traits(partitions=traitlets.Any(partitions))


class TransformerButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, transformer=None, df=None, *args, **kwargs):
        super(TransformerButton, self).__init__(*args, **kwargs)
        self.add_traits(
            transformer=traitlets.Any(transformer),
            df=traitlets.Any(df)
            )

class IDButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, id=None, *args, **kwargs):
        super(IDButton, self).__init__(*args, **kwargs)
        self.add_traits(id=traitlets.Any(id))

class LayerButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, idx=None, layer=None, *args, **kwargs):
        super(LayerButton, self).__init__(*args, **kwargs)
        self.add_traits(idx=traitlets.Any(idx))
        self.add_traits(layer=traitlets.Any(layer))

class TransformerDropdown(widgets.Dropdown):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, transformer=None, *args, **kwargs):
        super(TransformerDropdown, self).__init__(*args, **kwargs)
        self.add_traits(transformer=traitlets.Any(transformer))


class PreprocessorDropdown(widgets.Dropdown):
    """ Dropdown that stores preprocessor state

    Args:
        model: xplainable model
    """

    def __init__(self, preprocessor=None, version=None, metadata={},\
        version_data=None, *args, **kwargs):
        super(PreprocessorDropdown, self).__init__(*args, **kwargs)
        self.add_traits(preprocessor=traitlets.Any(preprocessor))
        self.add_traits(version=traitlets.Any(version))
        self.add_traits(metadata=traitlets.Any(metadata))
        self.add_traits(version_data=traitlets.Any(version_data))


class ModelDropdown(widgets.Dropdown):
    """ Dropdown that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, model=None, version=None, metadata={}, \
        version_data=None, *args, **kwargs):
        super(ModelDropdown, self).__init__(*args, **kwargs)
        self.add_traits(model=traitlets.Any(model))
        self.add_traits(version=traitlets.Any(version))
        self.add_traits(metadata=traitlets.Any(metadata))
        self.add_traits(version_data=traitlets.Any(version_data))


class ClickResponsiveToggleButtons(widgets.ToggleButtons):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_handlers = widgets.CallbackDispatcher()
        self.on_msg(self._handle_button_msg)
        pass

    def on_click(self, callback, remove=False):
        """Register a callback to execute when the button is clicked.

        The callback will be called with one argument, the clicked button
        widget instance.

        Parameters
        ----------
        remove: bool (optional)
            Set to true to remove the callback from the list of callbacks.
        """
        self._click_handlers.register_callback(callback, remove=remove)

    def _handle_button_msg(self, _, content, buffers):
        """Handle a msg from the front-end.

        Parameters
        ----------
        content: dict
            Content of the msg.
        """
        if content.get('event', '') == 'click':
            self._click_handlers(self)

class TextInput:
    
    def __init__(self, label, label_type='h4', label_margin='0 10px 0 0', box_width='200px'):
        self.label = label
        self.label_type = label_type
        self.label_margin = label_margin
        self.box_width = box_width
        
        # Widgets
        self.w_text_input = None
        self.w_label = None
        self.w = widgets.HBox([])
        self.init()

    def init(self):
        self.w_text_input = widgets.Text()        
        self.w_label = widgets.HTML(
            f'<{self.label_type}>{self.label}</{self.label_type}>')
        
        self.w_label.layout = widgets.Layout(margin=self.label_margin)
        self.w_text_input.layout = widgets.Layout(width=self.box_width)
        self.w.children = [self.w_label, self.w_text_input]
        
    def __call__(self, ):
        
        return self.w
    
    @property
    def value(self):
        return self.w_text_input.value
    
    @value.setter
    def value(self, value):
        try:
            self.w_text_input.value = value
        except:
            pass
        
    def hide(self):
        self.w.layout.display = 'none'
    
    def show(self):
        self.w.layout.display = 'flex'

    def disable(self):
        self.w_text_input.disabled = True

    def enable(self):
        self.w_text_input.disabled = False

# linking button for offline version
offline_button = widgets.Button(description='offline')
offline_button.layout = widgets.Layout(width='75px')
offline_button.style.button_color = '#e14067'
offline_button.style.text_color = 'white'

def on_offline_button_click(b):
    webbrowser.open_new_tab('https://www.xplainable.io/sign-up')

offline_button.on_click(on_offline_button_click)

# linking button for offline version
# docs_button = widgets.Button(description='docs')
# docs_button.layout = widgets.Layout(width='75px', margin='0 15px 0 0')
# docs_button.style.button_color = '#0080ea'
# docs_button.style.text_color = 'white'

# def on_docs_button_click(b):
#     webbrowser.open_new_tab('https://docs.xplainable.io')

# docs_button.on_click(on_docs_button_click)


docs_html = """
<a href="https://docs.xplainable.io" target="_blank" rel="noopener noreferrer" style="cursor: pointer;" title="Docs">
    <svg height="35px" width="35px" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
        viewBox="0 0 512.001 512.001" xml:space="preserve">
    <path style="fill:#0080EA;" d="M439.652,512H72.348c-9.217,0-16.696-7.479-16.696-16.696V16.696C55.652,7.479,63.131,0,72.348,0
    h233.739c4.424,0,8.674,1.761,11.804,4.892l133.565,133.565c3.131,3.13,4.892,7.379,4.892,11.804v345.043
    C456.348,504.521,448.869,512,439.652,512z"/>
<path style="fill:#2860CC;" d="M317.891,4.892C314.761,1.761,310.511,0,306.087,0H256v512h183.652
    c9.217,0,16.696-7.479,16.696-16.696V150.261c0-4.424-1.761-8.674-4.892-11.804L317.891,4.892z"/>
<path style="fill:#0080EA;" d="M451.459,138.459L317.891,4.892C314.76,1.76,310.511,0,306.082,0h-16.691l0.001,150.261
    c0,9.22,7.475,16.696,16.696,16.696h150.26v-16.696C456.348,145.834,454.589,141.589,451.459,138.459z"/>
<path style="fill:#FFFFFF;" d="M272.696,411.826H139.13c-9.217,0-16.696-7.479-16.696-16.696c0-9.217,7.479-16.696,16.696-16.696
    h133.565c9.217,0,16.696,7.479,16.696,16.696C289.391,404.348,281.913,411.826,272.696,411.826z"/>
<path style="fill:#E6F3FF;" d="M272.696,378.435H256v33.391h16.696c9.217,0,16.696-7.479,16.696-16.696
    C289.391,385.913,281.913,378.435,272.696,378.435z"/>
<path style="fill:#FFFFFF;" d="M372.87,345.043H139.13c-9.217,0-16.696-7.479-16.696-16.696c0-9.217,7.479-16.696,16.696-16.696
    H372.87c9.217,0,16.696,7.479,16.696,16.696C389.565,337.565,382.087,345.043,372.87,345.043z"/>
<path style="fill:#E6F3FF;" d="M372.87,311.652H256v33.391h116.87c9.217,0,16.696-7.479,16.696-16.696
    C389.565,319.131,382.087,311.652,372.87,311.652z"/>
<path style="fill:#FFFFFF;" d="M372.87,278.261H139.13c-9.217,0-16.696-7.479-16.696-16.696c0-9.217,7.479-16.696,16.696-16.696
    H372.87c9.217,0,16.696,7.479,16.696,16.696C389.565,270.782,382.087,278.261,372.87,278.261z"/>
<path style="fill:#E6F3FF;" d="M372.87,244.87H256v33.391h116.87c9.217,0,16.696-7.479,16.696-16.696
    C389.565,252.348,382.087,244.87,372.87,244.87z"/>
</svg>
</a>
"""

docs_widget = widgets.HTML(value=docs_html)

docs_button = widgets.Box(
    [docs_widget],
    layout = widgets.Layout(margin='5px 15px 0 0')
    )
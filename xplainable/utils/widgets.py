from traitlets import traitlets
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display

class LiveDF:

    def __init__(self, df):
        self.df = df

    def __call__(self):
        
        def _set_params(visualise=True):
            if visualise:
                display(self.df)

        return interactive(_set_params)

class TrainButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, model=None, *args, **kwargs):
        super(TrainButton, self).__init__(*args, **kwargs)
        self.add_traits(model=traitlets.Any(model))


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

    def __init__(self, preprocessor=None, version=None, metadata={}, version_data=None, *args, **kwargs):
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

    def __init__(self, model=None, version=None, metadata={}, version_data=None, *args, **kwargs):
        super(ModelDropdown, self).__init__(*args, **kwargs)
        self.add_traits(model=traitlets.Any(model))
        self.add_traits(version=traitlets.Any(version))
        self.add_traits(metadata=traitlets.Any(metadata))
        self.add_traits(version_data=traitlets.Any(version_data))

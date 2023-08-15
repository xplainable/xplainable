import warnings
from packaging import version

HAS_WARNED = False

def _try_optional_dependencies_gui():
    global HAS_WARNED
    try:
        import ipywidgets
        import gradio
        from IPython.display import display
        import drawsvg
        import traitlets
        import seaborn
        import matplotlib
        import tornado

        if not HAS_WARNED:
            if version.parse(ipywidgets.__version__) < version.parse('8.0.0'):
                warnings.warn("Your version of ipywidgets may not properly"
                " render the xplainable gui. Please consider upgrading to"
                " ipywidgets>= 8.0.0")
                
            HAS_WARNED = True

        return True
    
    except ImportError:
        return False
    
def _check_critical_versions():
    """ This is implemented to ensure critical dependencies are imported."""
    
    # Tornado
    try:
        import tornado

        if version.parse(tornado.version) > version.parse('6.1'):
            warnings.warn("Your version of Tornado is greater than 6.1, which"
                          " is known to crash the Jupyter kernel when training"
                          " models. Please consider downgrading to Tornado 6.1")
            
    except ImportError:
        pass
    
    return

def _check_ipywidgets():
    try:
        import ipywidgets
        return ipywidgets
    except ImportError:
        raise ImportError(
            "Optional dependency ipywidgets not found. Please install to render"
            " output, or set verbose=False") from None

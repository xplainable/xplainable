from .client.client import Client
from .gui import *
from .client.init import initialise
from . import config
import warnings
from ._dependencies import _try_optional_dependencies_gui

__author__ = 'xplainable pty ltd'
from ._version import __version__

OPTIONAL_DEPENDENCIES_GUI = _try_optional_dependencies_gui()

def _check_optional_dependencies_gui():
    if not OPTIONAL_DEPENDENCIES_GUI:
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[gui]' to use "
                          "this feature.") from None
    
# Filter retry warnings as retries are expected and already handled
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3.connectionpool")

client = None

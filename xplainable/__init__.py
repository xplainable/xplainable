from .client._client import Client
from .gui.screens.classifier import *
from .gui.screens.preprocessor import *
from .gui.screens.loader import *
from .gui.screens.save import *
from .client.access import initialise
import warnings

__author__ = 'xplainable pty ltd'
from ._version import __version__

# Filter retry warnings as retries are expected and already handled
warnings.filterwarnings(
    "ignore", category=UserWarning, module="urllib3.connectionpool")

client = None
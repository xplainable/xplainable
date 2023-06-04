from .client.client import Client
from .gui import *
from .client.init import initialise, reinitialise
from .utils.handlers import check_critical_versions
import warnings

# Ensure critical dependency versions match
check_critical_versions()

__author__ = 'xplainable pty ltd'
from ._version import __version__

# Filter retry warnings as retries are expected and already handled
warnings.filterwarnings(
    "ignore", category=UserWarning, module="urllib3.connectionpool")

client = None
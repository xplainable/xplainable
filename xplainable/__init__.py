import warnings

__author__ = 'xplainable pty ltd'
from ._version import __version__

# Import external client functionality
try:
    from xplainable_client import Client, initialise, load_dataset, list_datasets
    client = None
except ImportError:
    # If xplainable-client is not installed, provide helpful error message
    def _client_not_available(*args, **kwargs):
        raise ImportError(
            "xplainable-client is not installed. Please install it with:\n"
            "pip install xplainable-client"
        )

    Client = _client_not_available
    initialise = _client_not_available
    load_dataset = _client_not_available
    list_datasets = _client_not_available
    client = None

# Filter retry warnings as retries are expected and already handled
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3.connectionpool")

# Export main model classes for easy access
from .core.models import XClassifier, XRegressor, PartitionedClassifier, PartitionedRegressor

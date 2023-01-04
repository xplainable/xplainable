import requests
from xplainable.client._client import Client
from xplainable.gui.classification import *
from xplainable.gui.regression import *
from xplainable.gui.preprocessor import Preprocessor
from xplainable.gui.loading import load_model, load_preprocessor
from xplainable.client.access import *

# Current Version
__version__ = "0.1.0"

client = None
from ._client import Client
import xplainable

def initialise(api_key):
    xplainable.client = Client(api_key)
    return "Initialised"
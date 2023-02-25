from ._client import Client
import xplainable
from IPython.display import clear_output
import os
from getpass import getpass
from dotenv import dotenv_values
import warnings
import logging

def create_file_if_not_exisits():
    pass

def store_api_key(api_key):
    try:
        env_file_dir = os.path.join(xplainable.__path__[0], '.env')
        if not os.path.isdir(env_file_dir):
            os.makedirs(env_file_dir)
        
        env_file_path = os.path.join(env_file_dir, '.api_key')
        
        with open(env_file_path, 'w') as env_file:
            env_file.write(f'XPLAINABLE_API_KEY={api_key}\n')

        logging.info(f"Stored API Key to {env_file_path}")
    except Exception as e:
        warnings.warn("Could not store API key. Check documentation.")
        logging.info(f"Could not store API key to {env_file_path}")

def get_api_key():
    # Determine the path to the .xp.env file
    try:
        env_file_path = os.path.join(xplainable.__path__[0], '.env', '.api_key')

        api_key = dotenv_values(env_file_path).get('XPLAINABLE_API_KEY')
        
        if api_key:
            logging.info(f"Retreived API Key from {env_file_path}")
            return api_key
        else:
            logging.info(f"Could not retreive API key from {env_file_path}")
            warnings.warn("Could not retreive API key. Check documentation.")
    
    except:
        return False

def initialise():
    
    has_set = False
    api_key = get_api_key()
    if not api_key:
        api_key = getpass("Paste a valid API Key: ")
        has_set = True
    
    try:
        xplainable.client = Client(api_key)
        store_api_key(api_key)
        clear_output()
        return "Initialised"
    except:
        clear_output()
        text = "Invalid. Paste a valid API Key: "
        text = "Time to update your API Key: " if not has_set else text
        api_key = getpass(text)
        clear_output()
        xplainable.client = Client(api_key)
        store_api_key(api_key)
        return "Initialised"

from ._client import Client
import xplainable
from IPython.display import clear_output
import os
import sys
from getpass import getpass
from dotenv import dotenv_values
import warnings
import logging
from ..gui.components import KeyValueTable, Header
from IPython.display import display, clear_output


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

        pyinf = sys.version_info
        data = {
            "xplainable version": xplainable.__version__,
            "python version": f'{pyinf.major}.{pyinf.minor}.{pyinf.micro}',
            "user": xplainable.client._user
        }

        try:
            import ipywidgets as widgets

            table = KeyValueTable(
                data,
                transpose=False,
                padding="0px 45px 0px 5px",
                table_width='auto',
                header_color='#e8e8e8',
                border_color='#dddddd',
                header_font_color='#20252d',
                cell_font_color= '#374151'
                )

            header = Header('Initialised', 30, 16)
            header.divider.layout.display = 'none'
            header.title = {'margin':'4px 0 0 8px'}
            output = widgets.VBox([header.show(), table.html_widget])
            display(output)
        except:
            return data

    except:
        clear_output()
        text = "Invalid. Paste a valid API Key: "
        text = "Time to update your API Key: " if not has_set else text
        api_key = getpass(text)
        clear_output()
        xplainable.client = Client(api_key)
        store_api_key(api_key)
        return "Initialised"

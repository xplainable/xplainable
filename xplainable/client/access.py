from ._client import Client
import xplainable
from .._version import __version__ as XP_VERSION
from IPython.display import clear_output
import sys
from getpass import getpass
from ..gui.components import KeyValueTable, Header
from IPython.display import display, clear_output
import keyring

def render_init_table(data):
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

    header = Header('Initialised', 30, 16, avatar=False)
    header.divider.layout.display = 'none'
    header.title = {'margin':'4px 0 0 8px'}
    output = widgets.VBox([header.show(), table.html_widget])
    display(output)

def initialise():

    version_info = sys.version_info
    PY_VERSION = f'{version_info.major}.{version_info.minor}.{version_info.micro}'
    
    has_set = False
    api_key = keyring.get_password('XPLAINABLE', PY_VERSION)
    if not api_key:
        api_key = getpass("Paste a valid API Key: ")
        has_set = True
    
    try:
        xplainable.client = Client(api_key)
        xplainable.client.xplainable_version = XP_VERSION
        xplainable.client.python_version = PY_VERSION

        keyring.set_password('XPLAINABLE', PY_VERSION, api_key)
        clear_output()

        data = {
            "xplainable version": XP_VERSION,
            "python version": PY_VERSION,
            "user": xplainable.client._user['username']
        }

        try:
            render_init_table(data)
        except:
            return data

    except:
        clear_output()
        text = "Invalid. Paste a valid API Key: "
        text = "Time to update your API Key: " if not has_set else text
        api_key = getpass(text)
        clear_output()

        xplainable.client = Client(api_key)
        xplainable.client.xplainable_version = XP_VERSION
        xplainable.client.python_version = PY_VERSION
        keyring.set_password('XPLAINABLE', PY_VERSION, api_key)
        
        data = {
            "xplainable version": XP_VERSION,
            "python version": PY_VERSION,
            "user": xplainable.client._user['username']
        }

        try:
            render_init_table(data)
        except:
            return data

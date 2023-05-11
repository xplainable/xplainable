from ._client import Client
import xplainable
from IPython.display import clear_output
import sys
from getpass import getpass
from ..gui.components import KeyValueTable, Header
from IPython.display import display, clear_output
import keyring

vs = sys.version_info
version = f'{vs.major}_{vs.minor}_{vs.micro}'
KEY = f'XPLAINABLE_{version}'

def initialise():
    
    has_set = False
    api_key = keyring.get_password('XPLAINABLE', version)
    if not api_key:
        api_key = getpass("Paste a valid API Key: ")
        has_set = True
    
    try:
        xplainable.client = Client(api_key)
        keyring.set_password('XPLAINABLE', version, api_key)
        clear_output()

        pyinf = sys.version_info
        data = {
            "xplainable version": xplainable.__version__,
            "python version": f'{pyinf.major}.{pyinf.minor}.{pyinf.micro}',
            "user": xplainable.client._user['username']
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

            header = Header('Initialised', 30, 16, avatar=False)
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
        keyring.set_password('XPLAINABLE', version, api_key)
        return "Initialised"

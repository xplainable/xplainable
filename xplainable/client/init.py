""" Copyright Xplainable Pty Ltd, 2023"""

from .client import Client
import xplainable
from .._version import __version__ as XP_VERSION
import sys
from getpass import getpass
from IPython.display import display, clear_output
from ..utils.exceptions import AuthenticationError

from .. import config

def _render_init_table(data):
    import ipywidgets as widgets
    from ..gui.components import KeyValueTable, Header
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

def initialise(api_key=None, hostname='https://api.xplainable.io'):
    """ Initialise the client with an API Key.

    API Keys can be generated from https://beta.xplainable.io with a valid
    account.

    Example:
        >>> import xplainable as xp
        >>> import os
        >>> xp.initialise(api_key=os.environ['XP_API_KEY'])

    Returns:
        dict: The users account information.
    """

    version_info = sys.version_info
    PY_VERSION = f'{version_info.major}.{version_info.minor}.{version_info.micro}'
    
    if not api_key:
        raise ValueError(
            'A valid API Key is required. Generate one from the xplainable app.'
            ) from None
    
    try:
        xplainable.client = Client(api_key, hostname)
        xplainable.client.xplainable_version = XP_VERSION
        xplainable.client.python_version = PY_VERSION

        data = {
            "xplainable version": XP_VERSION,
            "python version": PY_VERSION,
            "user": xplainable.client._user['username'],
            "organisation": xplainable.client._user['organisation_name'],
            "team": xplainable.client._user['team_name'],
        }

        if config.OUTPUT_TYPE == 'raw':
            return data
        
        try:
            _render_init_table(data)
        except:
            return data

    except Exception as e:
        raise e


from ...utils import ping_gateway
import xplainable
from ipywidgets import Button
import traitlets


class ConnectionButton(Button):
    def __init__(self, connected=False, *args, **kwargs):
        super(ConnectionButton, self).__init__(*args, **kwargs)
        self.add_traits(
            connected=traitlets.Bool(connected).tag(sync=True),
            link_disabled=traitlets.Bool(not connected).tag(sync=True)
            )

    def _check_connection(self):
        try:
            if ping_gateway(xplainable.client.hostname):
                self.connected = True
                self.link_disabled = False
                self.description = "Connected"
                self.style.button_color = '#12b980'
            else:
                self.connected = False
                self.link_disabled = True
                self.description = "Offline"
                self.style.button_color = '#e21c47'
        except:
            self.connected = False
            self.link_disabled = True
            self.description = "Offline"
            self.style.button_color = '#e21c47'

    
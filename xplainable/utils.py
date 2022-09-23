from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from traitlets import traitlets
import ipywidgets as widgets


class Loader:
    def __init__(self, desc="Loading...", end="Done", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ['■□□□□', '□■□□□', '□□■□□', '□□□■□', '□□□□■', '□□□■□', '□□■□□','□■□□□']
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)


class TrainButton(widgets.Button):
    """ Button that stores model state

    Args:
        model: xplainable model
    """

    def __init__(self, model=None, *args, **kwargs):
        super(TrainButton, self).__init__(*args, **kwargs)
        self.add_traits(model=traitlets.Any(model))
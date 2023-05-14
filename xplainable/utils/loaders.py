import ipywidgets as widgets
from IPython.display import display, clear_output

class LoadingPulse:
    def __init__(self, size=40, svg=None):
        self.size = size
        self.svg = svg
        self.out = widgets.Output()
        
        if svg:
            # Use the provided SVG
            self.spinner_html = f"""
            <div style="width: {size*1.5}px; height: {size*1.5}px; overflow: hidden; position: relative;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
                    <div style="animation: pulse 1s linear infinite;">
                    {svg}
                    </div>
                </div>
            </div>
            <style>
            @keyframes pulse {{
                0% {{ transform: scale(0.5); }}
                50% {{ transform: scale(1); }}
                100% {{ transform: scale(0.5); }}
            }}
            </style>
            """
        else:
            # Use the default spinner
            self.spinner_html = f"""
            <div style="width: {size}px; height: {size}px; overflow: hidden; position: relative; display: flex; justify-content: center; align-items: center;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); border: {size//10}px solid #f3f3f3; border-radius: 50%; border-top: {size//10}px solid #3498db; animation: spin 2s linear infinite;">
                </div>
            </div>
            <style>
            @keyframes spin {{
                0% {{ transform: translate(-50%, -50%) rotate(0deg); }}
                100% {{ transform: translate(-50%, -50%) rotate(360deg); }}
            }}
            </style>
            """
    def init(self):
        with self.out:
            clear_output(wait=True)
            display(widgets.HTML(self.svg))

    def start(self):
        with self.out:
            clear_output(wait=True)
            display(widgets.HTML(self.spinner_html))

    def stop(self):
        with self.out:
            clear_output(wait=True)
            display(widgets.HTML(self.svg))

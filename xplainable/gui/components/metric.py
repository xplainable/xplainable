import ipywidgets as widgets
from IPython.display import display

class Metric:
    
    def __init__(self):
        self.label = widgets.HTML('')
        self.value = widgets.HTML('')

        self.box = widgets.VBox([self.label, self.value])
        self.box.layout = widgets.Layout(display='flex',
                        flex_flow='column',
                        align_items='center',
                        width='100px')

    def set_label(self, label):

        self.label.value = f'<h5>{label}</h5>'
    
    def set_value(self, value):
        self.value.value = f'<h3>{value}</h3>'
        
    def show(self):
        display(self.box)
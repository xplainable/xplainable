import ipywidgets as widgets
from ...utils.xwidgets import IDButton

class BarGroup:
    
    def __init__(self, items=[], heading=None):
        self.items = items
        self.displays = {i: {} for i in items}
        self.heading=heading
        
        self.suffix={i: '' for i in items}
        self.prefix={i: '' for i in items}
        
        self.bar_layout = widgets.Layout(
            width='200px',
            height='20px',
            margin='6px 5px 0 0')
        
        self.label_layout = widgets.Layout(
            width='65px')
        
        self._initialise_bars()
        
        self.window = widgets.VBox(
            [widgets.HTML(f"<h4>{self.heading}<h4>")]+
            [v['display'] for i, v in self.displays.items()])
        self.window_layout = widgets.Layout()
        self.window.layout = self.window_layout

    def _initialise_bars(self):
        
        for i in self.items:
            # Generate label
            label = widgets.HTML(f"{i}: ", layout=self.label_layout)
            self.displays[i]['label'] = label
            
            # Generate Bar
            bar = widgets.FloatProgress(
                min=0, max=100, value=0, layout=self.bar_layout)
            self.displays[i]['bar'] = bar
            
            # Generate Value
            val = widgets.HTML("-")
            self.displays[i]['value'] = val
            
            # Generate Display
            self.displays[i]['display'] = widgets.HBox([label, bar, val])
            
    def show(self):
        if self.heading is None:
            self.window.children[0].layout.display = 'none'
        return self.window
    
    def set_value(self, item, value):
        self.displays[item]['bar'].value = value
        self.displays[item]['value'].value = f'{self.prefix[item]}{value}{self.suffix[item]}'
        
    def set_bounds(self, items=None, min_val=None, max_val=None):
        if items is None:
            items = self.items
        for i in items:
            self.displays[i]['bar'].min = min_val
            self.displays[i]['bar'].max = max_val
            
    def set_bar_color(self, items=None, color=None):
        if items is None:
            items = self.items
        for i in items:
            self.displays[i]['bar'].style.bar_color = color

    def set_prefix(self, items, prefix):
        if items is None:
            items = self.items
        for i in items:
            self.prefix[i] = prefix

    def set_suffix(self, items, suffix):
        if items is None:
            items = self.items
        for i in items:
            self.suffix[i] = suffix
        
    def collapse_items(self, items=None):
        if items is None:
            items = self.items
        for item in items:
            self.displays[item]['display'].layout.display = 'none'
        
    def expand_items(self, items=None):
        if items is None:
            items = self.items
        for item in items:
            self.displays[item]['display'].layout.display = 'flex'

    def add_button(self, items=None, text='button', side='right', on_click=None):
        if items is None:
            items = self.items
        for item in items:
            button = IDButton(description=text, id=item)
            button_layout = widgets.Layout(
                height='25px', width='50px', display='flex', margin='3px 0 0 10px')
            button.layout = button_layout

            if on_click is not None:
                button.on_click(on_click)

            if side == 'left':
                self.displays[item]['display'].children = (button,) + self.displays[item]['display'].children
            else:
                self.displays[item]['display'].children = self.displays[item]['display'].children + (button,)

import ipywidgets as widgets
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from ...utils.xwidgets import IDButton

class BarGroup:
    
    def __init__(self, items=[], heading=None, footer=False):
        self.items = items
        self.displays = {i: {} for i in items}
        self.heading = heading
        self.footer = footer
        
        self.suffix={i: '' for i in items}
        self.prefix={i: '' for i in items}
        
        self.bar_layout = widgets.Layout(
            width='200px',
            height='20px',
            margin='6px 5px 0 0')
        
        self.label_layout = widgets.Layout(
            width='65px')

        self.value_layout = widgets.Layout(
            width='40px')

        self.displays_layout = widgets.Layout()
        
        self._initialise_bars()

        self.window = widgets.VBox(
            [widgets.HTML(f"<h4>{self.heading}<h4>")]+
            [v['display'] for i, v in self.displays.items()]
            )

        if self.footer:
            self.footer_label = widgets.HTML(
                f"metric: ", layout=self.label_layout)

            self.footer_bar = widgets.FloatProgress(
                    min=0, max=100, value=0, layout=self.bar_layout)

            self.footer_val = widgets.HTML("-")

            self.footer_display = widgets.HBox(
                    [self.footer_label, self.footer_bar, self.footer_val],
                    layout=self.displays_layout)

            self.window.children = self.window.children + (
                widgets.HTML(f'<hr class="solid">'),
                self.footer_display
            )

        self.window_layout = widgets.Layout()
        self.window.layout = self.window_layout

    def _initialise_bars(self):
        
        for i in self.items:
            # Generate label
            label = widgets.HTML(f"{i}", layout=self.label_layout)
            self.displays[i]['label'] = label
            
            # Generate Bar
            bar = widgets.FloatProgress(
                min=0, max=100, value=0, layout=self.bar_layout)
            self.displays[i]['bar'] = bar
            
            # Generate Value
            val = widgets.HTML("-")
            val.layout = self.value_layout
            self.displays[i]['value'] = val
            
            # Generate Display
            self.displays[i]['display'] = widgets.HBox(
                [label, bar, val], layout=self.displays_layout)
            
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

    def set_prefix(self, prefix, items=None):
        if items is None:
            items = self.items
        for i in items:
            self.prefix[i] = prefix

    def set_suffix(self, suffix, items=None):
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

    def set_footer_label(self, label):
        self.footer_label.value = label

    def set_footer_bounds(self, min_val, max_val):
        self.footer_bar.min = min_val
        self.footer_bar.max = max_val
    
    def set_footer_value(self, value):
        self.footer_val.value = str(value)
        self.footer_bar.value = value

    def close(self):
        self.window.close()

    def add_button(
            self, items=None, text='button', side='right', on_click=None,
            height=25, width=50):
        if items is None:
            items = self.items
        for item in items:
            button = IDButton(description=text, id=item)
            button_layout = widgets.Layout(
                height=f'{height}px', width=f'{width}px', display='flex',
                margin='3px 0 0 10px')
            
            button.layout = button_layout

            if on_click is not None:
                button.on_click(on_click)

            if side == 'left':
                self.displays[item]['display'].children = (button,) + \
                    self.displays[item]['display'].children
            else:
                self.displays[item]['display'].children = self.displays[
                    item]['display'].children + (button,)


class IncrementalBarPlot:
    def __init__(self, output):
        self.output = output
        self.labels = []
        self.values = []
        self.width = 0.35

    def add_bar(self, label, value):
        self.labels.append(label)
        self.values.append(value)
        with self.output:
            self._update_plot()

    def _update_plot(self):
        min_height = 1
        max_height = 5
        bar_height = 0.8
        spacing = 0.1

        num_bars = len(self.labels)
        dynamic_height = max(
            min_height, min(max_height, num_bars * (bar_height + spacing)))

        fig, ax = plt.subplots()
        fig.set_size_inches(6, dynamic_height)

        x = range(num_bars)
        ax.barh(x, self.values, self.width, color='#0080ea')
        ax.set_yticks(x)
        ax.set_yticklabels(self.labels)
        ax.set_title('Partition Metrics')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        clear_output(wait=True)
        display(fig)
        plt.close(fig)

    def show(self):
        self._update_plot()
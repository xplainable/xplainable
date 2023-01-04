import ipywidgets as widgets
from IPython.display import clear_output


class ClassificationProgress():
    
    def __init__(self, partitions, opt_metric, n_features):
        self.partitions = partitions
        self.opt_metric = opt_metric
        self.n_features = n_features
        self.state = {}
        self.params = None
        self.info = {}
        self.active_partition = None
        self.info_display = None
    
    def add_partition(self, idx):
        
        partition_layout = widgets.Layout(min_width='25px')
        button_layout = widgets.Layout(
            height='25px', width='80px', visibility='hidden')
        
        desc_button = widgets.Button(
            description="details", layout=button_layout)
        
        progress_number_layout = widgets.Layout(
            margin='0 10px 0 10px', min_width='50px')
            
        progress_layout = widgets.Layout(height='20px', margin=' 6px 10px 0 0 ')
        progress_style = {"bar_color": '#0080ea'}
        
        partition_widget = widgets.HTML(
            value=f'<p>{idx}: </p>', layout=partition_layout)

        progress_widget = widgets.HTML(
            value=f'<p>waiting...</p>', layout=progress_number_layout)

        progress_bar_widget = widgets.IntProgress(
            min=0, max=100, value=0, layout=progress_layout,
            style=progress_style)

        initial_state = {
            "partition": partition_widget,
            "status": "waiting",
            "optimisation_value": 0,
            "optimisation_iterations": 0,
            "training_value": 0,
            "training_iterations": 0,
            "fold": 0,
            "folds": 0,
            "progress": progress_widget,
            "progress_bar": progress_bar_widget,
            "details": desc_button,
            "optimisation_complete": False,
            "train_complete": False,
            "best_metric": "-",
            "params": {"max_depth": 0, "min_leaf_size": 0, "min_info_gain": 0}
        }

        self.state[idx] = initial_state
        
        def toggle_button_on(idx):
            self.state[idx]['details'].style.button_color = '#0080ea'
            self.state[idx]['details'].style.text_color = 'white'
            
        def toggle_button_off(idx):
            if idx is not None:
                self.state[idx]['details'].style.button_color = '#EEEEEE'
                self.state[idx]['details'].style.text_color = 'black'
        
        def update_description(_):
            if idx == self.active_partition:
                toggle_button_off(idx)
                self.active_partition = None
            else:
                toggle_button_off(self.active_partition)
                self.active_partition = idx
                toggle_button_on(idx)
            
            self.update_param_display()

        desc_button.on_click(update_description)
        
        # Build progress bar with state elements
        progress_display = widgets.HBox([
            self.state[idx]['partition'],
            self.state[idx]['progress_bar'],
            self.state[idx]['progress'],
            self.state[idx]['details']]
        )
        
        return progress_display
    
    def generate_info_display(self):
        
        param_layout = widgets.Layout(
            height='20px', margin=' 6px 10px 0 0 ', width='200px')

        folds_layout = widgets.Layout(
            height='20px', margin=' 6px 10px 0 0 ', width='200px')

        label_layout = widgets.Layout(min_width='90px')
        param_style ={"bar_color": '#e14067'}
        folds_style ={"bar_color": '#0080ea'}
        
        info_layout = widgets.Layout(margin='0 50px 0 0')
        
        folds_bar = widgets.IntProgress(
            min=0, max=5, value=0, style=folds_style, layout=folds_layout)

        folds_label = widgets.HTML(
            value=f'<p>Fold 0: </p>', layout=label_layout)

        folds_display = widgets.HBox([folds_label, folds_bar])
        self.info = {
            "status": widgets.HTML(value=f'<p>-</p>'),
            "partition": widgets.HTML(value=f'<p>-</p>'),
            "metric": widgets.HTML(value=f'<p>-</p>'),
            "folds_bar": folds_bar,
            "folds_label": folds_label,
            "folds_display": folds_display
        }
        
        self.params = {
            "bars": {
                "max_depth": widgets.IntProgress(
                    min=2,
                    max=30,
                    value=2,
                    layout=param_layout,
                    style=param_style
                    ),
                "min_leaf_size": widgets.FloatProgress(
                    min=0.001,
                    max=0.2,
                    value=0.001,
                    layout=param_layout,
                    style=param_style
                    ),
                "min_info_gain": widgets.FloatProgress(
                    min=0.001,
                    max=0.2,
                    value=0.001,
                    layout=param_layout,
                    style=param_style
                    )
            },
            "values": {
                "max_depth": widgets.HTML(value=f'<p>-</p>'),
                "min_leaf_size": widgets.HTML(value=f'<p>-</p>'),
                "min_info_gain": widgets.HTML(value=f'<p>-</p>')
            }
        }
        
        info_heading = widgets.HTML(value=f'<h4>Info</h4>')
        status_display = widgets.VBox(
            [widgets.HTML(value=f'<h6>Status</h6>'), self.info["status"]],
            layout=info_layout)

        partition_display = widgets.VBox(
            [widgets.HTML(value=f'<h6>Partition</h6>'), self.info["partition"]],
            layout=info_layout)

        metric_display = widgets.VBox(
            [widgets.HTML(value=f'<h6>Best {self.opt_metric}</h6>'),
            self.info["metric"]])

        status_labels = widgets.HBox(
            [status_display, partition_display, metric_display])

        info_box = widgets.VBox(
            [info_heading, status_labels, folds_display],
            layout=widgets.Layout(height='150px'))
        
        
        parameters_title = widgets.HTML(value=f'<h5>Hyperparameters</h5>')
        param_display = [widgets.HBox(
            [widgets.HTML(value=f'<p>{i}</p>', layout=label_layout), v]) for \
                i, v in self.params['bars'].items()]
        
        param_display = widgets.VBox([
            widgets.HBox([
                widgets.HTML(value=f'<p>max_depth</p>', layout=label_layout),
                self.params['bars']['max_depth'],
                self.params['values']['max_depth'],
                ]),
            widgets.HBox([
                widgets.HTML(value=f'<p>min_leaf_size</p>', layout=label_layout),
                self.params['bars']['min_leaf_size'],
                self.params['values']['min_leaf_size'],
                ]),
            widgets.HBox([
                widgets.HTML(value=f'<p>min_info_gain</p>', layout=label_layout),
                self.params['bars']['min_info_gain'],
                self.params['values']['min_info_gain'],
                ])
        ])

        info_display_layout = widgets.Layout(margin='0px 0px 0px 50px')
        self.info_display = widgets.VBox(
            [info_box, parameters_title, param_display],
            layout=info_display_layout)
        
        return self.info_display

    def generate_screen(self):
        progress_title = widgets.HTML(value=f'<h4>Training</h4>')
        progress_bars = widgets.VBox([progress_title])        
        
        for i, v in enumerate(self.partitions):
            p = self.add_partition(i)
            progress_bars.children = progress_bars.children + (p,)
        
        self.generate_info_display()
        
        divider = widgets.HTML(f'<hr class="solid">')
        
        def close_button_click(_):
            clear_output()
            
        close_button = widgets.Button(description='Close')
        close_button.layout = widgets.Layout(margin=' 10px 0 10px 20px')
        close_button.on_click(close_button_click)
        
        footer = widgets.VBox([divider, widgets.HBox([close_button])])
        body = widgets.HBox([progress_bars, self.info_display])
        screen = widgets.VBox([body, footer])

        return screen
    
    def start_partition_optimisation(self, idx, iterations, folds):
        self.state[idx]['optimisation_iterations'] = iterations
        self.state[idx]['progress_bar'].max = iterations
        self.state[idx]['status'] = "optimising"
        self.info['folds_bar'].max = folds
        self.state[idx]['progress'].value = f'<p>(0/{iterations})</p>'
        self.state[idx]['details'].layout.visibility = 'visible'
        
    def start_partition_training(self, idx):
        self.state[idx]['progress_bar'].max = 100
        self.state[idx]['status'] = "fitting"
        self.state[idx]['progress'].value = f'<p>0%</p>'
        self.state[idx]['details'].layout.visibility = 'visible'

    def complete_training(self, idx):
        self.state[idx]['progress_bar'].style.bar_color = "#12b980"
        self.state[idx]['progress'].value = f'<p>done</p>'
        self.state[idx]['status'] = "done"
        self.info['folds_display'].layout.display = 'none'
    
    def update_optimisation_progress(self, idx, iteration):
        iterations = self.state[idx]['optimisation_iterations']
        self.state[idx]['progress_bar'].value = iteration
        self.state[idx]['progress'].value = f'<p>({iteration}/{iterations})</p>'
    
    def update_training_progress(self, idx, iteration):
        if self.partitions[idx] == '__dataset__':
            p = iteration / self.n_features
        else:
            p = iteration / (self.n_features - 1)
        v = int(p*100)
        self.state[idx]['progress_bar'].value = v
        self.state[idx]['progress'].value = f'<p>{v}%</p>'
        
    def update_stage(self, idx, stage):
        self.state[idx]['stage'].value = f'<p>{stage}</p>'
    
    def update_param_display(self):
        idx = self.active_partition
        if idx is None:
            self.info_display.layout.visibility = 'hidden'
            return
        self.info_display.layout.visibility = 'visible'
        self.info['status'].value = self.state[idx]['status']
        self.info['partition'].value = str(self.partitions[idx])
        self.info['metric'].value = self.state[idx]['best_metric']
        self.info['folds_bar'].value = self.state[idx]['fold']
        self.params['bars']['max_depth'].value = self.state[
            idx]['params']['max_depth']

        self.params['bars']['min_leaf_size'].value = self.state[
            idx]['params']['min_leaf_size']  

        self.params['bars']['min_info_gain'].value = self.state[
            idx]['params']['min_info_gain']

        self.params['values']['max_depth'].value = str(self.state[
            idx]['params']['max_depth'])

        self.params['values']['min_leaf_size'].value = str(self.state[
            idx]['params']['min_leaf_size'])

        self.params['values']['min_info_gain'].value = str(self.state[
            idx]['params']['min_info_gain'])
            
    def update_params(self, idx, max_depth=None, min_leaf_size=None,\
        min_info_gain=None, best_metric=None, fold=None):
        
        if max_depth:
            self.state[idx]['params']['max_depth'] = max_depth
            
        if min_leaf_size:
            self.state[idx]['params']['min_leaf_size'] = min_leaf_size
            
        if min_info_gain:
            self.state[idx]['params']['min_info_gain'] = min_info_gain
            
        if best_metric:
            self.state[idx]['best_metric'] = str(best_metric)
            
        if fold:
            self.state[idx]['fold'] = str(fold)
            self.info['folds_label'].value = f'<p>Fold {fold}: </p>'
            
        self.update_param_display()


class BarGroup:
    
    def __init__(self, items=[], suffix='', prefix=''):
        self.items = items
        self.displays = {i: {} for i in items}
        
        self.suffix=suffix
        self.prefix=prefix
        
        self.bar_layout = widgets.Layout(
            width='200px',
            height='25px',
            margin='3px 5px 0 0')
        
        self.label_layout = widgets.Layout(
            width='65px')
        
        self._initialise_bars()
        
        self.window = widgets.VBox(
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
        return self.window
    
    def set_value(self, item, value):
        self.displays[item]['bar'].value = value
        self.displays[item]['value'].value = f'{self.prefix}{value}{self.suffix}'
        
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

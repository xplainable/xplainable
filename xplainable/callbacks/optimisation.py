
from ..gui.components.bars import BarGroup


class RegressionCallback:
    
    def __init__(self, network):
        self.network = network
        self.init()
        
    def init(self):
        self.items = [f'Layer {i} ({c.__class__.__name__})' for \
                      i, c in enumerate(self.network.future_layers, start=1)]
        
        self.group = BarGroup(items=self.items, heading='Pipeline', footer=True)
        self.group.label_layout.width = '120px'
        for i, item in enumerate(self.items):
            if self.network.future_layers[i].__class__.__name__ == 'Tighten':
                limit = self.network.future_layers[i].__dict__['iterations']

            elif self.network.future_layers[i].__class__.__name__ == 'Evolve':
                limit = self.network.future_layers[i].__dict__['generations']

            self.group.set_bounds(items=[item], min_val=0, max_val=limit)
            self.group.set_suffix(suffix=f'/{limit}', items=[item])
        
    def set_value(self, idx, value):
        item = self.items[idx]
        self.group.set_value(item, value)

    def set_metric_bounds(self, min_val, max_val):
        self.group.set_footer_bounds(min_val, max_val)
        
    def set_metric(self, metric, value):
        self.group.set_footer_label(metric)
        self.group.set_footer_value(value)
        
    def stopped_early(self, idx):
        item = self.items[idx]
        self.group.set_bar_color(items=[item], color='#fbc051')
        
    def finalise_bar(self, idx):
        item = self.items[idx]
        self.group.set_bar_color(items=[item], color='#12b980')

    def close(self):
        self.group.close()

class OptCallback():

    def __init__(self, progress, params):
        self.progress = progress
        self.params = params

    def fold(self, fold):
        self.progress.set_value(item='fold', value=fold)

    def iteration(self, i):
        self.progress.set_value(item='iteration', value=i)

    def metric(self, v):
        self.progress.set_value(item='best', value=v)

    def update_params(
            self, max_depth, min_leaf_size, min_info_gain, weight, power_degree,
            sigmoid_exponent, *args, **kwargs):
        
        self.params.set_value(item='max_depth', value=int(max_depth))

        self.params.set_value(
            item='min_leaf_size', value=round(min_leaf_size, 4))
        
        self.params.set_value(
            item='min_info_gain', value=round(min_info_gain, 4))
        
        self.params.set_value(item='weight', value=round(weight, 2))
        self.params.set_value(item='power_degree', value=int(power_degree))
        
        self.params.set_value(
            item='sigmoid_exponent', value=round(sigmoid_exponent, 2))

    def reset(self):
        self.params.set_value(item='max_depth', value=0)
        self.params.set_value(item='min_leaf_size', value=0)
        self.params.set_value(item='min_info_gain', value=0)
        self.params.set_value(item='weight', value=0)
        self.params.set_value(item='power_degree', value=0)
        self.params.set_value(item='sigmoid_exponent', value=0)
        
        self.progress.set_value(item='best', value=0)
        self.progress.set_value(item='fold', value=0)
        self.progress.set_value(item='iteration', value=0)

    def finalise(self):
        self.progress.set_bar_color(color='#12b980')
        self.params.set_bar_color(
            items=[
                'max_depth',
                'min_leaf_size',
                'min_info_gain',
                'weight',
                'power_degree', 
                'sigmoid_exponent'
                ], color='#0080ea')
        self.progress.collapse_items(items=['fold'])

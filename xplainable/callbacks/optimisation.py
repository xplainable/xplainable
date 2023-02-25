
import ray

class OptCallback():

    def __init__(self, progress, params):
        self.progress = progress
        self.params = params

    def fold(self, fold):
        self.params.set_value(item='fold', value=fold)

    def iteration(self, i):
        self.progress.set_value(item='iteration', value=i)

    def metric(self, v):
        self.params.set_value(item='best', value=v)

    def update_params(self, max_depth, min_leaf_size, min_info_gain, alpha=None):
        self.params.set_value(item='max_depth', value=max_depth)
        self.params.set_value(item='min_leaf_size', value=min_leaf_size)
        self.params.set_value(item='min_info_gain', value=min_info_gain)

    def finalise(self):
        self.progress.set_bar_color(color='#12b980')
        self.params.set_bar_color(
            items=['max_depth', 'min_leaf_size', 'min_info_gain'], color='#0080ea')
        self.params.collapse_items(items=['fold'])


@ray.remote
class ValueActor:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, i):
        self._value = i

    def to_value(self):
        return self._value


class OptCallbackRay():

    def __init__(self, fold=0, iteration=0, metric=0, max_depth=0, min_leaf_size=0, min_info_gain=0, alpha=None):
        self._fold = ValueActor.remote(fold)
        self._iteration = ValueActor.remote(iteration)
        self._metric = ValueActor.remote(metric)
        self._max_depth = ValueActor.remote(max_depth)
        self._min_leaf_size = ValueActor.remote(min_leaf_size)
        self._min_info_gain = ValueActor.remote(min_info_gain)

    def fold(self, fold):
        self._fold.set.remote(fold)

    def iteration(self, i):
        self._iteration.set.remote(i)

    def metric(self, v):
        self._metric.set.remote(v)

    def update_params(self, max_depth, min_leaf_size, min_info_gain, alpha=None):
        self._max_depth.set.remote(max_depth)
        self._min_leaf_size.set.remote(min_leaf_size)
        self._min_info_gain.set.remote(min_info_gain)

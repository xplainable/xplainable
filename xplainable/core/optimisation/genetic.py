import numpy as np
import pandas as pd

class XEvolutionaryNetwork:

    def __init__(self, model, metric=None):
        
        self.model = model
        self.metric = metric
        self.root_chromosome = np.array([])

        self.x = None
        self.y = None
        self._mask = None

        self.leaves = []
        self.future_layers = []
        self.completed_layers = []
        self.layer_id = 1

    def add_layer(self, layer):
        self.future_layers.append(layer)

    def drop_layer(self, idx):
        self.future_layers.pop(idx)

    def clear_layers(self):
        self.future_layers = []

    def fit(self, x, y):
        """ Transforms input variables into an array of leaf nodes for opt.

        Args:
            X (pandas.DataFrame): A dataframe of x values.

        Returns:
            numpy.array: An array of optimisable leaf nodes.
        """

        x = x.copy()

        # infer metric from type
        if self.metric == None:
            if type(self.model).__name__ == 'XRegressor':
                self.metric = 'mae'
            elif type(self.model).__name__ == 'XClassifier':
                self.metric = 'f1'

        # Remove id columns if they exist
        if len(self.model.id_columns) > 0:
            x = x.drop(columns=self.model.id_columns)

        mask_columns = [str(i) for i in range(len(x.columns))]

        # create leaf-value mask
        _mask = pd.get_dummies(
            pd.DataFrame(self.model._get_leaf_ids(x).astype(int), columns=mask_columns),
            columns=mask_columns,
            prefix_sep="_") != 0

        self._mask = _mask.values

         # Copy mask for output
        _df = _mask.copy()

         # Get list of leaf nodes
        self.leaves = np.array(_df.columns)

         # create values column for each leaf
        for i in _df.columns:
            f, _id = i.split("_")
            score = self.model._profile[int(f)][int(_id)][2]

            _df[i] = _df[i].map({True: score})
            self.root_chromosome = np.append(self.root_chromosome, score)

        self.x = _df.values
        self.y = y.values

        return self.model

    def optimise(self, callback=None):

        if len(self.future_layers) == 0:
            raise ValueError('Must include at least one layer')

        for layer in list(self.future_layers):
            self.layer_name = type(layer).__name__
            
            self.x, self.root_chromosome = layer.transform(self, self.x, self.y, callback)

            self.completed_layers.append(layer)
            self.future_layers.remove(layer)
            self.layer_id += 1

        return self
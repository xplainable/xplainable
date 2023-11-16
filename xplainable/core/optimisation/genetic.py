""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import pandas as pd
from typing import Union
from ..models import XRegressor

class XEvolutionaryNetwork:
    """ A layer-based optimisation framework for XRegressor models.

    XEvolutionaryNetwork is a novel optimisation framework for XRegressor
    models that allows for flexibility and depth. It is inspired by deep
    learning frameworks, but is applied over additive models for weight
    optimisation.

    It works by taking a pre-trained XRegressor model and fitting it,
    along with the training data, to an evolutionary network. The
    evolutionary network consists of a series of layers, each of which is
    responsible for optimising the model weights given a set of constraints.

    What are layers?:
        There are currently two types of layers: Tighten() and Evolve().
        
        More information on each layer can be found in their respective
        documentation.

    There is no limit to the number of layers that can be added to the
    network, and each layer can be customised for specific objectives.
    Like other machine learning methods, the network can be prone to
    over-fitting, so it is recommended to use a validation set to monitor
    performance.

    An XEvolutionaryNetwork can be stopped mid-training and resumed at any
    time. This is useful for long-running optimisations and iterative work.
    You can track the remaining and completed layers using the
    `future_layers` and `completed_layers` attributes.

    Args:
        model (XRegressor): The model to optimise.
        apply_range (bool): Whether to apply the model's prediction range to the output.
    """

    def __init__(self, model: 'XRegressor', apply_range: bool = False):
        
        self.model = model
        self.apply_range = apply_range

        self.root_chromosome = np.array([])
        self.x = None
        self.y = None
        self._mask = None
        self.static_scores = None

        self.leaves = []
        self.future_layers = []
        self.completed_layers = []
        self.layer_id = 0
        self.checkpoint_score = None

    def add_layer(self, layer, idx:int = None):
        """ Adds a layer to the network.

        Args:
            layer (Tighten | Evolve): The layer to add.
            idx (int, optional): The index to add the layer at.
        """
        idx = len(self.future_layers) if idx is None else idx
        self.future_layers.insert(idx, layer)

    def drop_layer(self, idx: int):
        """ Removes a layer from the network.

        Args:
            idx (int): The index of the layer to remove.
        """
        self.future_layers.pop(idx)

    def clear_layers(self):
        """ Removes all layers from the network.
        """
        self.future_layers = []

    def fit(
            self, x: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], subset: list = []
        ) -> 'XEvolutionaryNetwork':
        """ Fits the model and data to the evolutionary network.

        Args:
            x (pd.DataFrame | np.ndarray): The data to fit.
            y (pd.Series | np.ndarray): The target to fit.
            subset (list, optional): A list of columns to subset for feature level optimisation.

        Returns:
            XEvolutionaryNetwork: The fitted network.
        """

        x = x.copy()

        # Remove id columns if they exist
        if len(self.model.id_columns) > 0:
            x = x.drop(columns=[i for i in self.model.id_columns if i in x.columns])

        # Handlers for subsetting
        if len(subset) > 0:
            x_trans = self.model._transform(x)
            subset_locs = x.columns.get_indexer(subset)
            self.static_scores = np.delete(
                x_trans, subset_locs, axis=1).sum(axis=1)
            subset_locs = [str(i) for i in subset_locs]

        mask_columns = [str(i) for i in range(len(x.columns))]

        # create leaf-value mask
        if len(subset) == 0:
            _mask = pd.get_dummies(
                pd.DataFrame(
                    self.model._get_leaf_ids(x).astype(int),
                    columns=mask_columns),
                columns=mask_columns,
                prefix_sep="_") != 0
        
        # Filter a subset for feature level optimisation
        else:
            id_df = pd.DataFrame(
                self.model._get_leaf_ids(x).astype(int),
                columns=mask_columns
                )
            id_df = id_df[subset_locs]
            _mask = pd.get_dummies(id_df, columns=subset_locs, prefix_sep='_') != 0

        self._mask_df = _mask
        self._mask = _mask.values

         # Copy mask for output
        _df = _mask.copy()

         # Get list of leaf nodes
        self.leaves = np.array(_df.columns)

         # create values column for each leaf
        for i in _df.columns:
            f, _id = i.split("_")
            score = self.model._profile[int(f)][int(_id)][-4]

            _df[i] = _df[i].map({True: score})
            self.root_chromosome = np.append(self.root_chromosome, score)

        self.x = _df.values
        self.y = y.values

        return self

    def optimise(self, callback=None) -> 'XEvolutionaryNetwork':
        """ Sequentially runs the layers in the network.

        Args:
            callback (any, optional): Callback for progress tracking.

        Returns:
            XEvolutionaryNetwork: The evolutionary network.
        """

        if len(self.future_layers) == 0:
            raise ValueError('Must include at least one optimisation layer')

        for i, layer in enumerate(list(self.future_layers)):
            self.layer_name = type(layer).__name__

            self.x, self.root_chromosome = layer.transform(self, self.x, self.y, callback)

            self.completed_layers.append(layer)
            self.future_layers.remove(layer)
            self.layer_id += 1

        # Set optimised flag
        self.model.metadata["optimised"] = True

        return self
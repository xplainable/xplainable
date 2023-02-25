import numpy as np
import pandas as pd

class BaseModel:

    def __init__(self, max_depth=8, min_leaf_size=0.02, min_info_gain=0.02, alpha=0.01):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        self.alpha = alpha

        self.columns = []
        self.id_columns = []
        self._profile = {}
        self.base_value = None
        self.target_map = {}
        self.target_map_inv = {}
        self.feature_map = {}
        self.feature_map_inv = {}

    def _encode_feature(self, x, y):
        """ Encodes features in order of their relationship with the target.

        Args:
            x (pandas.Series): The feature to encode.
            y (pandas.Series): The target feature.

        Returns:
            pd.Series: The encoded Series.
        """

        name = x.name
        x = x.copy()
        y = y.copy()

        if len(self.target_map) > 0:
            y = y.map(self.target_map)

        # Order categories by their relationship with the target variable
        ordered_values = pd.DataFrame(
            {'x': x, 'y': y}).groupby('x').agg({'y': 'mean'}).sort_values(
            'y', ascending=True).reset_index()

        # Encode feature
        feature_map = {val: i for i, val in enumerate(ordered_values['x'])}

        # Store map for transformation
        self.feature_map[name] = feature_map

        # Store inverse map for reversal
        self.feature_map_inv[name] = {v: i for (i, v) in feature_map.items()}

        return

    def _encode_target(self, y):
        
        y = y.copy()

        # Cast as category
        target_ = y.astype('category')

        # Get the inverse label map
        self.target_map_inv = dict(enumerate(target_.cat.categories))

        # Get the label map
        self.target_map = {
            value: key for key, value in self.target_map_inv.items()}

        # Encode the labels
        return

    def _fetch_meta(self, x, y):
        # Assign target variable
        self.target = y.name

        # Store numeric column names
        self.numeric_columns = list(x.select_dtypes('number'))

        # Store categorical column names
        self.categorical_columns = list(x.select_dtypes('object'))

        self.columns=x.columns

    def _preprocess(self, x, y=None):
        
        x = x[self.columns]

        x = x.astype('float64')
        if y is not None:
            y = y.astype('float64')
            return x, y
        
        return x

    def _learn_encodings(self, x, y):
        
        if y.dtype == 'object': self._encode_target(y)

        for f in self.categorical_columns:
            self._encode_feature(x[f], y)

        return

    def _encode(self, x, y=None):
        # Apply encoding
        for f, m in self.feature_map.items():
            x.loc[:, f] = x.loc[:, f].map(m)

        if y is not None:
            if len(self.target_map) > 0:
                y = y.map(self.target_map)
                
            y = y.astype(float)
            return x, y
        
        return x

    def _transform(self, x):

        x = x.copy()
        
        x = self._encode(x)
        x = self._preprocess(x).values

        for i in range(x.shape[1]):
            nodes = np.array(self._profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:,i])

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes)) # flag unknown categories
            
            x[unknown, i] = 0 # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], 2]
        
        return x

    def _buid_leaf_id_map(self):
        id_map = []

        lid = 0
        for idx in range(len(self._profile)):
            fmap = [i for i in range(len(self._profile[idx]))]
            # for i in range(len(self._profile[idx])):
            #     fmap.append(lid)
            #     lid += 1
            id_map.append(fmap)
        
        return id_map

    def _get_leaf_ids(self, x):

        x = x.copy()
        
        x = self._encode(x)
        x = self._preprocess(x).values

        id_map = self._buid_leaf_id_map()

        for i in range(x.shape[1]):
            nodes = np.array(self._profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:,i])
            x[:,i] = np.vectorize(lambda x: id_map[i][x])(idx.astype(int))
        
        return x.astype(int)

    def get_profile(self):
        # instantiate Profile
        profile = {
            'base_value': self.base_value,
            'numeric': {c: [] for c in self.numeric_columns},
            'categorical': {c: [] for c in self.categorical_columns}
        }
        for (c, p) in zip(self.columns, self._profile):
            _key = "numeric" if c in self.numeric_columns else "categorical"
            
            if len(p) < 2:
                profile[_key][c] = []
                continue

            leaf_nodes = []
            for v in p:
                _prof = {
                    'lower': v[0],
                    'upper': v[1],
                    'score': v[2],
                    'mean': v[3],
                    'freq': v[4]
                    }
                leaf_nodes.append(_prof)
                
            profile[_key][c] = leaf_nodes

        return profile
    
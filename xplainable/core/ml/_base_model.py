""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from typing import Union


class BaseModel:

    def __init__(
            self, max_depth: int = 8, min_leaf_size: float = 0.02,
            min_info_gain: float = 0.02, alpha: float = 0.01):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        self.alpha = alpha

        self.columns = []
        self.id_columns = []
        self._profile = []
        self.base_value = None
        self.base_value = None
        self.target_map = {}
        self.target_map_inv = {}
        self.feature_map = {}
        self.feature_map_inv = {}
        self.category_meta = {}

        self.metadata = {"optimised": False}

    @property
    def feature_importances(self) -> dict:
        """ Calculates the feature importances for the model decision process.

        Returns:
            dict: The feature importances.
        """
        return self._get_feature_importances()
    
    @property
    def profile(self) -> dict:
        """ Returns the model profile.

        The model profile contains more granular information about the model and
        how it makes decisions. It is the primary property for interpreting a
        model and is used by the xplainable client to render the model.

        Returns:
            dict: The model profile.
        """
        return self._get_profile()

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
        self.feature_map_inv[name] = {v: i for (i, v) in feature_map.items()}  # TODO maybe use DualDict

        return

    def _cast_to_pandas(self, x, y=None, target_name='target', column_names=None):
        
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
            if column_names is not None:
                assert x.shape[1] == len(column_names), \
                    "column names must match array shape"
                x.columns = column_names
            else:
                x.columns = [f"feature_{i}" for i in range(x.shape[1])]
            x = x.apply(pd.to_numeric, errors='ignore')

        if y is not None: 
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
                assert isinstance(target_name, str), \
                    "target name must be str type"
                y.name = target_name

            return x, y

        return x

    def _encode_target(self, y):
        
        y = y.copy()

        # Cast as category
        target_ = y.astype('category')

        # Get the inverse label map
        self.target_map_inv = dict(enumerate(target_.cat.categories))

        # Get the label map
        self.target_map = {
            value: key for key, value in self.target_map_inv.items()}

        return

    def _fetch_meta(self, x, y):
        # Assign target variable
        self.target = y.name

        # Store numeric column names
        self.numeric_columns = list(x.select_dtypes('number'))

        # Store categorical column names
        self.categorical_columns = list(x.select_dtypes('object'))  # TODO make categorical

        self.columns = list(x.columns)

    def _calculate_category_meta(self, x, y):
        # calculate mean and frequency information for categories
        self.category_meta = {c: {} for c in self.categorical_columns}
        for col in self.categorical_columns:
            group = pd.DataFrame({
                col: x[col],
                'target': y
            })
            
            self.category_meta[col]["means"] = dict(
                group.groupby(col)['target'].mean())

            self.category_meta[col]["freqs"] = dict(
                x[col].value_counts() / len(y))  # TODO potential dov by 0?

    def _preprocess(self, x, y=None):
        
        x = x[[c for c in self.columns if c in x.columns]]

        x = x.astype('float64')
        if y is not None:
            y = y.astype('float64')
            return x, y
        
        return x

    def _coerce_dtypes(self, x, y=None):

        for col in x.columns:
            if is_bool_dtype(x[col]):
                x[col] = x[col].astype(str)

        if y is not None:
            if is_bool_dtype(y):
                y = y.astype(int)
            
            return x, y
        
        return x

    def _learn_encodings(self, x, y):
        
        if y.dtype == 'object':
            self._encode_target(y)

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
        
        x = self._cast_to_pandas(x, column_names=self.columns)
        x = self._coerce_dtypes(x)
        x = self._encode(x)
        x = self._preprocess(x).values

        for i in range(x.shape[1]):
            nodes = np.array(self._profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:, i])  # TODO get the bin ids?

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes))  # flag unknown categories
            
            x[unknown, i] = 0  # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], 2]  # TODO is this score?
        
        return x
    
    def predict_explain(self, x):
        """ Predictions with explanations.
        
        Args:
            x (array-like): data to predict

        Returns:
            pd.DataFrame: prediction and explanation
        """
        
        t = pd.DataFrame(self._transform(x), columns=self.columns)
        t['base_value'] = self.base_value
        t['score'] = t.sum(axis=1)
        t['proba'] = (t['score'] * 100).astype(int).map(self._calibration_map)
        t['multiplier'] = t['proba'] / t['base_value']
        t['support'] = (t['score'] * 100).astype(int).map(self._support_map)

        return t


    def _build_leaf_id_map(self):
        id_map = []

        for idx in range(len(self._profile)):
            fmap = [i for i in range(len(self._profile[idx]))]
            id_map.append(fmap)
        
        return id_map

    def convert_to_model_profile_categories(self, x):
        return self._get_leaf_ids(x)

    def _get_leaf_ids(self, x):

        x = x.copy()
        
        x = self._encode(x)
        x = self._preprocess(x).values

        id_map = self._build_leaf_id_map()

        for i in range(x.shape[1]):

            nodes = np.array(self._profile[i])
            if len(nodes) > 1:
                idx = np.searchsorted(nodes[:, 1], x[:, i])
                x[:,i] = np.vectorize(lambda x: id_map[i][x])(idx.astype(int))
            else:
                x[:,i] = 0

        return x.astype(int)

    def _get_profile(self):
        # instantiate Profile
        profile = {
            'base_value': self.base_value,
            'numeric': {c: [] for c in self.numeric_columns},
            'categorical': {c: [] for c in self.categorical_columns}
        }
        for i, (c, p) in enumerate(zip(self.columns, self._profile)):
            p = np.array(p)
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
                if _key == "categorical":
                    _prof.update({
                        'categories': [],
                        'means': [],
                        'frequencies': []
                    })

                leaf_nodes.append(_prof)

            if _key == 'categorical':
                mapp_inv = self.feature_map_inv[c]
                for k in np.array(list(mapp_inv.keys())):
                    idx = np.where((p[:, 0] < k) & (k < p[:, 1]))
                    leaf_nodes[idx[0][0]]['categories'].append(mapp_inv[k])
                    
                    leaf_nodes[idx[0][0]]['means'].append(
                        self.category_meta[c]["means"][k])

                    leaf_nodes[idx[0][0]]['frequencies'].append(
                        self.category_meta[c]["freqs"][k])

            profile[_key][c] = leaf_nodes

        return profile
    
    def _get_feature_importances(self):
        """ Calculates the feature importances for the model decision process.

        Returns:
            dict: The feature importances.
        """

        importances = {}
        total_importance = 0
        profile = self.profile
        for i in ["numeric", "categorical"]:
            for feature, leaves in profile[i].items():        
                importance = 0
                for leaf in leaves:
                    importance += abs(leaf['score']) * np.log2(leaf['freq']*100)
                
                importances[feature] = importance
                total_importance += importance

        return {k: v/total_importance for k, v in sorted(
            importances.items(), key=lambda item: item[1])}
    
    def explain(self, label_rounding=5):
        try:
            from ...visualisation.explain import _plot_explainer
        except Exception as e:
            raise ImportError(e)

        return _plot_explainer(self, label_rounding)
    
    def local_explainer(self, x, subsample):
        try:
            from ...visualisation.explain import _plot_local_explainer
        except Exception as e:
            raise ImportError(e)
        
        t = pd.DataFrame(self._transform(x), columns=self.columns)
        t['base_value'] = self.base_value

        return _plot_local_explainer(self, t, subsample)

    def _check_param_bounds(self):
        assert self.max_depth >= 0, \
            'max_depth must be greater than or equal to 0'

        assert -1 <= self.min_leaf_size < 1, \
            'min_leaf_size must be between -1 and 1'

        assert -1 <= self.min_info_gain < 1, \
            'min_info_gain must be between -1 and 1'

        assert 0 <= self.alpha <= 1, 'alpha must be between 0 and 1'

    def _fit_check(  # TODO better function name
        self, x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], id_columns: list = [],
        column_names: list = None, target_name: str = 'target', map_calibration=False
    ):  # TODO clarify return type: ndarray, ndarray, df, df
        # Ensure parameters are valid
        self._check_param_bounds()

        x = x.copy()
        y = y.copy()

        # casts ndarray to pandas
        x, y = self._cast_to_pandas(x, y, target_name, column_names)

        if map_calibration:
            x_cal = x.copy()
            y_cal = y.copy()

        # Store meta data
        self.id_columns = id_columns
        x = x.drop(columns=id_columns)

        # Preprocess data
        x, y = self._coerce_dtypes(x, y)
        self._fetch_meta(x, y)
        self._learn_encodings(x, y)
        x, y = self._encode(x, y)
        self._calculate_category_meta(x, y)
        x, y = self._preprocess(x, y)

        x = x.values
        y = y.values
        self.base_value = np.mean(y)

        if map_calibration:
            return x, y, x_cal, y_cal
        else:
            return x, y, None, None


class BasePartition:

    def __init__(self):
        self.partition_on = None
        self.partitions = {}

    def __verify_mappings(self, model):
        assert model.target_map == self.partitions['__dataset__'].target_map, \
            "Target mappings are mismatched"

    def add_partition(self, model , partition: str):
        """ Adds a partition to the model.
        
        All partitions must be of the same type.

        Args:
            model (XClassifier | XRegressor): The model to add.
            partition (str): The name of the partition to add.
        """
        partition = str(partition)
        self.partitions[partition] = model
        if hasattr(model, 'target_map'):
            self.__verify_mappings(model)
            
    def drop_partition(self, partition: str):
        """ Removes a partition from the model.

        Args:
            partition (str): The name of the partition to drop.
        """
        self.partitions.pop(partition)

    def _encode(self, x, y=None, partition='__dataset__'):

        x = x.copy()
        partition = str(partition)

        # Apply encoding
        for f, m in self.partitions[partition].feature_map.items():
            x.loc[:, f] = x.loc[:, f].map(m)

        if y is not None:
            if len(self.partitions[partition].target_map) > 0:
                y = y.map(self.partitions[partition].target_map)
                
            y = y.astype(float)
            return x, y
        
        return x

    def _preprocess(self, x, y=None):
        
        x = x[self.partitions['__dataset__'].columns]

        x = x.astype('float64')
        if y is not None:
            y = y.astype('float64')
            return x, y
        
        return x

    def _transform(self, x, partition):
        """ Transforms a dataset into the model weights.
        
        Args:
            x (pandas.DataFrame): The dataframe to be transformed.
            
        Returns:
            pandas.DataFrame: The transformed dataset.
        """

        assert str(partition) in self.partitions.keys(), \
            f'Partition {partition} does not exist'

        x = x.copy()
        partition = str(partition)

        x = self._encode(x, None, partition)
        x = self._preprocess(x).values

        profile = self.partitions[partition]._profile

        for i in range(x.shape[1]):
            nodes = np.array(profile[i])
            idx = np.searchsorted(nodes[:, 1], x[:,i])

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes)) # flag unknown categories
            
            x[unknown, i] = 0 # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], 2]

        return x
    
    def explain(self, partition: str = '__dataset__'):
        """ Generates a global explainer for the model.

        Args:
            partition (str): The partition to explain.

        Raises:
            ImportError: If user does not have altair installed.

        """
        try:
            from ...visualisation.explain import _plot_explainer
        except Exception as e:
            raise ImportError(e)

        return _plot_explainer(self.partitions[partition])

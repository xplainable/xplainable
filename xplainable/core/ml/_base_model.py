""" Copyright Xplainable Pty Ltd, 2023"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from typing import Union
from ...utils.dualdict import FeatureMap, TargetMap
from ._constructor_parameters import ConstructorParams
from ._constructor import XConstructor


class BaseModel:

    def __init__(
        self,
        default_parameters: ConstructorParams
    ):
        self.default_parameters = default_parameters

        self.columns = []
        self.id_columns = []

        self._constructs = []
        self._profile = []

        self.base_value = None
        self.base_value = None
        self.target_map = TargetMap()
        self.feature_map = dict()
        self.category_meta = {}

        self.min_seen = 0
        self.max_seen = 1

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

    def _get_profile(self):
        # instantiate Profile
        profile = {
            'base_value': self.base_value,
            'numeric': {c: [] for c in self.numeric_columns},
            'categorical': {c: [] for c in self.categorical_columns}
        }
        for c, p in zip(self.columns, self._profile):
            p = np.array(p)
            _key = "numeric" if c in self.numeric_columns else "categorical"

            if len(p) < 2:
                profile[_key][c] = []
                continue

            leaf_nodes = []
            for v in p:
                if _key == "categorical":
                    _prof = {
                        'category': self.feature_map[c].get_item_directional(v[0], reverse=True),
                        'score': v[1],
                        'mean': v[2],
                        'freq': v[3],
                    }
                else:
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

    @property
    def params(self) -> ConstructorParams:
        """ Returns the parameters of the model.

        Returns:
            ConstructorParams: The default model parameters.
        """

        return self.default_parameters

    def set_params(
            self,
            default_parameters: ConstructorParams
    ) -> None:
        """ Sets the parameters of the model. Generally used for model tuning.

        Args:
            default_parameters (ConstructorParams): default constructor parameters

        Returns:
            None
        """
        self.default_parameters = default_parameters

    def update_feature_params(
        self,
        features: list,
        max_depth=None,
        min_info_gain=None,
        min_leaf_size=None,
        ignore_nan=None,
        weight=None,
        power_degree=None,
        sigmoid_exponent=None,
        tail_sensitivity=None,
        *args, **kwargs
    ) -> 'XClassifier':
        """ Updates the parameters for a subset of features.

        XClassifier allows you to update the parameters for a subset of features
        for a more granular approach to model tuning. This is useful when you
        identify under or overfitting on some features, but not all.

        This also refered to as 'refitting' the model to a new set of params.
        Refitting parameters to an xplainable model is extremely fast as it has
        already pre-computed the complex metadata required for training.
        This can yeild huge performance gains compared to refitting
        traditional models, and is particularly powerful when parameter tuning.
        The desired result is to have a model that is well calibrated across all
        features without spending considerable time on parameter tuning.

        Args:
            features (list): The features to update.
            max_depth (int): The maximum depth of each decision tree in the subset.
            min_info_gain (float): The minimum information gain required to make a split in the subset.
            min_leaf_size (float): The minimum number of samples required to make a split in the subset.
            ignore_nan (bool): Whether to ignore nan/missing values for training
            weight (float): Activation function weight.
            power_degree (float): Activation function power degree.
            sigmoid_exponent (float): Activation function sigmoid exponent.
            tail_sensitivity (float): Adds weight to divisive leaf nodes in the subset.
            x (pd.DataFrame | np.ndarray, optional): The x variables used for training. Use if map_calibration is True.
            y (pd.Series | np.array, optional): The target values. Use if map_calibration is True.

        Returns:
            XClassifier: The refitted model.
        """

        for feature in features:
            idx = self.columns.index(feature)
            self._constructs[idx].params.update_parameters(
                max_depth,
                min_info_gain,
                min_leaf_size,
                ignore_nan,
                weight,
                power_degree,
                sigmoid_exponent,
                tail_sensitivity
            )
            self._constructs[idx].construct()

        self._build_profile()

        return self

    def _build_profile(self):
        """ Builds the profile from each feature construct."""
        self._profile = []
        _sum_min, _sum_max = [
            sum(m) for m in zip(
                *[
                    (const.min_raw_score, const.max_raw_score) for const in self._constructs
                ]
            )
        ]

        for xconst in self._constructs:
            xconst.normalise_scores(_sum_min, _sum_max, self.base_value, self.min_seen, self.max_seen)
            self._profile.append(np.array([list(x) for x in xconst._nodes]))

        return self

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
            {'x': x, 'y': y}
        ).groupby('x').agg({'y': 'mean'}).sort_values(
            'y', ascending=True
        ).reset_index()
        # Encode feature
        self.feature_map[name] = FeatureMap(
            {val: i for i, val in enumerate(ordered_values['x'])}
        )
        return

    @staticmethod
    def _cast_to_pandas(x, y=None, target_name='target', column_names=None):

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

        # Get the label map
        self.target_map = TargetMap({value: key for key, value in dict(
            enumerate(target_.cat.categories)).items()})

        return

    def _fetch_meta(self, x, y):
        # Assign target variable
        self.target = y.name
        # Store numeric column names
        self.numeric_columns = list(x.select_dtypes('number'))
        # Store categorical column names
        self.categorical_columns = list(x.select_dtypes('object'))
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
                x[col].value_counts() / len(y))

    def _preprocess(self, x, y=None):
        """Removes unknown categories and puts them into the self.columns order"""
        
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
            idx = np.searchsorted(nodes[:, -5], x[:, i])

            known = np.where(idx < len(nodes))
            unknown = np.where(idx >= len(nodes))  # flag unknown categories, the addition of nan might change this
            
            x[unknown, i] = 0  # Set new categories to 0 contribution
            x[known, i] = nodes[idx[known], -4]  # get score

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
        t['multiplier'] = t['proba'] / t['base_value']

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
                idx = np.searchsorted(nodes[:, -5], x[:, i])
                x[:,i] = np.vectorize(lambda x: id_map[i][x])(idx.astype(int))
            else:
                x[:,i] = 0

        return x.astype(int)

    def _calculate_gini_gain(self):
        gini_gains = {}

        # Function to calculate Gini Impurity
        def gini_impurity(freqs):
            return min(max(1 - sum(f ** 2 for f in freqs), 0), 1)
        
        # Calculate Gini gain for categorical features
        for feature, categories in list(self.profile["categorical"].items()) + list(self.profile["numeric"].items()):

            impurity = gini_impurity(
                [category["freq"] for category in categories]
            )

            weighted_impurity = impurity * abs(sum(
                abs(category["score"]) for category in categories
            ))
            
            gini_gains[feature] = weighted_impurity

        return gini_gains

    def _get_feature_importances(self):
        """ Calculates the feature importances for the model decision process.

        Returns:
            dict: The feature importances.
        """

        importances = self._calculate_gini_gain()
        sum_importances = sum(importances.values())
        importances ={k: v/sum_importances for k, v in importances.items()}
        # order by importance
        importances = dict(sorted(
            importances.items(), key=lambda item: item[1], reverse=False
        ))

        return importances

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

    def _fit_check(
        self, x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], id_columns: list = [],
        column_names: list = None, target_name: str = 'target',
        map_calibration=False
    ):

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
        x, y = self._encode(x, y)  # turns categories into into bin indices
        self._calculate_category_meta(x, y)
        x, y = self._preprocess(x, y)

        x = x.values
        y = y.values
        self.base_value = np.mean(y)

        # Dynamic min_leaf_size
        if self.params.min_leaf_size == -1:
            self.min_leaf_size = self.base_value / 10

        # Dynamic min_info_gain
        if self.params.min_info_gain == -1:
            self.min_info_gain = self.base_value / 10

        if map_calibration:
            return x, y, x_cal, y_cal
        else:
            return x, y, None, None

    def constructs_to_json(self):
        constructs = []
        for c in self._constructs:
            constructs.append(c.to_json(self.params))
            # XConstructor.from_json(c.to_json(model.params), default_parameter_set)
        return constructs

    def constructs_from_json(self, data):
        self._constructs = []
        for c in data:
            self._constructs.append(XConstructor.from_json(c))


class BasePartition:

    def __init__(self):
        self.partition_on = None
        self.partitions = {}

    def __verify_mappings(self, model):
        if '__dataset__' not in self.partitions.keys():
            return
        assert dict(model.target_map) == dict(self.partitions['__dataset__'].target_map), \
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

        assert str(partition) in self.partitions.keys(), f'Partition {partition} does not exist'

        x = x.copy()
        partition = str(partition)

        x = self._encode(x, None, partition)
        x = self._preprocess(x).values

        profile = self.partitions[partition]._profile

        for i in range(x.shape[1]):
            nodes = np.array(profile[i])
            idx = np.searchsorted(nodes[:, -5], x[:, i])

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

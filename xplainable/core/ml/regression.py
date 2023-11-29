""" Copyright Xplainable Pty Ltd, 2023"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ._base_model import BaseModel, BasePartition
from ._constructor import XCatConstructor, XNumConstructor, ConstructorParams
from sklearn.metrics import *
from time import time
from typing import Union


class XRegressor(BaseModel):
    """ Xplainable Regression model for transparent machine learning.

    XRegressor offers powerful predictive power and complete transparency
    for regression problems on tabular data. It is designed to be used
    in place of black box models such as Random Forests and Gradient
    Boosting Machines when explainabilty is important.

    XRegressor is a feature-wise ensemble of decision trees. Each tree is
    constructed using a custom algorithm that optimises for information with
    respect to the target variable. The trees are then weighted and
    normalised against one another to produce a variable step function
    for each feature. The summation of these functions produces a score that can
    be explained in real time. The bounds of the prediction can
    be set using the prediction_range parameter.

    When the fit method is called, the specified params are set across all
    features. Following the initial fit, the update_feature_params method
    may be called on a subset of features to update the params for those
    features only. This allows for a more granular approach to model tuning.

    Important note on performance:
        XRegressor alone can be a weak predictor. There are a number of ways
        to get the most out of the model in terms of predictive power:
        - use the optimise_tail_sensitivity method
        - fit an XEvolutionaryNetwork to the model. This will
        iteratively optimise the weights of the model to produce a much
        more accurate predictor. You can find more information on this
        in the XEvolutionaryNetwork documentation at
        xplainable/core/optimisation/genetic.py.

        
    Example:
        >>> from xplainable.core.models import XRegressor
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split

        >>> data = pd.read_csv('data.csv')
        >>> x = data.drop(columns=['target'])
        >>> y = data['target']
        >>> x_train, x_test, y_train, y_test = train_test_split(
        >>>     x, y, test_size=0.2, random_state=42)

        >>> model = XRegressor()
        >>> model.fit(x_train, y_train)

        >>> # This will be a weak predictor
        >>> model.predict(x_test)
        
        >>> # For a strong predictor, apply optimisations
        >>> model.optimise_tail_sensitivity(x_train, y_train)
        >>> # Add evolutionary network here
        >>> ...

    Args:
        max_depth (int): The maximum depth of each decision tree.
        min_leaf_size (float): The minimum number of samples required to make a split.
        min_info_gain (float): The minimum information gain required to make a split.
        tail_sensitivity (float): Adds weight to divisive leaf nodes.
        prediction_range (tuple): The lower and upper limits for predictions.
    
    """

    def __init__(
        self,
        max_depth=8,
        min_info_gain=0.0001,
        min_leaf_size=0.0001,
        ignore_nan=False,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        tail_sensitivity: float = 1.0,
        prediction_range: tuple = (-np.inf, np.inf)
    ):
        super().__init__(
            ConstructorParams(
                max_depth,
                min_info_gain,
                min_leaf_size,
                ignore_nan,
                weight,
                power_degree,
                sigmoid_exponent,
                tail_sensitivity
            )
        )

        self.prediction_range = prediction_range

        self.feature_params = {}
        self.samples = None

        self.min_seen = 0
        self.max_seen = 0

    def fit(
        self, x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray], id_columns: list = [],
        column_names: list = None, target_name: str = 'target', alpha=0.1
    ) -> 'XRegressor':
        """ Fits the model to the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables used for training.
            y (pd.Series | np.array): The target values.
            id_columns (list, optional): id_columns to ignore from training.
            column_names (list, optional): column_names to use for training if using a np.ndarray
            target_name (str, optional): The name of the target column if using a np.ndarray
            alpha (float, optional): Sets the number of possible splits with respect to unique values.

        Returns:
            XRegressor: The fitted model.
        """

        self.min_seen = y.min()
        self.max_seen = y.max()

        start = time()

        x, y, _, _ = super()._fit_check(
            x,
            y,
            id_columns,
            column_names,
            target_name
        )

        self.samples = y.size
        for i in range(x.shape[1]):
            f = x[:, i]

            # chooses constructor type based on type of input feature
            constructor = XCatConstructor if self.columns[i] in self.categorical_columns else XNumConstructor

            xconst = constructor(True, self.default_parameters.__copy__())

            xconst.fit(f, y, alpha)
            xconst.construct()
            self._constructs.append(xconst)

        self._build_profile()

        # record metadata
        self.metadata['fit_time'] = time() - start
        self.metadata['observations'] = x.shape[0]
        self.metadata['features'] = x.shape[1]

        return self
    
    def optimise_tail_sensitivity(
            self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'XRegressor':
        """ Optimises the tail_sensitivity parameter at a global level.

        Args:
            X (pd.DataFrame | np.ndarray): The x variables to fit.
            y (pd.Series | np.ndarray): The target values.

        Returns:
            XRegressor: The optimised model.
        """
        
        centre = 0.1
        window = 0.1
        
        def trio_trial(a, b, c):
            _best_ts = None
            _best_metric = np.inf
            _best_i = None
        
            for i, ts in enumerate([a, b, c]):
                self.update_feature_params(self.columns, tail_sensitivity = 1+ts)
                
                _metric = self.evaluate(X, y)['MAE']
                if _metric < _best_metric:
                    _best_metric = _metric
                    _best_ts = ts
                    _best_i = i
                    
            return _best_i, round(_best_metric,2), round(_best_ts-1,2)
        
        while True:
            i, metric, ts = trio_trial(
                round(1+centre-window,2),
                round(1+centre,2),
                round(1+centre+window,2)
                )
            
            if i == 1:
                new_window = int(window*100/2)/100
                if new_window == window or new_window == 0:
                    break
                else:
                    window = new_window
            
            elif i == 2:
                centre += window
            
            elif i == 0:
                new_window = int(window*100/2)/100
                if new_window == window or new_window == 0:
                    break
                centre = int(centre/2)
        
        self.update_feature_params(self.columns, tail_sensitivity = 1+ts)
        
        return self

    def update_feature_params(
            self, features: list,
            max_depth=None,
            min_info_gain=None,
            min_leaf_size=None,
            ignore_nan=None,
            weight=None,
            power_degree=None,
            sigmoid_exponent=None,
            tail_sensitivity=None,
            *args, **kwargs) -> 'XRegressor':
        """ Updates the parameters for a subset of features.

        XRegressor allows you to update the parameters for a subset of features
        for a more granular approach to model tuning. This is useful when you
        identify under or overfitting on some features, but not all.

        This also refered to as 'refitting' the model to a new set of params.
        Refitting parameters to an xplainable model is extremely fast as it has
        already pre-computed the complex metadata required for training.
        This can yeild huge performance gains compared to refitting
        traditional models, and is particularly powerful when parameter tuning.
        The desired result is to have a model that is well calibrated across all
        features without spending considerable time on parameter tuning.

        It's important to note that if a model has been further optimised using
        an XEvolutionaryNetwork, the optimised feature_params will be
        overwritten by this method and will need to be re-optimised.

        Args:
            features (list): The features to update.
            max_depth (int): The maximum depth of each decision tree in the subset.
            min_info_gain (float): The minimum information gain required to make a split in the subset.
            min_leaf_size (float): The minimum number of samples required to make a split in the subset.
            ignore_nan (bool): Whether to ignore nan/null/empty values
            weight (float): Activation function weight.
            power_degree (float): Activation function power degree.
            sigmoid_exponent (float): Activation function sigmoid exponent.
            tail_sensitivity (float): Adds weight to divisive leaf nodes in the subset.

        Returns:
            XRegressor: The refitted model.
        """

        super().update_feature_params(
            features,
            max_depth=max_depth,
            min_info_gain=min_info_gain,
            min_leaf_size=min_leaf_size,
            ignore_nan=ignore_nan,
            weight=weight,
            power_degree=power_degree,
            sigmoid_exponent=sigmoid_exponent,
            tail_sensitivity=tail_sensitivity,
            *args, **kwargs
        )
        
        return self

    def predict(self, x: Union[pd.DataFrame, np.ndarray]) -> np.array:
        """ Predicts the target value for each row in the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted target values.
        """
        trans = self._transform(x)
        pred = np.sum(trans, axis=1) + self.base_value

        return pred.clip(*self.prediction_range)

    def evaluate(
            self, x: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> dict:
        """ Evaluates the model performance.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.
            y (pd.Series | np.array): The target values.

        Returns:
            dict: The model performance metrics.
        """

        y_pred = self.predict(x)

        mae = round(mean_absolute_error(y, y_pred), 4)
        mape = round(mean_absolute_percentage_error(y, y_pred), 4)
        r2 = round(r2_score(y, y_pred), 4)
        mse = round(mean_squared_error(y, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        evs = round(explained_variance_score(y, y_pred), 4)
        
        try:
            msle = round(mean_squared_log_error(y, y_pred), 4)
        except Exception:
            msle = np.nan

        metrics = {
            "Explained Variance": evs,
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "RMSE": rmse,
            "RMSLE": msle,
            "R2 Score": r2
        }

        return metrics


class PartitionedRegressor(BasePartition):
    """ Partitioned XRegressor model.

    This class is a wrapper for the XRegressor model that allows for
    individual models to be trained on subsets of the data. Each model
    can be used in isolation or in combination with the other models.

    Individual models can be accessed using the partitions attribute.

    Example:
        >>> from xplainable.core.models import PartitionedRegressor
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split

        >>> data = pd.read_csv('data.csv')
        >>> train, test = train_test_split(data, test_size=0.2)

        >>> # Train your model (this will open an embedded gui)
        >>> partitioned_model = PartitionedClassifier(partition_on='partition_column')

        >>> # Iterate over the unique values in the partition column
        >>> for partition in train['partition_column'].unique():
        >>>         # Get the data for the partition
        >>>         part = train[train['partition_column'] == partition]
        >>>         x_train, y_train = part.drop('target', axis=1), part['target']
        >>>         # Fit the embedded model
        >>>         model = XRegressor()
        >>>         model.fit(x_train, y_train)
        >>>         model.optimise_tail_sensitivity(x_train, y_train)
        >>>         # <-- Add XEvolutionaryNetwork here -->
        >>>         # Add the model to the partitioned model
        >>>         partitioned_model.add_partition(model, partition)
        
        >>> # Prepare the test data
        >>> x_test, y_test = test.drop('target', axis=1), test['target']

        >>> # Predict on the partitioned model
        >>> y_pred = partitioned_model.predict(x_test)

    Args:
        partition_on (str, optional): The column to partition on.
    """

    def __init__(self, partition_on=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_on = partition_on

    def predict(self, x) -> np.array:
        """ Predicts the target value for each row in the data across all partitions.

        The partition_on columns will be used to determine which model to use
        for each observation. If the partition_on column is not present in the
        data, the '__dataset__' model will be used.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted target values
        """
        x = pd.DataFrame(x).copy().reset_index(drop=True)

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict(x)

        else:
            partitions = self.partitions.keys()
            frames = []
            unq = [str(i) for i in list(x[self.partition_on].unique())]

            # replace unknown partition values with __dataset__ for general model
            for u in unq:
                if u not in partitions:
                    x[self.partition_on] = x[self.partition_on].replace(
                        u, '__dataset__')
                        
                    unq.remove(u)
                    if "__dataset__" not in unq:
                        unq.append("__dataset__")

            for partition in unq:
                part = x[x[self.partition_on].astype(str) == partition]
                idx = part.index

                # Use partition model first
                part_trans = self.partitions[partition]._transform(part)
                _base_value = self.partitions[partition].base_value

                scores = pd.Series(part_trans.sum(axis=1) + _base_value)
                scores.index = idx
                frames.append(scores)
        
            return np.array(pd.concat(frames).sort_index())
        
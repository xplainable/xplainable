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


class XClassifier(BaseModel):
    """ Xplainable Classification model for transparent machine learning.

    XClassifier offers powerful predictive power and complete transparency
    for classification problems on tabular data. It is designed to be used
    in place of black box models such as Random Forests and Gradient
    Boosting Machines when explainabilty is important.

    XClassifier is a feature-wise ensemble of decision trees. Each tree is
    constructed using a custom algorithm that optimises for information with
    respect to the target variable. The trees are then weighted and
    normalised against one another to produce a variable step function
    for each feature. The summation of these functions produces a score that can
    be explained in real time. The score is a float value between 0 and 1
    and represents the likelihood of the positive class occuring. The score
    can also be mapped to a probability when probability is important.

    When the fit method is called, the specified params are set across all
    features. Following the initial fit, the update_feature_params method
    may be called on a subset of features to update the params for those
    features only. This allows for a more granular approach to model tuning.

    Example:
        >>> from xplainable.core.models import XClassifier
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split

        >>> data = pd.read_csv('data.csv')
        >>> x = data.drop(columns=['target'])
        >>> y = data['target']
        >>> x_train, x_test, y_train, y_test = train_test_split(
        >>>     x, y, test_size=0.2, random_state=42)

        >>> model = XClassifier()
        >>> model.fit(x_train, y_train)

        >>> model.predict(x_test)

    Args:
        max_depth (int, optional): The maximum depth of each decision tree.
        min_info_gain (float, optional): The minimum information gain required to make a split.
        min_leaf_size (float, optional): The minimum number of samples required to make a split.
        alpha (float, optional): Sets the number of possible splits with respect to unique values.
        weight (float, optional): Activation function weight.
        power_degree (float, optional): Activation function power degree.
        sigmoid_exponent (float, optional): Activation function sigmoid exponent.
        map_calibration (bool, optional): Maps the associated probability for each possible feature score.
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
        map_calibration: bool = True
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

        self._calibration_map = {}
        self._support_map = {}

        self.map_calibration = map_calibration

    def _map_calibration(self, y, y_prob, smooth=15):
        """ Maps the associated probability for each possible feature score.

        Args:
            x (pandas.DataFrame): The x variables used for training.
            y (pandas.Series): The target series.
        """

        # Make prediction and set to x
        x = pd.DataFrame(
            {
                'y_prob': y_prob,
                'target': y.copy().values
            })

        # Record prediction bins
        x['bin'] = pd.cut(x['y_prob'], [i / 100 for i in range(0, 101, 5)])

        # Get target info grouped by bins
        df = x.groupby('bin').agg({'target': ['mean', 'count']})

        # Fix column formatting
        df.columns = df.columns.map('_'.join)
        df = df.rename(columns={'target_mean': 'tm', 'target_count': 'tc'})

        # Record last and next scores for normalisation
        df['lc'] = df['tc'].shift(1)
        df['nc'] = df['tc'].shift(-1)

        df['lm'] = df['tm'].shift(1)
        df['nm'] = df['tm'].shift(-1)

        # np.nan is the same as 0
        df.fillna(0, inplace=True)

        # Record rolling total for normalisation calc
        df['rt'] = df['tc'].rolling(3, center=True, min_periods=2).sum()

        # Scale counts to rolling total
        df['tc_pct'] = df['tc'] / df['rt']
        df['lc_pct'] = df['lc'] / df['rt']
        df['nc_pct'] = df['nc'] / df['rt']

        # Calculate weighted probability
        df['wp'] = (df['lc_pct'] * df['lm']) + (df['tc_pct'] * df['tm']) + \
            (df['nc_pct'] * df['nm'])

        # Forward fill zero values
        df['wp'] = df['wp'].replace(
            to_replace=0, method='ffill')

        # Get weighted probability and arrange
        wp = df['wp']
        wp = pd.DataFrame(np.repeat(wp.values, 5, axis=0))
        wp = pd.concat([wp, wp.iloc[99]], ignore_index=True)

        # Forward fill nan values
        wp = wp.fillna(method='ffill')

        # Fill missing values that could not be
        # forward filled
        wp = wp.fillna(0)

        # Calculate support at each bin
        s = df['tc']
        s = pd.DataFrame(np.repeat(s.values, 5, axis=0))
        s = pd.concat([s, s.iloc[99]], ignore_index=True)
        s = s.fillna(method='ffill')
        s = s.fillna(0)
        self._support_map.update(dict(s[0]))

        wp = wp.rolling(smooth, center=True, min_periods=3).mean()[0]
        wp.fillna(method='ffill')
        wp.fillna(0)

        # Store results dict to class variable
        return dict(wp)

    def fit(
        self, x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.array], id_columns: list = [],
        column_names: list = None, target_name: str = 'target', alpha=0.1
    ) -> 'XClassifier':
        """ Fits the model to the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables used for training.
            y (pd.Series | np.array): The target values.
            id_columns (list, optional): id_columns to ignore from training.
            column_names (list, optional): column_names to use for training if using a np.ndarray
            target_name (str, optional): The name of the target column if using a np.array
            alpha (float): Controlls the number of possible splits with respect to unique values.

        Returns:
            XClassifier: The fitted model.
        """

        start = time()
        x, y, x_cal, y_cal = super()._fit_check(
            x,
            y,
            id_columns,
            column_names,
            target_name,
            map_calibration=self.map_calibration
        )

        for i in range(x.shape[1]):
            f = x[:, i]

            # chooses constructor type based on type of input feature
            constructor = XCatConstructor if self.columns[i] in self.categorical_columns else XNumConstructor

            xconst = constructor(False, self.default_parameters.__copy__())
            xconst.fit(f, y, alpha)
            xconst.construct()
            self._constructs.append(xconst)

        self._build_profile()

        # Calibration map
        if self.map_calibration:
            if len(self.target_map) > 0:
                y_cal = y_cal.map(self.target_map)
            y_prob = self.predict_score(x_cal)
            self._calibration_map = self._map_calibration(y_cal, y_prob, 15)
        
        # record metadata
        self.metadata['fit_time'] = time() - start
        self.metadata['observations'] = x.shape[0]
        self.metadata['features'] = x.shape[1]

        return self

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
        x: Union[pd.DataFrame, np.ndarray] = None,
        y: Union[pd.Series, np.array] = None, *args, **kwargs
    ) -> 'XClassifier':
        """ Updates the parameters for a subset of features.

        XClassifier allows you to update the parameters for a subset of features
        for a more granular approach to model tuning. This is useful when you
        identify under or overfitting on some features, but not all.

        This also referred to as 'refitting' the model to a new set of params.
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
            ignore_nan (bool): Whether to ignore nan/null/empty values
            weight (float): Activation function weight.
            power_degree (float): Activation function power degree.
            sigmoid_exponent (float): Activation function sigmoid exponent.
            tail_sensitivity (float): Adds weight to divisive leaf nodes in the subset.
            x (pd.DataFrame | np.ndarray, optional): The x variables used for training. Use if map_calibration is True.
            y (pd.Series | np.array, optional): The target values. Use if map_calibration is True.

        Returns:
            XClassifier: The refitted model.
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

        if self.map_calibration and x is not None and y is not None:
            y_prob = self.predict_score(x)
            self._calibration_map = self._map_calibration(y, y_prob, 15)

        return self

    def predict_score(self, x: Union[pd.DataFrame, np.ndarray]) -> np.array:
        """ Predicts the score for each row in the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted scores
        """
        trans = self._transform(x)
        scores = np.sum(trans, axis=1) + self.base_value

        return scores

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]) -> np.array:
        """ Predicts the probability for each row in the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted probabilities
        """
        scores = self.predict_score(x) * 100
        scores = scores.astype(int)

        scores = np.vectorize(self._calibration_map.get)(scores)

        return scores

    def predict(
            self, x: Union[pd.DataFrame, np.ndarray],
            use_prob: bool=False, threshold: float = 0.5,
            remap: bool = True
    ) -> np.array:
        """ Predicts the target for each row in the data.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.
            use_prob (bool, optional): Use probability instead of score.
            threshold (float, optional): The threshold to use for classification.
            remap (bool, optional): Remap the target values to their original values.

        Returns:
            np.array: The predicted targets
        """
        scores = self.predict_proba(x) if use_prob else self.predict_score(x)
        pred = (scores > threshold).astype(int)

        if len(self.target_map) > 0 and remap:
            pred = np.vectorize(self.target_map.reverse.get)(pred)

        return pred

    def predict_explain(self, x):
        """ Predictions with explanations.

        Args:
            x (array-like): data to predict

        Returns:
            pd.DataFrame: prediction and explanation
        """

        t = super().predict_explain(x)
        t['proba'] = (t['score'] * 100).astype(int).map(self._calibration_map)
        t['support'] = (t['score'] * 100).astype(int).map(self._support_map)

        return t

    def evaluate(
        self, x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.array], use_prob: bool = False,
        threshold: float = 0.5
    ):
        """ Evaluates the model performance.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.
            y (pd.Series | np.array): The target values.
            use_prob (bool, optional): Use probability instead of score.
            threshold (float, optional): The threshold to use for classification.

        Returns:
            dict: The model performance metrics.
        """

        # Make predictions
        y_prob = self.predict_proba(x) if use_prob else self.predict_score(x)
        y_prob = np.clip(y_prob, 0, 1)  # because of rounding errors
        y_pred = (y_prob > threshold).astype(int)

        if (len(self.target_map) > 0) and (y.dtype == 'object'):
            y = y.copy().map(self.target_map)

        # Calculate metrics
        cm = confusion_matrix(y, y_pred).tolist()
        cr = classification_report(y, y_pred, output_dict=True, zero_division=0)

        try:
            roc_auc = roc_auc_score(y, y_prob)
        except Exception:
            roc_auc = np.nan

        try:
            brier_loss = 1 - brier_score_loss(y, y_prob)
        except Exception:
            brier_loss = np.nan

        try:
            cohen_kappa = cohen_kappa_score(y, y_pred)
        except Exception:
            cohen_kappa = np.nan
        
        try:
            log_loss_score = log_loss(y, y_prob)
        except Exception:
            log_loss_score = np.nan

        # Produce output
        evaluation = {
            'confusion_matrix': cm,
            'classification_report': cr,
            'roc_auc': roc_auc,
            'neg_brier_loss': brier_loss,
            'log_loss': log_loss_score,
            'cohen_kappa': cohen_kappa

        }

        return evaluation


class PartitionedClassifier(BasePartition):
    """ Partitioned XClassifier model.

    This class is a wrapper for the XClassifier model that allows for
    individual models to be trained on subsets of the data. Each model
    can be used in isolation or in combination with the other models.

    Individual models can be accessed using the partitions attribute.

    Example:
        >>> from xplainable.core.models import PartitionedClassifier
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
        >>>         model = XClassifier()
        >>>         model.fit(x_train, y_train)
        >>>         # Add the model to the partitioned model
        >>>         partitioned_model.add_partition(model, partition)
        
        >>> # Prepare the test data
        >>> x_test, y_test = test.drop('target', axis=1), test['target']

        >>> # Predict on the partitioned model
        >>> y_pred = partitioned_model.predict(x_test)

    Args:
        partition_on (str, optional): The column to partition on.
    """

    def __init__(self, partition_on: str=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_on = partition_on

    def predict_score(
            self, x: Union[pd.DataFrame, np.ndarray], proba: bool = False):
        """ Predicts the score for each row in the data across all partitions.

        The partition_on columns will be used to determine which model to use
        for each observation. If the partition_on column is not present in the
        data, the '__dataset__' model will be used.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted scores
        """

        x = pd.DataFrame(x).copy().reset_index(drop=True)

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict_score(x)

        else:
            partitions = self.partitions.keys()
            frames = []
            unq = list(x[self.partition_on].unique())

            # replace unknown partition values with __dataset__ for general model
            for u in unq:
                if u not in partitions:
                    x[self.partition_on] = x[self.partition_on].replace(u, '__dataset__')
                    unq.remove(u)
                    if "__dataset__" not in unq:
                        unq.append("__dataset__")

            partition_map = []
            for partition in unq:
                part = x[x[self.partition_on] == partition]
                idx = part.index

                # Use partition model first
                part_trans = self.partitions[partition]._transform(part)
                _base_value = self.partitions[partition].base_value

                scores = pd.Series(part_trans.sum(axis=1) + _base_value)
                scores.index = idx
                frames.append(scores)

                if proba:
                    [partition_map.append((i, partition)) for i in idx]
        
            all_scores = np.array(pd.concat(frames).sort_index())

        if proba:
            partition_map = np.array(partition_map)
            partition_map = partition_map[partition_map[:, 0].argsort()][:,1]
            return all_scores, partition_map

        return all_scores

    def predict_proba(self, x):
        """ Predicts the probability for each row in the data across all partitions.

        The partition_on columns will be used to determine which model to use
        for each observation. If the partition_on column is not present in the
        data, the '__dataset__' model will be used.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted probabilities
        """

        if self.partition_on is None:
            model = self.partitions['__dataset__']
            return model.predict_proba(x)

        scores, partition_map = self.predict_score(x, True)
        scores = (scores * 100).astype(int)
        
        def get_proba(p, score):
            mapp = self.partitions[str(p)]._calibration_map
            return mapp.get(score)

        scores = np.vectorize(get_proba)(partition_map, scores)

        return scores

    def predict(self, x, use_prob = False, threshold = 0.5):
        """ Predicts the target for each row in the data across all partitions.

        The partition_on columns will be used to determine which model to use
        for each observation. If the partition_on column is not present in the
        data, the '__dataset__' model will be used.

        Args:
            x (pd.DataFrame | np.ndarray): The x variables to predict.

        Returns:
            np.array: The predicted targets
        """

        # Get the score for each observation
        y_pred = self.predict_proba(x) if use_prob else self.predict_score(x)

        # Return 1 if feature value > threshold else 0
        pred = pd.Series(y_pred).map(lambda x: 1 if x >= threshold else 0)

        map_inv = self.partitions['__dataset__'].target_map.reverse

        if map_inv:
            return np.array(pred.map(map_inv))
        else:
            return np.array(pred)

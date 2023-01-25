import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ._base_model import BaseModel
from ..xcython.classification import XCProfiler
from sklearn.metrics import *

class Classifier(BaseModel):

    def __init__(self, max_depth=8, min_leaf_size=0.02, min_info_gain=0.02, alpha=0.01, map_calibration=True):
        super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

        self.__profiler = {}
        self._calibration_map = {}
        self.map_calibration = map_calibration

    def get_params(self):
        """ Returns the parameters of the model.

        Returns:
            dict: The model parameters.
        """

        params =  {
            'max_depth': self.max_depth,
            'min_leaf_size': self.min_leaf_size,
            'alpha': self.alpha,
            'min_info_gain': self.min_info_gain}

        return params

    def set_params(self, max_depth, min_leaf_size, min_info_gain, alpha):
        """ Sets the parameters of the model.

        Returns:
            dict: The model parameters.
        """

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        self.alpha = alpha

    def _map_calibration(self, x, y, smooth=15):
        """ Maps the associated probability for each possible feature score.

        Args:
            x (pandas.DataFrame): The x variables used for training.
            y (pandas.Series): The target series.
        """

        x = x.copy()

        # Make prediction and set to x
        x['pred'] = self.predict_score(x)

        # Store target in x df
        x['target'] = y.copy().values

        # Record prediction bins
        x['bin'] = pd.cut(x['pred'], [i / 100 for i in range(0, 101, 5)])

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

        # Store results dict to class variable
        return dict(wp.rolling(smooth, center=True, min_periods=5).mean()[0])

    def fit(self, x, y, id_columns=[]):

        x = x.copy()
        y = y.copy()
        if self.map_calibration:
            x_cal = x.copy()
            y_cal = y.copy()

        # Store meta data
        self.id_columns = id_columns
        x = x.drop(columns=id_columns)

        self._fetch_meta(x, y)
        
        # Preprocess data
        self._learn_encodings(x, y)
        x, y = self._encode(x, y)
        x, y = self._preprocess(x, y)

        # Create profiler
        profiler = XCProfiler(**self.get_params())
        profiler.fit(x.values, y.values)
        self.__profiler = profiler
        self._profile = profiler.profile
        self.base_value = profiler.base_value

        # Calibration map
        if self.map_calibration:
            if len(self.target_map) > 0:
                y_cal = y_cal.map(self.target_map)
            self._calibration_map = self._map_calibration(x_cal, y_cal, 15)

    def update_tree_params(self, idx, max_depth, min_info_gain, \
        min_leaf_size, alpha, x=None, y=None):

        self.__profiler.rebuild_tree(
            idx, max_depth, min_info_gain, min_leaf_size, alpha)
        self._profile = self.__profiler.profile

        if self.map_calibration and x is not None and y is not None:
            self._calibration_map = self._map_calibration(x, y, 15)

        return self

    def predict_score(self, x):
        trans = self._transform(x)
        scores = np.sum(trans, axis=1) + self.base_value

        return scores

    def predict_proba(self, x):
        scores = self.predict_score(x) * 100
        scores = scores.astype(int)

        scores = np.vectorize(self._calibration_map.get)(scores)

        return scores

    def predict(self, x, use_prob=False, threshold=0.5):
        scores = self.predict_proba(x) if use_prob else self.predict_score(x)
        pred = (scores > threshold).astype(int)

        if len(self.target_map_inv) > 0:
            pred = np.vectorize(self.target_map_inv.get)(pred)

        return pred

    def evaluate(self, x, y, use_prob=True, threshold=0.5):
        """ Evaluates the model metrics

        Args:
            x (pandas.DataFrame): The x data to test.
            y (pandas.Series): The true y values.

        Returns:
            dict: The model performance metrics.
        """

        # Make predictions
        y_pred = self.predict(x, use_prob, threshold)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)
        cr = classification_report(y, y_pred, output_dict=True)

        # Produce output
        evaluation = {
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'confusion_matrix': cm,
            'classification_report': cr
        }

        return evaluation

    def get_feature_importances(self):
        """ Calculates the feature importances for the model decision process.

        Returns:
            dict: The feature importances.
        """

        importances = {}
        total_importance = 0
        profile = self.get_profile()
        for i in ["numeric", "categorical"]:
            for feature, leaves in profile[i].items():        
                importance = 0
                for leaf in leaves:
                    importance += abs(leaf['score']) * np.log2(leaf['freq']*100)
                
                importances[feature] = importance
                total_importance += importance

        return {k: v/total_importance for k, v in sorted(
            importances.items(), key=lambda item: item[1])}
            
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ._base_model import BaseModel
from ._constructor import XConstructor
#from ..xcython.classification import XCProfiler
from sklearn.metrics import *
import copy


class XClassifier(BaseModel):
    
    def __init__(
        self,
        max_depth=8,
        min_info_gain=-1,
        min_leaf_size=-1,
        alpha=0.1,
        weight=0.05,
        power_degree=1,
        sigmoid_exponent=1,
        map_calibration=True
        ):
        
        super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

        self._constructs = []
        self._calibration_map = {}
        self._support_map = {}
        self._profile = []

        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent

        self.map_calibration = map_calibration
        self.feature_params = {}
        
    def get_params(self):
        """ Returns the parameters of the model.

        Returns:
            dict: The model parameters.
        """

        params =  {
            'max_depth': self.max_depth,
            'min_leaf_size': self.min_leaf_size,
            'alpha': self.alpha,
            'min_info_gain': self.min_info_gain,
            'weight': self.weight,
            'power_degree': self.power_degree,
            'sigmoid_exponent': self.sigmoid_exponent
            }

        return params

    def set_params(self, max_depth, min_leaf_size, min_info_gain, alpha, \
        weight, power_degree, sigmoid_exponent):
        """ Sets the parameters of the model.

        Returns:
            dict: The model parameters.
        """

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        self.alpha = alpha
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent

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
    
    def _normalise_score(self, score, _sum_min, _sum_max):
        """ Normalise the scores to fit between 0 - 1 relative to base value.

        Args:
            score (float): The score to normalise.

        Returns:
            float: The normalised score.
        """

        # Return 0 scores as float
        if score == 0:
            return 0

        # Negative scores normalise relative to worst case scenario
        elif score < 0:
            return abs(score) / _sum_min * self.base_value

        # Positive scores normalise relative to best case scenario
        else:
            return score / _sum_max * (1 - self.base_value)
    
    def _build_profile(self):
        self._profile = []
        _min_scores = np.empty(0)
        _max_scores = np.empty(0)

        for i in range(len(self._constructs)):
            xconst = self._constructs[i]
            _max_scores = np.append(_max_scores, xconst._max_score)
            _min_scores = np.append(_min_scores, xconst._min_score)

            # don't update the original leaf nodes
            self._profile.append([i for i in xconst._nodes])

        _sum_min = np.sum(_min_scores)
        _sum_max = np.sum(_max_scores)

        for idx in range(len(self._profile)):
            v = self._profile[idx]
            for i in range(len(v)):
                node = list(v[i])
                self._profile[idx][i][2] = self._normalise_score(
                    node[5], _sum_min, _sum_max)
        
        self._profile = np.array([np.array(x) for x in self._profile])
        
        return self

    def fit(self, x, y, id_columns=[], column_names=None, target_name='target'):

        x = x.copy()
        y = y.copy()

        # casts ndarray to pandas
        x, y = self._cast_to_pandas(x, y, target_name, column_names)
        
        if self.map_calibration:
            x_cal = x.copy()
            y_cal = y.copy()

        # Store meta data
        self.id_columns = id_columns

        if type(x) == np.ndarray:
            x = pd.DataFrame(x)

        x = x.drop(columns=id_columns)

        # Preprocess data
        x, y = self._coerce_dtypes(x, y)
        self._fetch_meta(x, y)
        self._learn_encodings(x, y)
        x, y = self._encode(x, y)
        x, y = self._preprocess(x, y)

        x = x.values
        y = y.values
        self.base_value = np.mean(y)

        # Dynamic min_leaf_size
        if self.min_leaf_size == -1:
            self.min_leaf_size = self.base_value / 10

        # Dynamic min_info_gain
        if self.min_info_gain == -1:
            self.min_info_gain = self.base_value / 10
        
        for i in range(x.shape[1]):
            f = x[:, i]
            xconst = XConstructor(
                regressor=False, # classifier
                max_depth=self.max_depth,
                min_info_gain=self.min_info_gain,
                min_leaf_size=self.min_leaf_size,
                alpha=self.alpha,
                weight=self.weight,
                power_degree=self.power_degree,
                sigmoid_exponent=self.sigmoid_exponent,
                )

            xconst.fit(f, y)
            self._constructs.append(xconst)
            
        self._build_profile()

        # Calibration map
        if self.map_calibration:
            if len(self.target_map) > 0:
                y_cal = y_cal.map(self.target_map)
            y_prob = self.predict_score(x_cal)
            self._calibration_map = self._map_calibration(y_cal, y_prob, 15)

        params = self.get_params()
        self.feature_params = {c: copy.copy(params) for c in self.columns}
        
        return self

    def update_feature_params(self, features, max_depth, min_info_gain, \
        min_leaf_size, alpha, weight, power_degree, sigmoid_exponent,
        tail_sensitivity=None, x=None, y=None):
        
        for feature in features:
            idx = self.columns.index(feature)

            self._constructs[idx].reconstruct(
                max_depth = max_depth,
                min_info_gain = min_info_gain,
                min_leaf_size = min_leaf_size,
                alpha = alpha,
                weight = weight,
                power_degree = power_degree,
                sigmoid_exponent = sigmoid_exponent,
                tail_sensitivity = tail_sensitivity
            )

            self.feature_params[feature].update({
            'max_depth': max_depth,
            'min_info_gain': min_info_gain,
            'min_leaf_size': min_leaf_size,
            'alpha': alpha,
            'weight': weight,
            'power_degree': power_degree,
            'sigmoid_exponent': sigmoid_exponent
        })

        self._build_profile()

        if self.map_calibration and x is not None and y is not None:
            y_prob = self.predict_score(x)
            self._calibration_map = self._map_calibration(y, y_prob, 15)

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

    def predict(self, x, use_prob=False, threshold=0.5, remap=True):
        scores = self.predict_proba(x) if use_prob else self.predict_score(x)
        pred = (scores > threshold).astype(int)

        if len(self.target_map_inv) > 0 and remap:
            pred = np.vectorize(self.target_map_inv.get)(pred)

        return pred

    def evaluate(self, x, y, use_prob=False, threshold=0.5):
        """ Evaluates the model metrics

        Args:
            x (pandas.DataFrame): The x data to test.
            y (pandas.Series): The true y values.

        Returns:
            dict: The model performance metrics.
        """

        # Make predictions
        y_prob = self.predict_proba(x) if use_prob else self.predict_score(x)
        y_prob = np.clip(y_prob, 0, 1) # because of rounding errors
        y_pred = (y_prob > threshold).astype(int)

        if len(self.target_map) > 0:
            y = y.copy().map(self.target_map)

        # Calculate metrics
        cm = confusion_matrix(y, y_pred).tolist()
        cr = classification_report(y, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y, y_prob)
        brier_loss = 1 - brier_score_loss(y, y_prob)
        cohen_kappa = cohen_kappa_score(y, y_pred)
        log_loss_score = log_loss(y, y_pred)

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


# class Classifier(BaseModel):

#     def __init__(self, max_depth=8, min_leaf_size=0.02, min_info_gain=0.02, alpha=0.01,\
#         weight=1, power_degree=1, sigmoid_exponent=0, map_calibration=True):
#         super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

#         self.__profiler = {}
#         self._calibration_map = {}
#         self._support_map = {}

#         self.weight = weight
#         self.power_degree = power_degree
#         self.sigmoid_exponent = sigmoid_exponent

#         self.map_calibration = map_calibration
#         self.feature_params = {}

#     def get_params(self):
#         """ Returns the parameters of the model.

#         Returns:
#             dict: The model parameters.
#         """

#         params =  {
#             'max_depth': self.max_depth,
#             'min_leaf_size': self.min_leaf_size,
#             'alpha': self.alpha,
#             'min_info_gain': self.min_info_gain,
#             'weight': self.weight,
#             'power_degree': self.power_degree,
#             'sigmoid_exponent': self.sigmoid_exponent
#             }

#         return params

#     def set_params(self, max_depth, min_leaf_size, min_info_gain, alpha, \
#         weight, power_degree, sigmoid_exponent):
#         """ Sets the parameters of the model.

#         Returns:
#             dict: The model parameters.
#         """

#         self.max_depth = max_depth
#         self.min_leaf_size = min_leaf_size
#         self.min_info_gain = min_info_gain
#         self.alpha = alpha
#         self.weight = weight
#         self.power_degree = power_degree
#         self.sigmoid_exponent = sigmoid_exponent

#     def _map_calibration(self, y, y_prob, smooth=15):
#         """ Maps the associated probability for each possible feature score.

#         Args:
#             x (pandas.DataFrame): The x variables used for training.
#             y (pandas.Series): The target series.
#         """

#         # Make prediction and set to x
#         x = pd.DataFrame(
#             {
#                 'y_prob': y_prob,
#                 'target': y.copy().values
#             })

#         # Record prediction bins
#         x['bin'] = pd.cut(x['y_prob'], [i / 100 for i in range(0, 101, 5)])

#         # Get target info grouped by bins
#         df = x.groupby('bin').agg({'target': ['mean', 'count']})

#         # Fix column formatting
#         df.columns = df.columns.map('_'.join)
#         df = df.rename(columns={'target_mean': 'tm', 'target_count': 'tc'})

#         # Record last and next scores for normalisation
#         df['lc'] = df['tc'].shift(1)
#         df['nc'] = df['tc'].shift(-1)

#         df['lm'] = df['tm'].shift(1)
#         df['nm'] = df['tm'].shift(-1)

#         # np.nan is the same as 0
#         df.fillna(0, inplace=True)

#         # Record rolling total for normalisation calc
#         df['rt'] = df['tc'].rolling(3, center=True, min_periods=2).sum()

#         # Scale counts to rolling total
#         df['tc_pct'] = df['tc'] / df['rt']
#         df['lc_pct'] = df['lc'] / df['rt']
#         df['nc_pct'] = df['nc'] / df['rt']

#         # Calculate weighted probability
#         df['wp'] = (df['lc_pct'] * df['lm']) + (df['tc_pct'] * df['tm']) + \
#             (df['nc_pct'] * df['nm'])

#         # Forward fill zero values
#         df['wp'] = df['wp'].replace(
#             to_replace=0, method='ffill')

#         # Get weighted probability and arrange
#         wp = df['wp']
#         wp = pd.DataFrame(np.repeat(wp.values, 5, axis=0))
#         wp = pd.concat([wp, wp.iloc[99]], ignore_index=True)

#         # Forward fill nan values
#         wp = wp.fillna(method='ffill')

#         # Fill missing values that could not be
#         # forward filled
#         wp = wp.fillna(0)

#         # Calculate support at each bin
#         s = df['tc']
#         s = pd.DataFrame(np.repeat(s.values, 5, axis=0))
#         s = pd.concat([s, s.iloc[99]], ignore_index=True)
#         s = s.fillna(method='ffill')
#         s = s.fillna(0)
#         self._support_map.update(dict(s[0]))

#         # Store results dict to class variable
#         return dict(wp.rolling(smooth, center=True, min_periods=5).mean()[0])

#     def fit(self, x, y, id_columns=[]):

#         x = x.copy()
#         y = y.copy()
#         if self.map_calibration:
#             x_cal = x.copy()
#             y_cal = y.copy()

#         # Store meta data
#         self.id_columns = id_columns
#         x = x.drop(columns=id_columns)

#         self._fetch_meta(x, y)
        
#         # Preprocess data
#         self._learn_encodings(x, y)
#         x, y = self._encode(x, y)
#         x, y = self._preprocess(x, y)

#         # Create profiler
#         profiler = XCProfiler(**self.get_params())
#         profiler.fit(x.values, y.values)
#         self.__profiler = profiler
#         self._profile = profiler.profile
#         self.base_value = profiler.base_value

#         # Calibration map
#         if self.map_calibration:
#             if len(self.target_map) > 0:
#                 y_cal = y_cal.map(self.target_map)
#             y_prob = self.predict_score(x_cal)
#             self._calibration_map = self._map_calibration(y_cal, y_prob, 15)

#         params = self.get_params()
#         self.feature_params = {c: copy.copy(params) for c in self.columns}

#     def update_tree_params(self, idx, max_depth, min_info_gain, \
#         min_leaf_size, alpha, weight, power_degree, sigmoid_exponent,
#         x=None, y=None):

#         self.__profiler.rebuild_tree(
#             idx, max_depth, min_info_gain, min_leaf_size, alpha, weight,
#             power_degree, sigmoid_exponent)

#         self._profile = self.__profiler.profile

#         if self.map_calibration and x is not None and y is not None:
#             y_prob = self.predict_score(x)
#             self._calibration_map = self._map_calibration(y, y_prob, 15)

#         self.feature_params[list(self.columns)[idx]].update({
#             'max_depth': max_depth,
#             'min_info_gain': min_info_gain,
#             'min_leaf_size': min_leaf_size,
#             'alpha': alpha,
#             'weight': weight,
#             'power_degree': power_degree,
#             'sigmoid_exponent': sigmoid_exponent
#         })

#         return self

#     def predict_score(self, x):
#         trans = self._transform(x)
#         scores = np.sum(trans, axis=1) + self.base_value

#         return scores

#     def predict_proba(self, x):
#         scores = self.predict_score(x) * 100
#         scores = scores.astype(int)

#         scores = np.vectorize(self._calibration_map.get)(scores)

#         return scores

#     def predict(self, x, use_prob=False, threshold=0.5, remap=True):
#         scores = self.predict_proba(x) if use_prob else self.predict_score(x)
#         pred = (scores > threshold).astype(int)

#         if len(self.target_map_inv) > 0 and remap:
#             pred = np.vectorize(self.target_map_inv.get)(pred)

#         return pred

#     def evaluate(self, x, y, use_prob=False, threshold=0.5):
#         """ Evaluates the model metrics

#         Args:
#             x (pandas.DataFrame): The x data to test.
#             y (pandas.Series): The true y values.

#         Returns:
#             dict: The model performance metrics.
#         """

#         # Make predictions
#         y_prob = self.predict_proba(x) if use_prob else self.predict_score(x)
#         y_prob = np.clip(y_prob, 0, 1) # because of rounding errors
#         y_pred = (y_prob > threshold).astype(int)

#         if len(self.target_map) > 0:
#             y = y.copy().map(self.target_map)

#         # Calculate metrics
#         cm = confusion_matrix(y, y_pred).tolist()
#         cr = classification_report(y, y_pred, output_dict=True)
#         roc_auc = roc_auc_score(y, y_pred)
#         brier_loss = 1 - brier_score_loss(y, y_prob)
#         cohen_kappa = cohen_kappa_score(y, y_pred)
#         log_loss_score = log_loss(y, y_pred)

#         # Produce output
#         evaluation = {
#             'confusion_matrix': cm,
#             'classification_report': cr,
#             'roc_auc': roc_auc,
#             'neg_brier_loss': brier_loss,
#             'log_loss': log_loss_score,
#             'cohen_kappa': cohen_kappa

#         }

#         return evaluation

#     def get_feature_importances(self):
#         """ Calculates the feature importances for the model decision process.

#         Returns:
#             dict: The feature importances.
#         """

#         importances = {}
#         total_importance = 0
#         profile = self.get_profile()
#         for i in ["numeric", "categorical"]:
#             for feature, leaves in profile[i].items():        
#                 importance = 0
#                 for leaf in leaves:
#                     importance += abs(leaf['score']) * np.log2(leaf['freq']*100)
                
#                 importances[feature] = importance
#                 total_importance += importance

#         return {k: v/total_importance for k, v in sorted(
#             importances.items(), key=lambda item: item[1])}
            
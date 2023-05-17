import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ._base_model import BaseModel
#from ..xcython.regression import XRProfiler
from ._constructor import XConstructor
from sklearn.metrics import *
import copy


class XRegressor(BaseModel):
    def __init__(
        self,
        max_depth=8,
        min_leaf_size=0.02,
        min_info_gain=0.02,
        alpha=0.01,
        tail_sensitivity=1,
        weight=1,
        power_degree=1,
        sigmoid_exponent=0,
        prediction_range=(-np.inf, np.inf)
        ):
        super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

        self.tail_sensitivity = tail_sensitivity
        self.weight = weight
        self.power_degree = power_degree
        self.sigmoid_exponent = sigmoid_exponent
        self.prediction_range = prediction_range

        self._constructs = []
        self._profile = []
        self.feature_params = {}
        self.samples = None

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
            'tail_sensitivity': self.tail_sensitivity,
            'weight': self.weight,
            'power_degree': self.power_degree,
            'sigmoid_exponent': self.sigmoid_exponent
            }

        return params

    def set_params(self, max_depth, min_leaf_size, min_info_gain,\
        alpha, weight, power_degree, sigmoid_exponent, tail_sensitivity):
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
        self.tail_sensitivity = tail_sensitivity
    
    def _build_profile(self, features=[]):
        self._profile = []

        for i in range(len(self._constructs)):
            xconst = self._constructs[i]

            if self.columns[i] in features or not features:
                for n in range(len(xconst._nodes)):
                    xconst._nodes[n][2] = xconst._nodes[n][5] / len(self.columns)
            
            # don't update the original leaf nodes
            self._profile.append(np.array([list(x) for x in xconst._nodes]))

        self._profile = np.array(self._profile, dtype=object)
        
        return self

    def fit(self, x, y, id_columns=[], column_names=None, target_name='target'):

        x = x.copy()
        y = y.copy()

        # casts ndarray to pandas
        x, y = self._cast_to_pandas(x, y, target_name, column_names)
        
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
        self.samples = y.size
        
        for i in range(x.shape[1]):
            f = x[:, i]
            xconst = XConstructor(
                regressor=True,
                max_depth=self.max_depth,
                min_info_gain=self.min_info_gain,
                min_leaf_size=self.min_leaf_size,
                alpha=self.alpha,
                tail_sensitivity=self.tail_sensitivity
                )

            xconst.fit(f, y)
            self._constructs.append(xconst)
            
        self._build_profile()

        params = self.get_params()
        self.feature_params = {c: copy.copy(params) for c in self.columns}
        
        return self

    def update_feature_params(self, features, max_depth, min_info_gain, \
        min_leaf_size, alpha, weight, power_degree, sigmoid_exponent, \
            tail_sensitivity):
        
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
                'sigmoid_exponent': sigmoid_exponent,
                'tail_sensitivity': tail_sensitivity
            })

        self._build_profile(features)
        
        return self

    def predict(self, x):
        trans = self._transform(x)
        pred = np.sum(trans, axis=1) + self.base_value

        return pred.clip(*self.prediction_range)

    def evaluate(self, X, y):

        y_pred = self.predict(X)

        mae = round(mean_absolute_error(y, y_pred), 4)
        mape = round(mean_absolute_percentage_error(y, y_pred), 4)
        r2 = round(r2_score(y, y_pred), 4)
        mse = round(mean_squared_error(y, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        evs = round(explained_variance_score(y, y_pred), 4)
        try:
            msle = round(mean_squared_log_error(y, y_pred), 4)
        except ValueError:
            msle = None

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


def _optimise_tail_sensitivity(model, X, y):
    
    params = model.get_params()
    
    centre = 0.1
    window = 0.1
    
    def trio_trial(a, b, c):
        _best_ts = None
        _best_metric = np.inf
        _best_i = None
    
        for i, ts in enumerate([a, b, c]):
            params.update({"tail_sensitivity": ts})
            model.update_feature_params(model.columns, **params)
            _metric = model.evaluate(X, y)['MAE']
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
    
    params.update({"tail_sensitivity": round(1+ts, 2)})
    model.update_feature_params(model.columns, **params)
    
    return model


# class Regressor(BaseModel):

#     def __init__(self, max_depth=8, min_leaf_size=0.02, min_info_gain=0.02,\
#         alpha=0.01, tail_sensitivity=1, prediction_range=None):
#         super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

#         self.tail_sensitivity = tail_sensitivity
#         self.prediction_range = prediction_range

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
#             'tail_sensitivity': self.tail_sensitivity
#             }

#         return params

#     def fit(self, x, y, id_columns=[]):

#         x = x.copy()
#         y = y.copy()

#         # Store meta data
#         self.id_columns = id_columns
#         x = x.drop(columns=id_columns)

#         self._fetch_meta(x, y)
        
#         # Preprocess data
#         self._learn_encodings(x, y)
#         x, y = self._encode(x, y)
#         x, y = self._preprocess(x, y)

#         # Create profiler
#         profiler = XRProfiler(**self.get_params())
#         profiler.fit(x.values, y.values)
#         self.__profiler = profiler
#         self._profile = profiler.profile
#         self.base_value = profiler.base_value

#     def predict(self, x):
#         trans = self._transform(x)
#         pred = np.sum(trans, axis=1) + self.base_value

#         return pred

#     def evaluate(self, X, y):

#         y_pred = self.predict(X)

#         mae = round(mean_absolute_error(y, y_pred), 4)
#         mape = round(mean_absolute_percentage_error(y, y_pred), 4)
#         r2 = round(r2_score(y, y_pred), 4)
#         mse = round(mean_squared_error(y, y_pred), 4)
#         rmse = round(np.sqrt(mse), 4)
#         evs = round(explained_variance_score(y, y_pred), 4)
#         try:
#             msle = round(np.sqrt(mean_squared_log_error(y, y_pred)), 4)
#         except ValueError:
#             msle = None

#         metrics = {
#             "Explained Variance Score": evs,
#             "Mean Absolute Error (MAE)": mae,
#             "Mean Absolute Percentage Error": mape,
#             "Mean Squared Error (MSE)": mse,
#             "Root Mean Squared Error (RMSE)": rmse,
#             "Root Mean Squared Logarithmic Error (RMSLE)": msle,
#             "R2 Score": r2
#         }

#         return metrics
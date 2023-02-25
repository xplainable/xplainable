import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from ._base_model import BaseModel
from ..xcython.regression import XRProfiler
from sklearn.metrics import *

class Regressor(BaseModel):

    def __init__(self, max_depth=8, min_leaf_size=0.02, min_info_gain=0.02,\
        alpha=0.01, tail_sensitivity=1, prediction_range=None):
        super().__init__(max_depth, min_leaf_size, min_info_gain, alpha)

        self.tail_sensitivity = tail_sensitivity
        self.prediction_range = prediction_range

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
            'tail_sensitivity': self.tail_sensitivity
            }

        return params

    def fit(self, x, y, id_columns=[]):

        x = x.copy()
        y = y.copy()

        # Store meta data
        self.id_columns = id_columns
        x = x.drop(columns=id_columns)

        self._fetch_meta(x, y)
        
        # Preprocess data
        self._learn_encodings(x, y)
        x, y = self._encode(x, y)
        x, y = self._preprocess(x, y)

        # Create profiler
        profiler = XRProfiler(**self.get_params())
        profiler.fit(x.values, y.values)
        self.__profiler = profiler
        self._profile = profiler.profile
        self.base_value = profiler.base_value

    def predict(self, x):
        trans = self._transform(x)
        pred = np.sum(trans, axis=1) + self.base_value

        return pred

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
            "Explained Variance Score": evs,
            "Mean Absolute Error (MAE)": mae,
            "Mean Absolute Percentage Error": mape,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "Root Mean Squared Logarithmic Error (RMSLE)": msle,
            "R2 Score": r2
        }

        return metrics
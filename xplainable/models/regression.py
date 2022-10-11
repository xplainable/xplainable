import pandas as pd
import json
from .. import client
from xplainable.utils import Loader
import numpy as np
import sklearn.metrics as skm
from xplainable.models._base_model import BaseModel
from urllib3.exceptions import HTTPError


class XRegressor(BaseModel):
    """ xplainable regression model.

    Args:
        max_depth (int): Maximum depth of feature decision nodes.
        min_leaf_size (float): Minimum observations pct allowed in each leaf.
        bin_alpha (float): Set the number of possible splits for each decision.
        min_info_gain (float): Minimum pct diff from base value for splits.
        tail_sensitivity (int): Amplifies values for high and low predictions.
        prediction_range (tuple/str): Sets upper and lower prediction range or autodetects.
    """

    def __init__(self, max_depth=20, min_leaf_size=0.002, min_info_gain=0.01,\
        bin_alpha=0.05, tail_sensitivity=1, prediction_range='auto', *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.__session = client.__session__

        self._profile = None
        self._feature_importances = None
        self._base_value = None
        self._categorical_columns = None
        self._numeric_columns = None
        self._params = None
        self._layers = None

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        self.bin_alpha = bin_alpha
        self.tail_sensitivity = tail_sensitivity
        self.prediction_range = prediction_range

    def _load_metadata(self, data):
        """ Loads model metadata into model object attrs

        Args:
            data (dict): Data to be loaded
        """
        self._profile = self._update_profile_inf(data['profile'])
        self._feature_importances = data['feature_importances']
        self._base_value = data['base_value']
        self._categorical_columns = data['categorical_columns']
        self._numeric_columns = data['numeric_columns']
        self._params = data['parameters']
        self._layers = data['layers']

    def fit(self, X, y, id_columns=[]):
        """ Fits training dataset to the model.
        
        Args:
            x (pandas.DataFrame): The x features to fit the model on.
            y (pandas.Series): The target feature for prediction.
            id_columns (list): list of id columns.
        """

        df = X.copy()
        target = y.name if y.name else 'target'
        df[target] = y.values

        params = {
            "model_name": self.model_name,
            "target": target,
            "id_columns": id_columns,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "min_info_gain": self.min_info_gain,
            "bin_alpha": self.bin_alpha,
            "tail_sensitivity": self.tail_sensitivity,
            "prediction_range": self.prediction_range,
            "layers": self._layers
        }

        loader = Loader("Training", "Training completed").start()

        response = self.__session.post(
            f'{self.hostname}/train/regression',
            params=params,
            files={'data': df.to_csv(index=False)}
            )

        if response.status_code == 200:
            content = json.loads(response.content)
            self._load_metadata(content["data"])

            loader.stop()

            return self

        elif response.status_code == 401:
            loader.stop()
            raise HTTPError(f"401 Unauthorised")

        else:
            loader.end = response.content
            loader.stop()
            raise HTTPError(f'{response} {response.content}') 
             
    def explain(self):
        """ Generates a model explanation URL

        Returns:
            str: URL
        """

        return f'https://app.xplainable.io/models/{self.model_name}'

    def predict(self, x):
        """ Predicts the y value of a given set of x variables.

        Args:
            x (pandas.DataFrame): A dataset containing the observations.

        Returns:
            numpy.array: Array of predictions.
        """

        x = x.copy()

        # Map all values to fitted scores
        x = self._transform(x)

        # Add base value to the sum of all scores
        return np.array(x.sum(axis=1) + self._base_value)

    def evaluate(self, X, y):
        """ Evaluates the model metrics

        Args:
            x (pandas.DataFrame): The x data to test.
            y (pandas.Series): The true y values.
            
        Returns:
            dict: The model performance metrics.
        """

        y_pred = self.predict(X)

        mae = round(skm.mean_absolute_error(y, y_pred), 4)
        mape = round(skm.mean_absolute_percentage_error(y, y_pred), 4)
        r2 = round(skm.r2_score(y, y_pred), 4)
        mse = round(skm.mean_squared_error(y, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        evs = round(skm.explained_variance_score(y, y_pred), 4)
        try:
            msle = round(skm.mean_squared_log_error(y, y_pred), 4)
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

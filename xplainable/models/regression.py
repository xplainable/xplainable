import json
from .. import client
import numpy as np
import sklearn.metrics as skm
from xplainable.models._base_model import BaseModel
from ..utils import get_response_content
from IPython.display import display, clear_output
import ipywidgets as widgets
from xplainable.client import __session__
import time
import xplainable

class XRegressor(BaseModel):
    """ xplainable regression model.

    Args:
        max_depth (int): Maximum depth of feature decision nodes.
        min_leaf_size (float): Minimum observations pct allowed in each leaf.
        bin_alpha (float): Set the number of possible splits for each decision.
        min_info_gain (float): Minimum pct diff from base value for splits.
        tail_sensitivity (int): Amplifies values for high and low predictions.
        prediction_range (list/str): Sets upper and lower prediction range or autodetects.
        validation_size (float): pct of data to hold for validation.
    """

    def __init__(self, max_depth=20, min_leaf_size=0.002, min_info_gain=0.01,\
        bin_alpha=0.05, tail_sensitivity=1, prediction_range=None, validation_size=0.2, *args, **kwargs):

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
        self.validation_size = validation_size
        self.min_prediction = None
        self.max_prediction = None

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
        self.prediction_range = data['parameters']['prediction_range']
        self.min_prediction = data['parameters']['min_prediction']
        self.max_prediction = data['parameters']['max_prediction']

    def _set_prediction_range(self):
        """ Sets the prection range as parameterised.
        """

        if self.prediction_range is None:
            self.min_prediction = -np.inf
            self.max_prediction = np.inf

        elif type(self.prediction_range) == list and len(self.prediction_range) == 2:

            if self.prediction_range[0] is not None:
                self.min_prediction = self.prediction_range[0]
            else:
                self.min_prediction = -np.inf

            if self.prediction_range[1] is not None:
                self.max_prediction = self.prediction_range[1]
            else:
                self.max_prediction = np.inf

        else:
            raise ValueError("Prediction range must be list of length 2 or None")

    def __get_progress(self):

        user_id = xplainable.__client__.get_user_data()['sub']

        current_output = None
        current_stage = "Initialising..."
        progress_instantiated = False

        stage = widgets.HTML(value=f'<p>{current_stage}</p>')
        stage.layout = widgets.Layout(margin='0 0 0 25px')


        bars = widgets.VBox([])

        screen = widgets.VBox([
            bars,
            widgets.HTML(f'<hr class="solid">', layout=widgets.Layout(margin='15px 0 0 0')),
            stage
        ])
        display(screen)

        while True:
            
            data = json.loads(__session__.get(f'{xplainable.__client__.hostname}/progress/{user_id}').content)

            if data is None:
                time.sleep(0.1)
                continue

            if data['stage'] != current_stage:
                stage.value = f'<p>{data["stage"]}</p>'
                current_stage = data['stage']

            if current_stage in ['done', 'failed']:
                if type(data['data']) == str:
                    return eval(data['data'])
                else:
                    return data['data']

            elif current_stage == 'optimising...':
                pipeline = data['pipeline']
                if not progress_instantiated:
                    iterations = [i['iterations'] for i in pipeline.values()]
                    bars.children = [
                        widgets.HBox([
                            widgets.IntProgress(
                                description=f"{i} ({v['type']})",
                                value=0,
                                min=0,
                                max=v['iterations']),
                            widgets.HTML(
                                value=f'0/{iterations[int(i)-1]}')]
                                ) for i, v in pipeline.items()]

                    #display(bars)
                    progress_instantiated = True

                if pipeline != current_output:
                    for i, v in pipeline.items():
                        bars.children[int(i)-1].children[0].value = v['progress']
                        mae = "-" if v['metric'] == "-" else round(float(v['metric']), 2)
                        bars.children[int(i)-1].children[1].value = f"{v['progress']}/{iterations[int(i)-1]} (mae: {mae})"

                        if v['status']:
                            bars.children[int(i)-1].children[0].bar_style = 'success'
                    current_output = pipeline
                time.sleep(0.1)

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

        self._set_prediction_range()

        params = {
            "model_name": self.model_name,
            "model_description": self.model_description,
            "target": target,
            "id_columns": id_columns,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "min_info_gain": self.min_info_gain,
            "bin_alpha": self.bin_alpha,
            "tail_sensitivity": self.tail_sensitivity,
            "prediction_range": self.prediction_range,
            "validation_size": self.validation_size,
            "layers": self._layers
        }

        response = self.__session.post(
            f'{xplainable.__client__.hostname}/train/regression',
            params=params,
            files={'data': df.to_csv(index=False)}
            )

        content = get_response_content(response)

        if content:
            model_data = self.__get_progress()
            if model_data:
                self._load_metadata(model_data["data"])

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
        y_pred = np.array(x.sum(axis=1) + self._base_value)

        # clip predictions
        y_pred = np.clip(y_pred, self.min_prediction, self.max_prediction)
        
        return y_pred

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

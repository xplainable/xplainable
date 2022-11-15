import json
from .. import client
import numpy as np
import sklearn.metrics as skm
from xplainable.models._base_model import BaseModel
from ..utils.api import get_response_content
from IPython.display import display
import ipywidgets as widgets
from xplainable.client import __session__
import time
import xplainable
import warnings
import pickle
import zlib

warnings.filterwarnings('ignore')

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
        self._layers = data['layers']
        
        params = data['parameters']
        self.prediction_range = params['prediction_range']
        self.min_prediction = params['min_prediction']
        self.max_prediction = params['max_prediction']
        self.max_depth = params['max_depth']
        self.min_leaf_size = params['min_leaf_size']
        self.min_info_gain = params['min_info_gain']
        self.bin_alpha = params['bin_alpha']
        self.validation_size = params['validation_size']

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

        current_output = None
        current_stage = "Initialising..."
        optimisation_instantiated = False
        train_complete = False

        stage = widgets.HTML(value=f'<p>{current_stage}</p>')

        stage_display = widgets.HBox([
            widgets.HTML(value=f'<p>STATUS: </p>'),
            stage
        ])

        stage_display.layout = widgets.Layout(margin='0 0 0 25px')

        train_bar = widgets.IntProgress(
            description=f"Fitting: ",
            value=0,
            min=0,
            max=100
            )

        pipeline_title = widgets.HTML(
            "<h3>Pipeline</h3>",
            layout=widgets.Layout(margin='0 0 0 25px'))

        train_pct = widgets.HTML(value=f'0%')
        train_display = widgets.HBox([train_bar, train_pct])
        opt_bars = widgets.VBox([])

        colA = widgets.VBox([
            pipeline_title,
            train_display,
            opt_bars
        ])

        colA.layout = widgets.Layout(
            min_width='420px'
        )

        hyperparameters_title = widgets.HTML(
            "<h3>Hyperparameters</h3>",
            layout=widgets.Layout(margin='0 0 0 25px'))

        hyperparameters = widgets.Box([
                widgets.HTML(f"""<h4>
                max_depth: {self.max_depth}<br>
                min_leaf_size: {self.min_leaf_size}<br>
                min_info_gain: {self.min_info_gain}<br>
                bin_alpha: {self.bin_alpha}<br>
                tail_sensitivity: {self.tail_sensitivity}
                </h4>""")])

        hyperparameters.layout = widgets.Layout(
            margin='0 0 0 25px',
            border='solid 1px',
            min_width='250px',
            padding='0 0 0 15px'
            )

        colB = widgets.VBox([
            hyperparameters_title,
            hyperparameters
        ])

        screen = widgets.VBox([
            widgets.HBox([colA, colB]),
            widgets.HTML(f'<hr class="solid">', layout=widgets.Layout(margin='15px 0 0 0')),
            stage_display
            ])

        display(screen)

        while True:
            
            data = json.loads(__session__.get(f'{xplainable.__client__.compute_hostname}/progress').content)
        
            if data is None:
                time.sleep(0.1)
                continue

            if data['stage'] != current_stage:
                current_stage = data['stage']
                stage.value = f'<p>{current_stage}</p>'

            if current_stage == 'failed':
                return

            if current_stage == 'initialising...':
                time.sleep(0.1)
                continue

            if not train_complete:
                p = data["train"]["progress"] / data["train"]["iterations"]
                v = int(p*100)
                train_bar.value = v
                train_pct.value = f'{v}%'

                if p == 1:
                    train_complete = True
                    train_bar.bar_style = 'success'

                continue
            
            pipeline = data['optimise']['pipeline']
            if not optimisation_instantiated:
                iterations = [i['iterations'] for i in pipeline.values()]
                opt_bars.children = [
                    widgets.HBox([
                        widgets.IntProgress(
                            description=f"{i} ({v['type']})",
                            value=0,
                            min=0,
                            max=v['iterations']),
                        widgets.HTML(
                            value=f'0/{iterations[int(i)-1]}')]
                            ) for i, v in pipeline.items()]

                optimisation_instantiated = True

            if pipeline != current_output:
                for i, v in pipeline.items():
                    opt_bars.children[int(i)-1].children[0].value = v['progress']
                    mae = "-" if v['metric'] == "-" else round(float(v['metric']), 2)
                    opt_bars.children[int(i)-1].children[1].value = f"{v['progress']}/{iterations[int(i)-1]} (mae: {mae})"

                    if v['status'] == 'stopped early':
                        opt_bars.children[int(i)-1].children[0].bar_style = 'warning'

                    elif v['status'] == 1:
                        opt_bars.children[int(i)-1].children[0].bar_style = 'success'

                current_output = pipeline
            time.sleep(0.1)

            if current_stage == 'done':
                return json.loads(data['data'])
                    

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

        bts = pickle.dumps(df)
        compressed_bytes = zlib.compress(bts)

        uploading_text = widgets.HTML("Uploading Data...")
        display(uploading_text)

        response = self.__session.post(
            f'{xplainable.__client__.compute_hostname}/train/regression',
            params=params,
            files={'data': compressed_bytes}
            )

        content = get_response_content(response)
        uploading_text.close()

        if content:
            model_data = self.__get_progress()
            if model_data:
                self._load_metadata(model_data["data"])

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

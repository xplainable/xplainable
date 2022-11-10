import pandas as pd
import json
from .. import client
from IPython.display import display, clear_output
from xplainable.models._base_model import BaseModel
import numpy as np
import sklearn.metrics as skm
from urllib3.exceptions import HTTPError
import xplainable
from xplainable.client import __session__
from xplainable.utils.api import get_response_content
import ipywidgets as widgets
import time
import warnings
import pickle
import zlib

warnings.filterwarnings('ignore')


class XClassifier(BaseModel):
    """ xplainable classification model.

    Args:
        max_depth (int): Maximum depth of feature decision nodes.
        min_leaf_size (float): Minimum observations pct allowed in each leaf.
        min_info_gain (float): Minimum pct diff from base value for splits.
        bin_alpha (float): Set the number of possible splits for each decision.
        optimise (bool): Optimises the model parameters during training.
        n_trials (int): Number of optimisation trials to run
        early_stopping (int): Stop optimisation early if no improvement as n trials.
        validation_size (float): pct of data to hold for validation.
    """

    def __init__(self, max_depth=12, min_leaf_size=0.015, min_info_gain=0.015,\
        bin_alpha=0.05, optimise=False, n_trials=30, early_stopping=15, validation_size=0.2,\
        max_depth_space=[4, 22, 2], min_leaf_size_space=[0.005, 0.08, 0.005],\
        min_info_gain_space=[0.005, 0.08, 0.005], opt_metric='weighted-f1', *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.__session = client.__session__

        self._profile = None
        self._calibration_map = None
        self._target_map = None
        self._feature_importances = None
        self._base_value = None
        self._categorical_columns = None
        self._numeric_columns = None

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain
        
        self.optimise = optimise
        self.n_trials = n_trials
        self.early_stopping = early_stopping
        self.max_depth_space = max_depth_space
        self.min_leaf_size_space = min_leaf_size_space
        self.min_info_gain_space = min_info_gain_space
        self.opt_metric = opt_metric

        self.bin_alpha = bin_alpha
        self.validation_size = validation_size

    def _load_metadata(self, data: dict):
        """ Loads model metadata into model object attrs

        Args:
            data (dict): Data to be loaded
        """
        self._profile = self._update_profile_inf(data['profile'])
        self._calibration_map = data['calibration_map']
        self._feature_importances = data['feature_importances']
        self._base_value = data['base_value']
        self._categorical_columns = data['categorical_columns']
        self._numeric_columns = data['numeric_columns']

        params = data['parameters']
        self.max_depth = params['max_depth']
        self.min_leaf_size = params['min_leaf_size']
        self.min_info_gain = params['min_info_gain']
        self.bin_alpha = params['bin_alpha']
        self.validation_size = params['validation_size']
        self.optimise = params['optimise']
        self.n_trials = params['n_trials']
        self.early_stopping = params['early_stopping']
        self.max_depth_space = params['max_depth_space']
        self.min_leaf_size_space = params['min_leaf_size_space']
        self.min_info_gain_space = params['min_info_gain_space']
        self.opt_metric = params['opt_metric']

    def __get_progress(self):

        current_stage = "Initialising..."
        optimisation_complete = False
        train_complete = False

        stage = widgets.HTML(value=f'<p>{current_stage}</p>')
        stage_display = widgets.HBox([
            widgets.HTML(value=f'<p>STATUS: </p>'),
            stage])
        view_button = widgets.Button(description='View Report')
        view_button.style.button_color = '#0080ea'

        publish_button = widgets.Button(description='Publish')
        publish_button.style.button_color = '#12b980'

        deploy_button = widgets.Button(description='Deploy')
        deploy_button.style.button_color = '#12b980'

        view_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        publish_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')
        deploy_button.layout = widgets.Layout(margin=' 10px 0 10px 10px')

        view_button.layout.visibility = "hidden"
        publish_button.layout.visibility = "hidden"
        deploy_button.layout.visibility = "hidden"
        
        footer = widgets.HBox([
            stage_display,
            view_button,
            publish_button,
            deploy_button
        ])

        footer.layout = widgets.Layout(margin='0 0 0 25px')

        optimisation_title = widgets.HTML(
            "<h3>Optimisation</h3>",
            layout=widgets.Layout(margin='0 0 0 25px'))

        training_title = widgets.HTML(
            "<h3>Training</h3>",
            layout=widgets.Layout(margin='0 0 0 25px'))

        hyperopt_bar = widgets.IntProgress(
            description=f"Trial: ",
            value=0,
            min=0,
            max=self.n_trials
            )

        hyperopt_count = widgets.HTML(value=f'0%')
        hyperopt_display = widgets.HBox([hyperopt_bar, hyperopt_count])

        fold_bar = widgets.IntProgress(
            description=f"CV Fold: ",
            value=0,
            min=0,
            max=5
            )

        fold_count = widgets.HTML(value=f'0%')
        fold_display = widgets.HBox([fold_bar, fold_count])

        train_bar = widgets.IntProgress(
            description=f"Fitting: ",
            value=0,
            min=0,
            max=100
            )

        train_pct = widgets.HTML(value=f'0%')
        train_display = widgets.HBox([train_bar, train_pct])

        divider = widgets.HTML(f'<hr class="solid">')

        colA = widgets.VBox([
            optimisation_title,
            fold_display,
            hyperopt_display,
            training_title,
            train_display
        ])

        colA.layout = widgets.Layout(
            min_width='400px'
        )
        
        style = {'description_width': 'initial', 'bar_color': '#e14067'}

        # Display Max Depth Chart
        max_depth_bar = widgets.IntProgress(
            description=f"max_depth: ",
            value=0,
            min=self.max_depth_space[0],
            max=self.max_depth_space[1],
            style=style
            )
        max_depth_val = widgets.HTML("0")
        max_depth_display = widgets.HBox([max_depth_bar, max_depth_val])

        # Display Min Leaf Size Chart
        min_leaf_size_bar = widgets.FloatProgress(
            description=f"min_leaf_size: ",
            value=0,
            min=self.min_leaf_size_space[0],
            max=self.min_leaf_size_space[1],
            style=style
            )
        min_leaf_size_val = widgets.HTML("0")
        min_leaf_size_display = widgets.HBox([min_leaf_size_bar, min_leaf_size_val])

        # Display Min Info Gain Chart
        min_info_gain_bar = widgets.FloatProgress(
            description=f"min_info_gain: ",
            value=0,
            min=self.min_info_gain_space[0],
            max=self.min_info_gain_space[1],
            style=style
            )
        min_info_gain_val = widgets.HTML("0")
        min_info_gain_display = widgets.HBox([min_info_gain_bar, min_info_gain_val])

        # Current Metric
        current_metric_bar = widgets.FloatProgress(
            description=f"Last F1: ",
            value=0,
            min=0,
            max=100
            )
        current_metric = widgets.HTML("0")
        current_metric_display = widgets.HBox([current_metric_bar, current_metric])

        # Best Metric
        best_metric_bar = widgets.FloatProgress(
            description=f"Best {self.opt_metric}: ",
            value=0,
            min=0,
            max=100,
            style={'description_width': 'initial'}
            )
        best_metric = widgets.HTML("0")
        best_metric_display = widgets.HBox([best_metric_bar, best_metric])
        best_metric_display.layout = widgets.Layout(margin='0 20px 10px 0')
        best_metric_bar.bar_style = 'success'

        hyperparameters_title = widgets.HTML(
            "<h3>Hyperparameters</h3>",
            layout=widgets.Layout(margin='0 0 0 25px'))

        hyperparameters = widgets.VBox([
                max_depth_display,
                min_leaf_size_display,
                min_info_gain_display,
                current_metric_display,
                divider,
                best_metric_display
                ])

        hyperparameters.children[0].layout = widgets.Layout(
            padding='0 0 0 15px'
        )

        hyperparameters.layout = widgets.Layout(
            margin='0 0 0 25px',
            #border='solid 1px',
            min_width='250px'
            )

        colB = widgets.VBox([
            hyperparameters_title,
            hyperparameters
        ])

        if self.optimise:
            training_title.layout.display = "none"
            train_display.layout.display = "none"
        else:
            optimisation_title.layout.display = "none"
            fold_display.layout.display = "none"
            hyperopt_display.layout.display = "none"
            divider.layout.display = "none"
            best_metric_display.layout.display = "none"

        screen = widgets.VBox([
            widgets.HBox([colA, colB]),
            widgets.HTML(f'<hr class="solid">', layout=widgets.Layout(margin='15px 0 0 0')),
            footer
            ])

        display(screen)

        def display_optimise():
            fold_bar.value = data["optimise"]["fold"]
            fold_count.value = f'{data["optimise"]["fold"]}/5'
            hyperopt_bar.value = data["optimise"]["iteration"]
            hyperopt_count.value = f'{data["optimise"]["iteration"]}/{self.n_trials}'
            if self.optimise:
                bst = round(data["optimise"]["best_metric"]*100, 2)
                best_metric.value = f'{bst}'
                best_metric_bar.value = bst

                current = round(data["optimise"]["metric"]*100, 2)
                current_metric_bar.value = current
                current_metric.value = f'{current}'

        def update_params(params):
            max_depth_bar.value = params['max_depth']
            max_depth_val.value = str(params['max_depth'])

            min_leaf_size_bar.value = params['min_leaf_size']
            min_leaf_size_val.value = str(params['min_leaf_size'])

            min_info_gain_bar.value = params['min_info_gain']
            min_info_gain_val.value = str(params['min_info_gain'])

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

            if current_stage == 'optimising parameters...':
                display_optimise()
                params = data['optimise']['params']
                if len(params) > 0:
                    update_params(params)
                continue

            if self.optimise:
                training_title.layout.display = "flex"
                train_display.layout.display = "flex"

            if not train_complete:
                if not optimisation_complete:
                    display_optimise()
                    if data["optimise"]["iteration"] < self.n_trials:
                        hyperopt_bar.bar_style = 'warning'
                    params = data['optimise']['params']

                    if len(params) > 0:
                        update_params(params)

                    if self.optimise:
                        hyperparameters_title.value = "<h3>Best Hyperparameters</h3>"

                    current_metric_display.layout.display = "none"
                    optimisation_complete = True

                p = data["train"]["progress"] / data["train"]["iterations"]
                v = int(p*100)
                train_bar.value = v
                train_pct.value = f'{v}%'

                if p == 1:
                    train_complete = True
                    train_bar.bar_style = 'success'
            
            elif current_stage == 'done':
                stage_display.layout.display = "none"
                view_button.layout.visibility = "visible"
                publish_button.layout.visibility = "visible"
                deploy_button.layout.visibility = "visible"

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

        params = {
            "model_name": self.model_name,
            "model_description": self.model_description,
            "target": target,
            "id_columns": id_columns,
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "min_info_gain": self.min_info_gain,
            "bin_alpha": self.bin_alpha,
            "optimise": self.optimise,
            "n_trials": self.n_trials,
            "early_stopping": self.early_stopping,
            "validation_size": self.validation_size,
            "max_depth_space": self.max_depth_space,
            "min_leaf_size_space": self.min_leaf_size_space,
            "min_info_gain_space": self.min_info_gain_space,
            "opt_metric": self.opt_metric
        }

        bts = pickle.dumps(df)
        compressed_bytes = zlib.compress(bts)

        url = f'{xplainable.__client__.compute_hostname}/train/binary'
        response = self.__session.post(
            url=url,
            params=params,
            files={'data': compressed_bytes}
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

    def predict_score(self, x):
        """ Scores an observation's propensity to fall in the positive class.
        
        Args:
            df (pandas.DataFrame): A dataset containing the observations.
            
        Returns:
            numpy.array: An array of Scores.
        """

        x = x.copy()

        # Map all values to fitted scores
        x = self._transform(x)

        # Add base value to the sum of all scores
        return np.array(x.sum(axis=1) + self._base_value)

    def predict_proba(self, x):
        """ Predicts probability an observation falls in the positive class.

        Args:
            x: A dataset containing the observations.
            
        Returns:
            numpy.array: An array of predictions.
        """

        x = x.copy()

        scores = self.predict_score(x) * 100
        scores = scores.astype(int).astype(str)

        scores = np.vectorize(self._calibration_map.get)(scores)

        return scores

    def predict(self, x, use_prob=False, threshold=0.5):
        """ Predicts if an observation falls in the positive class.
        
        Args:
            x (pandas.DataFrame): A dataset containing the observations.
            use_prob (bool): Uses 'probability' instead of 'score' if True.
            threshold (float): The prediction threshold.
            
        Returns:
            numpy.array: Array of predictions.
        """

        x = x.copy()

        # Get the score for each observation
        y_pred = self.predict_proba(x) if use_prob else self.predict_score(x)

        # Return 1 if feature value > threshold else 0
        pred = pd.Series(y_pred).map(lambda x: 1 if x >= threshold else 0)

        if self._target_map:
            return np.array(pred.map(self._target_map))
        else:
            return np.array(pred)

    def evaluate(self, x, y, use_prob=True, threshold=0.5):
        """ Evaluates the model metrics

        Args:
            x (pandas.DataFrame): The x data to test.
            y (pandas.Series): The true y values.
            use_prob (bool): Uses probability instead of score.
            threshold (float): The prediction threshold.

        Returns:
            dict: The model performance metrics.
        """

        # Make predictions
        y_pred = self.predict(x, use_prob, threshold)

        # Calculate metrics
        accuracy = skm.accuracy_score(y, y_pred)
        f1 = skm.f1_score(y, y_pred, average='weighted')
        precision = skm.precision_score(y, y_pred, average='weighted')
        recall = skm.recall_score(y, y_pred, average='weighted')
        cm = skm.confusion_matrix(y, y_pred)
        cr = skm.classification_report(y, y_pred, output_dict=True)

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
    
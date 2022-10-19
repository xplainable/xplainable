import pandas as pd
import json
from .. import client
from xplainable.utils import Loader
from xplainable.models._base_model import BaseModel
import numpy as np
import sklearn.metrics as skm
from urllib3.exceptions import HTTPError
from xplainable.utils import get_response_content


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
        bin_alpha=0.05, optimise=False, n_trials=30, early_stopping=15, validation_size=0.2,
        *args, **kwargs):

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
        self.bin_alpha = bin_alpha
        self.optimise = optimise
        self.n_trials = n_trials
        self.early_stopping = early_stopping
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
        }

        loader = Loader("Training", "Training completed").start()

        response = self.__session.post(
            f'{self.hostname}/train/binary',
            params=params,
            files={'data': df.to_csv(index=False)}
            )

        try:
            content = get_response_content(response)
            self._load_metadata(content["data"])

        except Exception as e:
            raise e

        finally:
            loader.stop()

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
    
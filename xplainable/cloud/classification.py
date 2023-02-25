import xplainable
from ..utils.api import get_response_content
from ..gui.cloud.displays import ClassificationProgress
from ..gui.components import BarGroup
from ._base_model import BaseModel

import pandas as pd
import json
from IPython.display import display

import numpy as np
import sklearn.metrics as skm

import ipywidgets as widgets
import time
import warnings
import pickle
import zlib
import threading
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
        bin_alpha=0.05, optimise=False, n_trials=30, early_stopping=15,
        validation_size=0.2, max_depth_space=[4, 22, 2],\
            min_leaf_size_space=[0.005, 0.08, 0.005],\
                min_info_gain_space=[0.005, 0.08, 0.005],\
                    opt_metric='weighted-f1', *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.model_name = None
        self.model_description = None
        self.target = None
        
        # Settings
        self.partition_on = None
        self.bin_alpha = bin_alpha
        self.validation_size = validation_size

        # Optimisation
        self.optimise = optimise
        self.n_trials = n_trials
        self.early_stopping = early_stopping
        self.max_depth_space = max_depth_space
        self.min_leaf_size_space = min_leaf_size_space
        self.min_info_gain_space = min_info_gain_space
        self.opt_metric = opt_metric

        # Main partition
        self._profile = None
        self._calibration_map = None
        self._target_map = None
        self._feature_importances = None
        self._base_value = None
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.min_info_gain = min_info_gain

        # All partitions
        self.partitions = {}

        # Feature info
        self._categorical_columns = None
        self._numeric_columns = None
        
    def _load_metadata(self, data: dict):
        """ Loads model metadata into model object attrs

        Args:
            data (dict): Data to be loaded
        """

        if "__dataset__" in data:
            if 'partition_on' in data['__dataset__']['data']:
                self.partition_on = data['__dataset__']['data']['partition_on']
            meta = data['__dataset__']['data']['data']
            self._profile = self._update_profile_inf(meta['profile'])
            self._calibration_map = meta['calibration_map']
            self._feature_importances = meta['feature_importances']
            self._base_value = meta['base_value']
            self._categorical_columns = meta['categorical_columns']
            self._numeric_columns = meta['numeric_columns']

            params = meta['parameters']
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
        
        for p, v in data.items():
            meta = v['data']['data']
            self.partitions[p] = {}
            self.partitions[p]['profile'] = self._update_profile_inf(
                meta['profile'])
            self.partitions[p]['calibration_map'] = meta['calibration_map']
            self.partitions[p]['feature_importances'] = meta[
                'feature_importances']
            self.partitions[p]['base_value'] = meta['base_value']

            params = meta['parameters']
            self.partitions[p]['max_depth'] = params['max_depth']
            self.partitions[p]['min_leaf_size'] = params['min_leaf_size']
            self.partitions[p]['min_info_gain'] = params['min_info_gain']

    def __get_progress(self, partitions, n_features):

        progress = ClassificationProgress(
            partitions, self.opt_metric, n_features)
        
        partition_map = {str(v): i for i, v in enumerate(partitions)}

        partition_status = {
            i: {
                'start_optimise': False,
                'done_optimise': False,
                'start_training': False,
                'done_training': False,
                'done': False
                }  for i, v in enumerate(partitions)}

        def process_optimise(idx, v):
            if not partition_status[idx]['start_optimise']:
                progress.start_partition_optimisation(idx, self.n_trials, 5)
                partition_status[idx]['start_optimise'] = True
            
            progress.update_optimisation_progress(
                idx, v['optimise']['iteration'])

            params = v['optimise']['params']
            if 'bin_alpha' in params:
                params.pop('bin_alpha')
            if 'bin_alpha' in params:
                params.pop('bin_alpha')

            progress.update_params(
                idx,
                best_metric=round(v['optimise']['best_metric']*100, 2),
                fold=v['optimise']['fold'],
                **params
                )

            if v['optimise']['status']:
                partition_status[idx]['done_optimise'] = True

        def processes_fit(idx, v):
            # TODO: This is a temporary fix
            #partition_status[idx]['start_training'] = True
            #partition_status[idx]['done_training'] = True
            #return

            if not partition_status[idx]['start_training']:
                progress.start_partition_training(idx)
                partition_status[idx]['start_training'] = True

            progress.update_training_progress(idx, v['train']['progress'])

            if v['train']['progress'] == n_features:
                partition_status[idx]['done_training'] = True

        def run_loop():
            while True:
                data = json.loads(
                    xplainable.client.__session__.get(
                        f'{xplainable.client.hostname}/v1/compute/progress').content
                        )

                if data is None:
                    time.sleep(0.1)
                    continue

                for partition, v in data.items():
                    
                    idx = partition_map[partition]

                    if partition_status[idx]['done']:
                        continue

                    if v["stage"] == 'initialising':
                        continue

                    if v['stage'] == 'optimising':
                        process_optimise(idx, v)
                    
                    elif v['stage'] == 'fitting':
                        if not partition_status[idx]['done_optimise']:
                            process_optimise(idx, v)
                        
                        processes_fit(idx, v)
                        
                    elif v['stage'] == 'done':
                        if not partition_status[idx]['done_training']:
                            processes_fit(idx, v)
                        
                        progress.complete_training(idx)
                        partition_status[idx]['done'] = True

                if all([i['done'] for i in partition_status.values()]):
                    self._load_metadata(data)
                    return
                
                time.sleep(0.2)

        # Display the screen
        screen = progress.generate_screen()
        display(screen)

        # Start loop (separate thread to maintain button interactivity)
        thread = threading.Thread(target=run_loop, daemon = True)
        thread.start()

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
            "partition_on": self.partition_on,
            "id_columns": str(id_columns),
            "max_depth": self.max_depth,
            "min_leaf_size": self.min_leaf_size,
            "min_info_gain": self.min_info_gain,
            "bin_alpha": self.bin_alpha,
            "optimise": self.optimise,
            "n_trials": self.n_trials,
            "early_stopping": self.early_stopping,
            "validation_size": self.validation_size,
            "max_depth_space": str(self.max_depth_space),
            "min_leaf_size_space": str(self.min_leaf_size_space),
            "min_info_gain_space": str(self.min_info_gain_space),
            "opt_metric": self.opt_metric
        }

        uploading_text = widgets.HTML("Uploading Data...")
        display(uploading_text)

        bts = pickle.dumps(df)
        compressed_bytes = zlib.compress(bts)

        url = f'{xplainable.client.hostname}/v1/compute/train/binary'
        response = xplainable.client.__session__.post(
            url=url,
            params=params,
            files={'data': compressed_bytes}
            )

        content = get_response_content(response)
        uploading_text.close()
        
        if content:
            partitions = ["__dataset__"]
            n_features = X.drop(columns=id_columns).shape[1]
            if self.partition_on is not None:
                # Get partition on value counts
                vc = X[self.partition_on].value_counts()

                # Add categories with more than 30 observations
                partitions = partitions + list(vc[vc > 30].index)
            
            self.__get_progress(partitions, n_features)
            

    def predict_score(self, x, use_partitions=False):
        """ Scores an observation's propensity to fall in the positive class.
        
        Args:
            df (pandas.DataFrame): A dataset containing the observations.
            
        Returns:
            numpy.array: An array of Scores.
        """
        # reset index to maintain predict order
        x = pd.DataFrame(x).copy().reset_index(drop=True)

        partitions = self.partitions.keys()
        frames = []
        if use_partitions:
            unq = x[self.partition_on].unique()
            for partition in unq:
                part = x[x[self.partition_on] == partition]
                idx = part.index
                # Use partition model first
                if partition in partitions:
                    part_trans = self._partition_transform(part, partition)
                    _base_value = self.partitions[partition]['base_value']
                # Use macro model if no partition found
                else:
                    part_trans = self._transform(part)
                    _base_value = self._base_value

                scores = part_trans.sum(axis=1) + _base_value
                scores.index = idx
                frames.append(scores)

            return np.array(pd.concat(frames).sort_index())

        else:
            # Map all values to fitted scores
            x = self._transform(pd.DataFrame(x))
            _base_value = self._base_value

            # Add base value to the sum of all scores
            return np.array(x.sum(axis=1) + _base_value)

    def predict_proba(self, x, use_partitions=False):
        """ Predicts probability an observation falls in the positive class.

        Args:
            x: A dataset containing the observations.
            
        Returns:
            numpy.array: An array of predictions.
        """

        scores = self.predict_score(x, use_partitions) * 100
        scores = scores.astype(int).astype(str)
        scores = np.vectorize(self._calibration_map.get)(scores)

        return scores

    def predict(self, x, use_partitions=False, use_prob=False, threshold=0.5):
        """ Predicts if an observation falls in the positive class.
        
        Args:
            x (pandas.DataFrame): A dataset containing the observations.
            use_prob (bool): Uses 'probability' instead of 'score' if True.
            threshold (float): The prediction threshold.
            
        Returns:
            numpy.array: Array of predictions.
        """

        # Get the score for each observation
        y_pred = self.predict_proba(x, use_partitions) if use_prob else \
            self.predict_score(x, use_partitions)

        # Return 1 if feature value > threshold else 0
        pred = pd.Series(y_pred).map(lambda x: 1 if x >= threshold else 0)

        if self._target_map:
            return np.array(pred.map(self._target_map))
        else:
            return np.array(pred)

    def predict_explain(self, x, use_partitions=False):
        """ Predicts if an observation falls in the positive class.
        
        Args:
            x (pandas.DataFrame): A dataset containing the observations.
            use_prob (bool): Uses 'probability' instead of 'score' if True.
            threshold (float): The prediction threshold.
            
        Returns:
            numpy.array: Array of predictions.
        """

        x = x.copy()

        output = self._transform(x)
        output['base_value'] = self._base_value
        output['probability'] = np.array(output.sum(axis=1))

        return output

    @staticmethod
    def _visual_eval(evaluation):
        def show_chart(cm):
            with output:

                fig, ax = plt.subplots(figsize=(2,2))
                sns.heatmap(
                    cm,
                    square=False,
                    annot=True,
                    cmap = sns.dark_palette(
                        "#12b980", reverse=False, as_cmap=True),
                    annot_kws={'fontsize': 18, 'color': 'white'},
                    fmt='d',
                    cbar=False,
                    linewidths=3,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1']
                )
                plt.show(fig)
        
        bars = BarGroup(
            items=['Accuracy', 'F1', 'Precision', 'Recall', 'AUC'],
            heading='Metrics',
            suffix='%')
        
        for e, v in evaluation.items():
            if e in bars.items:
                val = round(v*100, 2)
                bars.set_value(e, val)
        
        output = widgets.Output()
        output.layout = widgets.Layout(margin='25px 0 0 100px')
        with output:
            show_chart(evaluation['confusion_matrix'])
        
        screen = widgets.HBox([bars.show(), output])
        
        display(screen)

    def evaluate(self, x, y, visualise=True, use_prob=True, use_partitions=False, threshold=0.5):
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
        y_pred = self.predict(x, use_partitions, use_prob, threshold)

        # Calculate metrics
        accuracy = skm.accuracy_score(y, y_pred)
        f1 = skm.f1_score(y, y_pred, average='weighted')
        precision = skm.precision_score(y, y_pred, average='weighted')
        recall = skm.recall_score(y, y_pred, average='weighted')
        cm = skm.confusion_matrix(y, y_pred)
        #cr = skm.classification_report(y, y_pred, output_dict=True)
        fpr, tpr, thresholds = skm.roc_curve(y, y_pred, pos_label=2)
        auc = skm.auc(fpr, tpr)

        # Produce output
        evaluation = {
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc,
            'confusion_matrix': cm,
            
         #   'classification_report': cr
        }

        if visualise:
            return self._visual_eval(evaluation)

        return evaluation
    
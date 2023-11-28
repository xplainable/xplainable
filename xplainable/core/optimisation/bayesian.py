import hyperopt
from hyperopt import hp, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import fmin
from timeit import default_timer as timer
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from ..ml.classification import XClassifier
from ...utils.dualdict import TargetMap
import numpy as np
import time


class XParamOptimiser:
    """ Baysian optimisation for hyperparameter tuning XClassifier models.

        This optimiser is built on top of the Hyperopt library. It has
        pre-configured optimisation objectives and an easy way to set the
        search space for each parameter.

        The accepted metrics are:
            - 'macro-f1'
            - 'weighted-f1'
            - 'positive-f1'
            - 'negative-f1'
            - 'macro-precision'
            - 'weighted-precision'
            - 'positive-precision'
            - 'negative-precision'
            - 'macro-recall'
            - 'weighted-recall'
            - 'positive-recall'
            - 'negative-recall'
            - 'accuracy'
            - 'brier-loss'
            - 'log-loss'
            - 'roc-auc'

        Args:
            metric (str, optional): Optimisation metric. Defaults to 'roc-auc'.
            n_trials (int, optional): Number of trials to run. Defaults to 30.
            n_folds (int, optional): Number of folds for CV split. Defaults to 5.
            early_stopping (int, optional): Stops early if no improvement after n trials.
            shuffle (bool, optional): Shuffle the CV splits. Defaults to False.
            subsample (float, optional): Subsamples the training data.
            alpha (float, optional): Sets the alpha of the model.
            max_depth_space (list, optional): Sets the max_depth search space.
            min_leaf_size_space (list, optional): Sets the min_leaf_size search space.
            min_info_gain_space (list, optional): Sets the min_info_gain search space.
            ignore_nan_space (list, optional): Sets the ignore_nan search space.
            weight_space (list, optional): Sets the weight search space.
            power_degree_space (list, optional): Sets the power_degree search space.
            sigmoid_exponent_space (list, optional): Sets the sigmoid_exponent search space.
            verbose (bool, optional): Sets output amount. Defaults to True.
            random_state (int, optional): Random seed. Defaults to 1.
    """

    def __init__(
        self,
        metric='roc-auc',
        n_trials=30,
        n_folds=5,
        early_stopping=30,
        shuffle=False,
        subsample=1,
        alpha=0.01,
        max_depth_space=[4, 10, 2],
        min_leaf_size_space=[0.005, 0.05, 0.005],
        min_info_gain_space=[0.005, 0.05, 0.005],
        ignore_nan_space=[False, True],
        weight_space=[0, 1.2, 0.05],
        power_degree_space=[1, 3, 2],
        sigmoid_exponent_space=[0.5, 1, 0.1],
        verbose=True,
        random_state=1
        ):

        super().__init__()

        # Store class variables
        self.metric = metric
        self.early_stopping = early_stopping
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.subsample = subsample
        self.alpha = alpha
        self.verbose = verbose
        self.random_state = random_state

        self.max_depth_space = max_depth_space
        self.min_leaf_size_space = min_leaf_size_space
        self.min_info_gain_space = min_info_gain_space
        self.ignore_nan_space = ignore_nan_space

        self.weight_space = weight_space
        self.power_degree_space = power_degree_space
        self.sigmoid_exponent_space = sigmoid_exponent_space

        # Callback support
        self.callback = None
        self.iteration = 1
        self.best_score = -np.inf

        # Instantiate class objects
        self.x = None
        self.y = None
        self.id_columns = []
        self.models = {i: XClassifier(map_calibration=False) for i in range(n_folds)}
        self.folds = {}
        self.results = []

        self.metadata = {}

    def _cv_fold(self, params):
        """ Runs an iteration of cross-validation for a set of parameters.

        Args:
            params (dict): The parameters to be tested in the iteration.

        Returns:
            float: The average cross-validated score of the selected metric.
        """

        # Copy x and y class variables
        X_ = self.x.reset_index(drop=True)
        y_ = self.y.reset_index(drop=True)

        scores = []
        _has_nan = False
        start = time.time()
        # Run iteration over n_folds
        for i, model in self.models.items():
            # Instantiate and fit model
            model.update_feature_params(model.columns, **params)

            test_index = self.folds[i]['test_index']

            # Get predictions for fold
            if self.metric in ['brier-loss', 'log-loss', 'roc-auc']:
                y_prob = model.predict_score(X_.loc[test_index])
                y_prob = np.clip(y_prob, 0, 1)
                y_pred = (y_prob > 0.5).astype(int)
            else:
                y_pred = model.predict(X_.loc[test_index], remap=False)

            y_test = y_.loc[test_index]

            # Calculate the score for the fold
            if self.metric == 'macro-f1':
                scores.append(skm.f1_score(y_test, y_pred, average='macro',
                                           zero_division=0))

            elif self.metric == 'weighted-f1':
                scores.append(skm.f1_score(y_test, y_pred, average='weighted',
                                           zero_division=0))

            elif self.metric == 'positive-f1':
                scores.append(skm.f1_score(y_test, y_pred, average=None,
                                           zero_division=0)[1])

            elif self.metric == 'negative-f1':
                scores.append(skm.f1_score(y_test, y_pred, average=None,
                                           zero_division=0)[0])

            elif self.metric == 'macro-precision':
                scores.append(
                    skm.precision_score(y_test, y_pred, average='macro',
                                        zero_division=0))

            elif self.metric == 'weighted-precision':
                scores.append(
                    skm.precision_score(y_test, y_pred, average='weighted',
                                        zero_division=0))

            elif self.metric == 'positive-precision':
                scores.append(
                    skm.precision_score(y_test, y_pred, average=None,
                                        zero_division=0)[1])

            elif self.metric == 'negative-precision':
                scores.append(
                    skm.precision_score(y_test, y_pred, average=None,
                                        zero_division=0)[0])

            elif self.metric == 'macro-recall':
                scores.append(
                    skm.recall_score(y_test, y_pred, average='macro',
                                        zero_division=0))

            elif self.metric == 'weighted-recall':
                scores.append(
                    skm.recall_score(y_test, y_pred, average='weighted'))

            elif self.metric == 'positive-recall':
                scores.append(
                    skm.recall_score(y_test, y_pred, average=None,
                                     zero_division=0)[1])

            elif self.metric == 'negative-recall':
                scores.append(
                    skm.recall_score(y_test, y_pred, average=None,
                                     zero_division=0)[0])

            elif self.metric == 'accuracy':
                scores.append(skm.accuracy_score(y_test, y_pred))

            elif self.metric == 'brier-loss':
                # Negative as we want to minimise the score
                scores.append(1 - skm.brier_score_loss(y_test, y_prob))

            elif self.metric == 'log-loss':
                # Negative as we want to minimise the score
                scores.append(-skm.log_loss(y_test, y_prob))

            elif self.metric == 'roc-auc':
                try:
                    scores.append(skm.roc_auc_score(y_test, y_prob))
                except Exception as e:
                    scores.append(np.nan)
                    _has_nan = True

            else:
                scores.append(skm.f1_score(y_test, y_pred, average='weighted',
                                           zero_division=0))

            if self.callback:
                # fold callback
                self.callback.fold(i+1)

        score = np.nanmean(scores) if _has_nan else np.mean(scores)

        run_time = time.time() - start
        run_info = {
            'params': params,
            'score': score,
            'run_time': run_time
        }
        
        self.results.append(run_info)

        return score

    def _objective(self, params):
        """ The objective function for hyperopt optimisation.

        Args:
            params (dict): A set of params for an optimisation iteration.

        Returns:
            dict: Meta-data for hyperopt.
        """

        # Instantiate start timer for param set
        start = timer()

        # # Set the alpha (this is never optimised)
        # params['alpha'] = self.alpha

        # Callback is used for jupyter gui
        if self.callback:
            self.callback.update_params(**params)

        # Run cross validation and get score
        score = self._cv_fold(params)

        # Callback is used for jupyter gui
        if self.callback:
            # iteration callback
            self.callback.iteration(self.iteration)
            # metric callback
            if score > self.best_score:
                self.callback.metric(abs(round(score*100, 2)))
                self.best_score = score

            if self.iteration < self.n_trials:
                self.iteration += 1

        # Calculate the run time
        run_time = timer() - start

        return {"loss": -score, "params": params, "train_time": run_time,
                "status": hyperopt.STATUS_OK}

    def _instantiate(self):

        X_ = self.x.reset_index(drop=True)
        y_ = self.y.reset_index(drop=True)

        if self.shuffle:
            folds = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
                )

        else:
            folds = StratifiedKFold(n_splits=self.n_folds, shuffle=self.shuffle)

        self.folds = {i: {'train_index': train_index, 'test_index': test_index} for \
            i, (train_index, test_index) in enumerate(folds.split(X_, y_))}
        
        for i, v in self.folds.items():
            self.models[i].fit(
                X_.loc[v['train_index']],
                y_.loc[v['train_index']],
                id_columns=self.id_columns
                )

    def optimise(
            self, x: pd.DataFrame, y: pd.Series, id_columns: list = [],
            verbose: bool = True, callback=None):
        """ Get an optimised set of parameters for an XClassifier model.

        Args:
            x (pd.DataFrame): The x variables used for prediction.
            y (pd.Series): The true values used for validation.
            id_columns (list, optional): ID columns in dataset. Defaults to [].
            verbose (bool, optional): Sets output amount. Defaults to True.
            callback (any, optional): Callback for progress tracking.
            return_model (bool, optional): Returna model, else returns params

        Returns:
            dict: The optimised set of parameters.
        """

        start = time.time()

        # Store class variables
        self.x = x.copy()
        self.y = y.copy()
        self.id_columns = id_columns
        self._instantiate()

        self.callback = callback

        # Encode target categories if not numeric
        if self.y.dtype == 'object':

            # Cast as category
            target_ = self.y.astype('category')

            # Get the label map
            target_map = TargetMap(dict(enumerate(target_.cat.categories)), True)

            # Encode the labels
            self.y = self.y.map(target_map)

        # updates data types for cython handling
        n_cols = self.x.select_dtypes(include=np.number).columns.tolist()
        self.x[n_cols] = self.x[n_cols].astype('float64')
        self.y = self.y.astype('float64')

        # Apply subsampling
        if self.subsample < 1:
            self.x = self.x.sample(
                int(len(self.x) * self.subsample),
                random_state=self.random_state
                )

            self.y = self.y[self.x.index]

        self.x = self.x.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        # Instantiate the search space for hyperopt
        space = {}

        parameters = {
            'max_depth': self.max_depth_space,
            'min_leaf_size': self.min_leaf_size_space,
            'min_info_gain': self.min_info_gain_space,
            'ignore_nan': self.ignore_nan_space,
            'weight': self.weight_space,
            'power_degree': self.power_degree_space,
            'sigmoid_exponent': self.sigmoid_exponent_space
        }
        
        best_params = {}
        for n, p in parameters.items():
            
            if type(p) == list:
                space[n] = hp.choice(n, np.arange(*p))
            
            else:
                for i, model in self.models.items():
                    # Instantiate and fit model
                    model.update_feature_params(model.columns, **{n: p})
                
                best_params[n] = p
            
        # Instantiate trials
        trials = Trials()

        # Run hyperopt parameter search
        fmin(
            fn=self._objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            verbose=verbose,
            early_stop_fn=no_progress_loss(self.early_stopping),
            rstate=np.random.default_rng(self.random_state)
         )

        # Find maximum metric value across the trials
        idx = np.argmin(trials.losses())
        best_params.update(trials.trials[idx]["result"]["params"])

        # iteration callback completed
        if self.callback:
            self.callback.update_params(**best_params)

        # record metadata
        self.metadata.update({
            "optimisation_time": time.time() - start,
            "metric": self.metric,
            "n_trials": self.n_trials,
            "n_folds": self.n_folds,
            "early_stopping": self.early_stopping,
            "shuffle": self.shuffle,
            "subsample": self.subsample
        })

        # Return the best parameters
        return best_params

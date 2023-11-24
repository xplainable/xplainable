import hyperopt
from hyperopt import hp, tpe, Trials
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import fmin
from timeit import default_timer as timer
import sklearn.metrics as skm
import numpy as np
import pandas as pd
from ..models import XClassifier
from ...utils.dualdict import TargetMap


class NLPOptimiser:
    """ Baysian optimisation for hyperparameter tuning of xplainable models.

        Args:
            metric (str, optional): Optimisation metric. Defaults to 'f1'.
            early_stopping (int, optional): Stops early if no improvement.
            n_trials (int, optional): Number of trials to run. Defaults to 30.
            folds (int, optional): Number of folds for CV split. Defaults to 5.
            shuffle (bool, optional): Shuffle the CV splits. Defaults to False.
            subsample (float, optional): Subsamples the training data.
            alpha (float, optional): Sets the alpha of the model.
            balance (bool): True to handle class balance.
            random_state (int, optional): Random seed. Defaults to 1.
    """

    def __init__(self, nlp, drop_cols=[], metric='weighted-f1',
                 early_stopping=100, n_trials=30, n_folds=5,
                 shuffle=False, subsample=1, random_state=1,
                 min_word_freq_space = [0.0002, 0.005, 0.0002],
                 max_word_freq_space = [0.005, 0.5, 0.005],
                 min_ngram_freq_space = [0.0002, 0.005, 0.0002],
                 max_ngram_freq_space=[0.005, 0.5, 0.005]):

        super().__init__()

        # Store class variables
        self.drop_cols = drop_cols
        self.metric = metric
        self.early_stopping = early_stopping
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.subsample = subsample
        self.random_state = random_state

        self.min_word_freq_space = min_word_freq_space
        self.max_word_freq_space = max_word_freq_space
        self.min_ngram_freq_space = min_ngram_freq_space
        self.max_ngram_freq_space = max_ngram_freq_space

        # Instantiate class objects
        self.nlp = nlp
        self.x = None
        self.x_val = None
        self.y = None
        self.y_val = None 
        self.id_columns = []

    def _cv_fold(self, params):
        """ Runs an iteration of cross-validation for a set of parameters.

        Args:
            params (dict): The parameters to be tested in iteration.

        Returns:
            float: The average cross-validated score of the selected metric.
        """

        # Copy x and y class variables
        X_ = self.x.reset_index(drop=True)
        y_ = self.y.reset_index(drop=True)

        X_val_ = self.x_val.reset_index(drop=True)
        y_val_ = self.y_val.reset_index(drop=True)

        self.nlp.set_word_map(**params)

        X_trans = self.nlp.transform(X_)
        X_trans = X_trans.drop(columns=self.drop_cols)

        X_val_trans = self.nlp.transform(X_val_)
        X_val_trans = X_val_trans.drop(columns=self.drop_cols)

        X_trans, y = self._preprocess(X_trans, y_)
        X_val_trans, y_val = self._preprocess(X_val_trans, y_val_)

        model = XClassifier(map_calibration=False)
        model.fit(X_trans, y, id_columns=self.id_columns)

        y_pred = model.predict(X_val_trans)

        return skm.f1_score(y_val, y_pred, average='weighted')

    def _objective(self, params):
        """ The objective function for hyperopt optimisation.

        Args:
            params (dict): A set of params for an optimisation iteration.

        Returns:
            dict: Meta-data for hyperopt.
        """

        # Instantiate start timer for param set
        start = timer()

        # Run cross validation and get score
        score = self._cv_fold(params)

        # Calculate the run time
        run_time = timer() - start

        return {"loss": -1 * score, "params": params, "train_time": run_time,
                "status": hyperopt.STATUS_OK}

    def _preprocess(self, x, y):
        x = x.copy()
        y = y.copy()

        if y.dtype == 'object':

            # Cast as category
            target_ = y.astype('category')

            # Get the label map
            target_map = TargetMap(dict(enumerate(target_.cat.categories)), True)

            # Encode the labels
            y = y.map(target_map)
        

        # updates data types for cython handling
        n_cols = x.select_dtypes(include=np.number).columns.tolist()
        x[n_cols] = x[n_cols].astype('float64')
        y = y.astype('float64')

        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        return x, y

    def optimise(self, x, y, x_val, y_val, id_columns=[], verbose=True):
        """ Get an optimised set of parameters for an xplainable model.

        Args:
            x (pandas.DataFrame): The x variables used for prediction.
            y (pandas.Series): The true values used for validation.
            id_columns (list, optional): ID columns in dataset. Defaults to [].
            verbose (bool, optional): Sets output amount. Defaults to True.

        Returns:
            dict: The optimised set of parameters.
        """

        # Store class variables
        self.x = x.copy()
        self.y = y.copy()
        self.x_val = x_val.copy()
        self.y_val = y_val.copy()
        self.id_columns = id_columns

        # Instantiate the search space for hyperopt
        mnwf = self.min_word_freq_space
        mxwf = self.max_word_freq_space
        mnnf = self.min_ngram_freq_space
        mxnf = self.max_ngram_freq_space

        space = {
            'min_word_freq': hp.quniform('min_word_freq', mnwf[0], mnwf[1], mnwf[2]),
            'max_word_freq': hp.quniform('max_word_freq', mxwf[0], mxwf[1], mxwf[2]),
            'min_ngram_freq': hp.quniform('min_ngram_freq', mnnf[0], mnnf[1], mnnf[2]),
            'max_ngram_freq': hp.quniform('max_ngram_freq', mxnf[0], mxnf[1], mxnf[2])
            }

        # Instantiate trials
        trials = Trials()

        # Run hyperopt parameter search
        fmin(fn=self._objective,
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

        # Return the best parameters
        return trials.trials[idx]["result"]["params"]

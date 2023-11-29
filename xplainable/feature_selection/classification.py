
import pandas as pd
import numpy as np
import random
import time
import copy
import sklearn.metrics as skm
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from ..core.models import XClassifier
from ..utils.dualdict import TargetMap


class XClfFeatureSelector:
    """ Machine learning driven feature selection for classification models.
    
    Args:
        n_samples (int, optional): The number of samples to generate. Defaults to 50.
        alpha (float, optional): The XClassifier aplha value. Defaults to 0.01.
        random_state (int, optional): The random state. Defaults to 1.

    Raises:
        AssertionError: If no samples are found.
        AssertionError: If the sample sizes do not match.
        AssertionError: If no samples are found.

    Returns:
        XClfFeatureSelector: The XClfFeatureSelector object.
    """

    def __init__(
        self,
        n_samples=50,
        alpha=0.01,
        random_state=1
        ):
        
        super().__init__()

        # Store class variables
        self.n_samples = n_samples
        self.alpha = alpha
        self.random_state = random_state
        random.seed(random_state)

        # Instantiate class objects
        self.x_train = None
        self.y_train = None
        self.id_columns = []
        self.columns = []
        self.results = []
        self.model = XClassifier()
        self.samples = []

        self.feature_scores = {}
        self.feature_counts = {}

    def _assert_all_values_equal(self, dictionary):
        values = list(dictionary.values())
        if len(values) > 0:
            first_value = values[0]
            if first_value == 0:
                AssertionError("No samples founds")
            for value in values[1:]:
                if value != first_value:
                    raise AssertionError("Sample size mismatch")
        else:
            raise AssertionError("No samples founds")

    def _get_even_samples(self, columns, n_samples):

        # Check if a remainer exisits
        r = len(columns) % 2 != 0

        sample_size = int((len(columns) - r) / 2)
        samples = []
        counts = {i: 0 for i in columns}

        for i in range(int(n_samples/2)):
            samp1 = random.sample(columns, sample_size)
            samp2 = [c for c in columns if c not in samp1]
            samples.append(samp1)
            samples.append(samp2)

            for s in samp1:
                counts[s] += 1

            for s in samp2:
                counts[s] += 1

        self._assert_all_values_equal(counts)

        return samples

    def fit(
            self, x: pd.DataFrame, y: pd.Series, id_columns: list = []):
        """ Get an optimised set of parameters for an XClassifier model.

        Args:
            x (pd.DataFrame): The x variables used for prediction.
            y (pd.Series): The true values used for validation.
            id_columns (list, optional): ID columns in dataset. Defaults to [].
            return_model (bool, optional): Returna model, else returns params

        Returns:
            dict: The optimised set of parameters.
        """

        start = time.time()

        # Store class variables
        x = x.copy()
        y = y.copy()
        
        # Encode target categories if not numeric
        if y.dtype == 'object':

            # Cast as category
            target_ = y.astype('category')

            # Get the label map
            target_map = TargetMap(dict(enumerate(target_.cat.categories)), True)

            # Encode the labels
            y = y.map(target_map)
        
        if len(id_columns) > 0:
            x = x.drop(columns=id_columns)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.25, random_state=42)
        
        self.id_columns = id_columns

        self.columns = list(self.X_train.columns)
        self.feature_scores = {c: 0 for c in self.columns}
        self.feature_counts = {c: 0 for c in self.columns}
                
        self.samples = self._get_even_samples(self.columns, self.n_samples)
        
        self.model.fit(self.X_train, self.y_train, id_columns=self.id_columns,
            alpha=self.alpha)
        
        params = self.model.params.to_json()
        
        for samp in tqdm(self.samples):
        
            # Instantiate and fit model
            self.model.update_feature_params(samp, **params)

            remaining = [c for c in self.columns if c not in samp]
            remaining_params = copy.copy(params)
            remaining_params['max_depth'] = 0
            self.model.update_feature_params(remaining, **remaining_params)

            # Get predictions for fold
            y_prob = self.model.predict_score(self.X_test)
            y_prob = np.clip(y_prob, 0, 1)

            score = skm.roc_auc_score(self.y_test, y_prob)

            try:
                fimp = self.model.feature_importances
            except:
                fimp = {c: 0 for c in self.columns}

            for f, i in fimp.items():
                s = i * (1 + score)
                self.feature_scores[f] += s
                self.feature_counts[f] += 1

        # Return the best parameters
        return dict(sorted(
            self.feature_scores.items(), key=lambda item: item[1], reverse=True
            ))
    
    def get_n_features(self, n=10):
        """ Get the top n features.
        
        Args:
            n (int, optional): The number of features to return. Defaults to 10.

        Returns:
            list: The top n features.
        """
        feature_scores_df = pd.DataFrame(
            self.feature_scores.items(),
            columns=['feature', 'score']).sort_values(
            'score', ascending=False).reset_index(drop=True)
        
        return list(feature_scores_df.head(n)['feature'])

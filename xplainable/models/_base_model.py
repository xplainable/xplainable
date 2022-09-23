from .. import client
from datetime import datetime
import dill
import os
import numpy as np
from sklearn.metrics import *


class BaseModel:

    def __init__(self, model_name, hostname):

        self.model_name = model_name
        self.hostname = hostname
        
    @staticmethod
    def _update_profile_inf(profile):

        for f, val in profile['numeric'].items():
            for i, v in val.items():
                if v['upper'] == 'inf':
                    profile['numeric'][f][i]['upper'] = np.inf
                if v['lower'] == '-inf':
                    profile['numeric'][f][i]['lower'] = -np.inf

        return profile

    @staticmethod
    def _map_categorical(x, mapp):
        
        for v in mapp.values():
            if x in v['categories']:
                return v['score']

        return 0

    @staticmethod
    def _map_numeric(x, mapp):
        
        for v in mapp.values():
            if x <= v['upper'] and x > v['lower']:
                return v['score']

        return 0

    def _transform(self, x):
        """ Transforms a dataset into the model weights.
        
        Args:
            x (pandas.DataFrame): The dataframe to be transformed.
            
        Returns:
            pandas.DataFrame: The transformed dataset.
        """

        if len(self._profile) == 0:
            raise ValueError('Fit the model before transforming')

        x = x.copy()

        n_cols = x.select_dtypes(include=np.number).columns.tolist()
        x[n_cols] = x[n_cols].astype('float64')

        # Get column names from training data
        columns = self._categorical_columns + self._numeric_columns

        # Filter x to only relevant columns
        x = x[[i for i in columns if i in list(x)]]

        # Map score for all categorical features
        for col in self._categorical_columns:
            if col in self._profile["categorical"].keys():
                mapp = self._profile["categorical"][col]
                x[col] = x[col].apply(self._map_categorical, args=(mapp,))

            else:
                x[col] = np.nan

        # Map score for all numeric features
        for col in self._numeric_columns:
            if col in self._profile["numeric"].keys():
                mapp = self._profile["numeric"][col]
                x[col] = x[col].apply(self._map_numeric, args=(mapp,))

            else:
                x[col] = np.nan

        return x

    def save_model(self, filename=None):
        if not filename:
            
            ts = str(datetime.utcnow().timestamp()).replace('.', '')
            filename = f'xplainable_{ts}'

        if not filename.endswith(".pkl"):
            filename = f'{filename}.pkl'

        isExist = os.path.exists('saved_models/')

        if not isExist:
            os.makedirs('saved_models/')
        
        filepath = f'saved_models/{filename}'

        with open(filepath, 'wb') as outp:
            dill.dump(self, outp)
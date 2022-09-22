import pandas as pd
import json
from .. import client
from datetime import datetime
import dill
import os
from IPython.display import display, clear_output
import ipywidgets as widgets
from src.xplainable.utils import Loader
import numpy as np
from sklearn.metrics import *


class XClassifier:

    def __init__(self, model_name, hostname):
        self.__session = client.__session__
        self.hostname = hostname

        self.model_name = model_name

        self.__profile = None
        self.__feature_importances = None
        self.__calibration_map = None
        self.__base_value = None
        self.__categorical_columns = None
        self.__numeric_columns = None
        self.__target_map = None
        self.__params = None

    def _update_profile_inf(self, profile):

        for f, val in profile['numeric'].items():
            for i, v in val.items():
                if v['upper'] == 'inf':
                    profile['numeric'][f][i]['upper'] = np.inf
                if v['lower'] == '-inf':
                    profile['numeric'][f][i]['lower'] = -np.inf

        return profile

    def get_params(self):
        return self.__params

    def get_feature_importances(self):
        return self.__feature_importances

    def _load_metadata(self, data):
        self.__profile = self._update_profile_inf(data['profile'])
        self.__feature_importances = data['feature_importances']
        self.__calibration_map = data['calibration_map']
        self.__base_value = data['base_value']
        self.__categorical_columns = data['categorical_columns']
        self.__numeric_columns = data['numeric_columns']

    def _map_categorical(self, x, mapp):
        
        for v in mapp.values():
            if x in v['categories']:
                return v['score']

        return 0

    def _map_numeric(self, x, mapp):
        
        for v in mapp.values():
            if x <= v['upper'] and x > v['lower']:
                return v['score']

        return 0

    def fit(self, df):

        files = {'data': df.to_csv(index=False)}

        style = {'description_width': 'initial'}

        target = widgets.Dropdown(options=df.columns, description='Target:')
        id_columns = widgets.SelectMultiple(options=df.columns, style=style, description='ID Columns:')
        max_depth = widgets.IntSlider(value=8, min=2, max=100, step=1, description='max_depth:', style=style)
        min_leaf_size = widgets.FloatSlider(value=0.005, min=0.001, max=0.2, step=0.001, readout_format='.3f', description='min_leaf_size:', style=style)
        min_info_gain = widgets.FloatSlider(value=0.005, min=0.001, max=0.2, step=0.001, readout_format='.3f', description='min_info_gain:', style=style)
        bin_alpha = widgets.FloatSlider(value=0.05, min=0.01, max=0.5, step=0.01, description='bin_alpha:', style=style)
        optimise = widgets.Dropdown(value=False, options=[True, False], description='optimise:', style=style)
        n_trials = widgets.IntSlider(value=30, min=5, max=150, step=5, description='n_trials:', style=style)
        early_stopping = widgets.IntSlider(value=15, min=5, max=50, step=5, description='early_stopping:', style=style)

        # button, output, function and linkage
        train_button = widgets.Button(description='Train Model')

        # close button
        close_button = widgets.Button(description='Close')

        outt = widgets.Output()

        # display
        warning = ''
        if self.__profile:
            warning = ' [model already fitted]'
        header = widgets.HTML(f"<h2>Model: {self.model_name}{warning}</h2>", layout=widgets.Layout(height='auto'))
        subheader1 = widgets.HTML(f"<h4>Target</h4>", layout=widgets.Layout(height='auto'))
        subheader2 = widgets.HTML(f"<h4>Optimise</h4>", layout=widgets.Layout(height='auto'))
        subheader3 = widgets.HTML(f"<h4>Set params manually</h4>", layout=widgets.Layout(height='auto'))
        divider = widgets.HTML(f'<hr class="solid">', layout=widgets.Layout(height='auto'))

        def close(_=None):
            header.close()
            subheader1.close()
            subheader2.close()
            subheader3.close()
            divider.close()
            train_button.close()
            close_button.close()
            target.close()
            id_columns.close()
            max_depth.close()
            min_leaf_size.close()
            min_info_gain.close()
            bin_alpha.close()
            optimise.close()
            n_trials.close()
            early_stopping.close()

        def close_button_click(_):
            close()
            clear_output()

        def on_button_clicked(b):
            with outt:
                clear_output()
                
                loader = Loader("Training", "Training completed").start()

                params = {
                    'model_name': self.model_name,
                    'target': target.value,
                    'id_columns': list(id_columns.value),
                    'max_depth': max_depth.value,
                    'min_leaf_size': min_leaf_size.value,
                    'min_info_gain': min_info_gain.value,
                    'bin_alpha': bin_alpha.value,
                    'optimise': optimise.value,
                    'n_trials': n_trials.value,
                    'early_stopping': early_stopping.value
                }

                response = self.__session.post(
                    f'{self.hostname}/models/classification/train',
                    params=params,
                    files=files
                    )

                if response.status_code == 200:
                    content = json.loads(response.content)
                    self.__profile = self._update_profile_inf(content['data']['profile'])
                    self.__feature_importances = content['data']['feature_importances']
                    self.__calibration_map = content['data']['calibration_map']
                    self.__base_value = content['data']['base_value'] 
                    self.__categorical_columns = content['data']['categorical_columns']
                    self.__numeric_columns = content['data']['numeric_columns']
                    self.__params = content['data']['parameters']

                    loader.stop()
                    close()

                    return self

                else:
                    loader.end = response.content
                    loader.stop()
                    close()
                    return response


        train_button.on_click(on_button_clicked)
        train_button.style.button_color = '#0080ea'

        close_button.on_click(close_button_click)

        display(header)
        
        return widgets.VBox([
            widgets.HBox([
                widgets.VBox([subheader1, target, divider, id_columns]),
                widgets.VBox([subheader2, optimise, divider, n_trials, early_stopping]),
                widgets.VBox([subheader3, max_depth, min_leaf_size, min_info_gain, bin_alpha])
            ]),
            divider,
            widgets.HBox([train_button, close_button]),
            outt
        ])

    def xplain(self):
        metadata = {
                "base_value": self.__base_value,
                "profile": self.__profile,
                "calibration_map": self.__calibration_map,
                "feature_importances": self.__feature_importances,
            }

        return metadata

    def __transform(self, x):
        """ Transforms a dataset into the model weights.
        
        Args:
            x (pandas.DataFrame): The dataframe to be transformed.
            
        Returns:
            pandas.DataFrame: The transformed dataset.
        """

        if len(self.__profile) == 0:
            raise ValueError('Fit the model before transforming')

        x = x.copy()

        n_cols = x.select_dtypes(include=np.number).columns.tolist()
        x[n_cols] = x[n_cols].astype('float64')

        # Get column names from training data
        columns = self.__categorical_columns + self.__numeric_columns

        # Filter x to only relevant columns
        x = x[[i for i in columns if i in list(x)]]

        # Apply preprocessing transformations
        #if self.preprocessor:
        #    x = self.preprocessor.transform(x)

        # Map score for all categorical features
        for col in self.__categorical_columns:
            if col in self.__profile["categorical"].keys():
                mapp = self.__profile["categorical"][col]
                x[col] = x[col].apply(self._map_categorical, args=(mapp,))

            else:
                x[col] = np.nan

        # Map score for all numeric features
        for col in self.__numeric_columns:
            if col in self.__profile["numeric"].keys():
                mapp = self.__profile["numeric"][col]
                x[col] = x[col].apply(self._map_numeric, args=(mapp,))

            else:
                x[col] = np.nan

        return x
    
    def predict_score(self, x):
        """ Scores an observation's propensity to fall in the positive class.
        
        Args:
            df (pandas.DataFrame): A dataset containing the observations.
            
        Returns:
            numpy.array: An array of Scores.
        """

        x = x.copy()

        # Map all values to fitted scores
        x = self.__transform(x)

        # Add base value to the sum of all scores
        return np.array(x.sum(axis=1) + self.__base_value)

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

        scores = np.vectorize(self.__calibration_map.get)(scores)

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

        if self.__target_map:
            return np.array(pred.map(self.__target_map))
        else:
            return np.array(pred)

    def evaluate(self, x, y, use_prob=True, threshold=0.5):
        """ Evaluates the model metrics
        Args:
            x (pandas.DataFrame): The x data to test.
            y (pandas.Series): The true y values.
        Returns:
            dict: The model performance metrics.
        """

        # Make predictions
        y_pred = self.predict(x, use_prob, threshold)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)
        cr = classification_report(y, y_pred, output_dict=True)

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
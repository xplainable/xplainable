import pandas as pd
import json
from .. import client
from datetime import datetime
import dill
import os
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import Layout, interact, interact_manual, fixed
from src.xplainable.utils import Loader
import numpy as np
import sklearn.metrics as skm
import time

class XRegressor:

    def __init__(self, model_name, hostname):
        self.__session = client.__session__
        self.hostname = hostname

        self.model_name = model_name

        self.__profile = None
        self.__feature_importances = None
        self.__base_value = None
        self.__categorical_columns = None
        self.__numeric_columns = None
        self.__params = None
        self.__layers = None

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
        output = widgets.Output()

        '''------------------------------------------ STYLE ------------------------------------------'''
        style = {'description_width': 'initial'}


        '''------------------------------------------ HEADER ------------------------------------------'''
        warning = ' [already fitted]' if self.__profile else ''
        header = widgets.HTML(f"<h2>Model: {self.model_name}{warning}</h2>", layout=widgets.Layout(height='auto'))


        '''------------------------------------------ TAB 1 ------------------------------------------'''


        '''--------------------- COLUMN A ---------------------'''
        title_a1 = widgets.HTML(f"<h4>Target</h4>", layout=widgets.Layout(height='auto'))
        title_a2 = widgets.HTML(f"<h4>ID Column(s)</h4>", layout=widgets.Layout(height='auto'))

        target = widgets.Dropdown(options=df.columns)
        id_columns = widgets.SelectMultiple(options=df.columns, style=style)

        colA = widgets.VBox([title_a1, target, title_a2, id_columns])


        '''--------------------- COLUMN B ---------------------'''
        title_b = widgets.HTML(f"<h4>Parameters (for initial fit)</h4>", layout=widgets.Layout(height='auto'))
        max_depth = widgets.IntSlider(value=25, min=2, max=100, step=1, description='max_depth:', style=style)
        min_leaf_size = widgets.FloatSlider(value=0.01, min=0.001, max=0.2, step=0.001, readout_format='.3f', description='min_leaf_size:', style=style)
        min_info_gain = widgets.FloatSlider(value=0.015, min=0.001, max=0.2, step=0.001, readout_format='.3f', description='min_info_gain:', style=style)
        bin_alpha = widgets.FloatSlider(value=0.2, min=0.01, max=0.5, step=0.01, description='bin_alpha:', style=style)
        tail_sensitivity = widgets.FloatSlider(value=1.1, min=1.0, max=1.5, step=0.01, description='tail_sensitivity:', style=style)

        colB = widgets.VBox([title_b, max_depth, min_leaf_size, min_info_gain, bin_alpha, tail_sensitivity])

        '''--------------------- FINALISE TAB ---------------------'''
        tab1 = widgets.HBox([colA, colB])

        # ----------------------------------------------------------------------------------------------------------------

        '''------------------------------------------ TAB 2 ------------------------------------------'''

        '''--------------------- FUNCTIONS ---------------------'''

        def add_button_evolve_clicked(_):

            vals = [
                mutations.value,
                generations.value,
                max_generation_depth.value,
                max_severity.value,
                max_leaves.value
            ]

            value_string = ','.join([str(i) for i in vals])

            if pipeline.options[0] == "":
                new_option = f'{len(pipeline.options)}: Evolve({value_string})'
            else:
                new_option = f'{len(pipeline.options)+1}: Evolve({value_string})'
            pipeline.options = (*pipeline.options, new_option)
            pipeline.options = (o for o in pipeline.options if o != "")

        def add_button_tighten_clicked(_):
            vals = [
                iterations.value,
                learning_rate.value,
                early_stopping.value
            ]

            value_string = ','.join([str(i) for i in vals])

            if pipeline.options[0] == "":
                new_option = f'{len(pipeline.options)}: Tighten({value_string})'
            else:
                new_option = f'{len(pipeline.options)+1}: Tighten({value_string})'
            pipeline.options = (*pipeline.options, new_option)
            pipeline.options = (o for o in pipeline.options if o != "")

        def drop_button_clicked(b):
            val = pipeline.value
            if val and val[0] != '':    
                pipeline.options = (o for o in pipeline.options if o not in val)
                pipeline.options = (f'{i+1}:{o.split(":")[1]}' for i, o in enumerate(pipeline.options))


        '''--------------------- COLUMN 1 ---------------------'''
        title_1 = widgets.HTML(f"<h4>Evolve</h4>", layout=widgets.Layout(height='auto'))

        mutations = widgets.IntSlider(value=100, min=5, max=200, step=5, description='mutations:', style=style)
        generations = widgets.IntSlider(value=50, min=5, max=200, step=5, description='generations:', style=style)
        max_generation_depth = widgets.IntSlider(value=10, min=2, max=100, step=1, description='max_generation_depth:', style=style)
        max_severity = widgets.FloatSlider(value=0.25, min=0.01, max=0.5, step=0.01, description='max_severity:', style=style)
        max_leaves = widgets.IntSlider(value=20, min=1, max=100, step=5, description='max_leaves:', style=style)

        add_button_evolve = widgets.Button(description="Add Stage",icon='plus')
        add_button_evolve.style.button_color = '#12b980'
        add_button_evolve.on_click(add_button_evolve_clicked)

        col1 = widgets.VBox([title_1, mutations, generations, max_generation_depth, max_severity, max_leaves, add_button_evolve])


        '''--------------------- COLUMN 2 ---------------------'''
        title_2 = widgets.HTML(f"<h4>Tighten</h4>", layout=widgets.Layout(height='auto'))
        add_button_tighten = widgets.Button(description="Add Stage",icon='plus')
        add_button_tighten.style.button_color = '#12b980'
        add_button_tighten.on_click(add_button_tighten_clicked)

        iterations = widgets.IntSlider(value=100, min=5, max=200, step=5, description='iterations:', style=style)
        learning_rate = widgets.FloatSlider(value=0.05, min=0.005, max=0.2, step=0.005, description='learning_rate:', style=style, readout_format='.3f')
        early_stopping = widgets.IntSlider(value=100, min=5, max=200, step=5, description='early_stopping:', style=style)

        col2 = widgets.VBox([title_2, iterations, learning_rate, early_stopping, add_button_tighten])


        '''--------------------- FINALISE TAB ---------------------'''
        tab2 = widgets.HBox([col1, col2])


        '''------------------------------------------ BODY ------------------------------------------'''
        tabs = widgets.Tab([tab1, tab2])
        tabs.set_title(0, 'Parameters')
        tabs.set_title(1, 'Optimisation')

        '''------------------------------------------ PIPELINE ------------------------------------------'''

        title_3 = widgets.HTML(
            f"<h4>Optimisation Pipeline</h4><p>Select from optimisation tab</p>",
            layout=widgets.Layout(height='auto'))

        pipeline = widgets.SelectMultiple(
            index=(0,),
            options=[''],
            rows=10,
            disabled=False,
            layout=Layout(width ='300px'))

        drop_button = widgets.Button(description="Drop Stage(s)",icon='times')
        drop_button.style.button_color = '#e21c47'
        drop_button.on_click(drop_button_clicked)

        col3 = widgets.VBox([title_3, pipeline, drop_button])

        body = widgets.HBox([tabs, col3])


        '''------------------------------------------ FOOTER ------------------------------------------'''
        def close_button_clicked(_):
            clear_output()

        def train_button_clicked(_):
            
            params = {
                'model_name': self.model_name,
                'target': target.value,
                'id_columns': list(id_columns.value),
                'max_depth': max_depth.value,
                'min_leaf_size': min_leaf_size.value,
                'min_info_gain': min_info_gain.value,
                'bin_alpha': bin_alpha.value,
                'tail_sensitivity': tail_sensitivity.value,
                'layers': [i.split(":")[1].strip() for i in list(pipeline.options) if i != '']
            }

            with output:
                header.close()
                body.close()
                footer.close()
                loader = Loader("Training", "Training completed").start()

                response = self.__session.post(
                    f'{self.hostname}/models/regression/train',
                    params=params,
                    files=files
                    )

                if response.status_code == 200:
                    content = json.loads(response.content)
                    self.__profile = self._update_profile_inf(content['data']['profile'])
                    self.__feature_importances = content['data']['feature_importances']
                    self.__base_value = content['data']['base_value'] 
                    self.__categorical_columns = content['data']['categorical_columns']
                    self.__numeric_columns = content['data']['numeric_columns']
                    self.__params = content['data']['parameters']
                    self.__layers = content['data']['layers']

                    loader.stop()

                    return json.loads(response.content)

                else:
                    loader.end = response.content
                    loader.stop()

                    return response.content


        train_button = widgets.Button(description="Train",icon='bolt')
        train_button.style.button_color = '#0080ea'
        train_button.on_click(train_button_clicked)

        close_button = widgets.Button(description='Close', icon='window-close')
        close_button.on_click(close_button_clicked)

        footer = widgets.HBox([train_button, close_button])


        '''------------------------------------------ SCREEN ------------------------------------------'''

        screen = widgets.VBox([header, body, footer, output])

        display(screen)
             
    def xplain(self):
        metadata = {
                "base_value": self.__base_value,
                "profile": self.__profile,
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
    
    def predict(self, x):
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

    def evaluate(self, X, y):

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
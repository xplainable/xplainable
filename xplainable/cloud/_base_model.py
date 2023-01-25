from datetime import datetime
import dill
import os
import numpy as np
import json
from urllib3.exceptions import HTTPError
from sklearn.metrics import *
from ..visualisation.explain import *
import xplainable
import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
from ..utils.xwidgets import ClickResponsiveToggleButtons


class BaseModel:

    def __init__(self, model_name, model_description=None):

        self.model_name = model_name
        self.model_description = model_description
        
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

    def _partition_transform(self, x, partition):
        """ Transforms a dataset into the model weights.
        
        Args:
            x (pandas.DataFrame): The dataframe to be transformed.
            
        Returns:
            pandas.DataFrame: The transformed dataset.
        """

        if partition not in self.partitions.keys():
            raise ValueError(f'Partition {partition} does not exist')

        x = x.copy()

        n_cols = x.select_dtypes(include=np.number).columns.tolist()
        x[n_cols] = x[n_cols].astype('float64')

        profile = self.partitions[partition]['profile']
        categorical_columns = list(profile['categorical'].keys())
        numeric_columns = list(profile['numeric'].keys())

        # Get column names from training data
        columns = categorical_columns + numeric_columns

        # Remove partition value
        columns.remove(self.partition_on) if self.partition_on in columns else None

        # Filter x to only relevant columns
        x = x[[i for i in columns if i in list(x)]]

        # Map score for all categorical features
        for col in categorical_columns:
            mapp = profile["categorical"][col]
            x[col] = x[col].apply(self._map_categorical, args=(mapp,))

        # Map score for all numeric features
        for col in numeric_columns:
            mapp = profile["numeric"][col]
            x[col] = x[col].apply(self._map_numeric, args=(mapp,))

        return x

    def publish(self):

        # Get models
        response = xplainable.client.__session__.get(
            url=f'{self.hostname}/v1/models'
            )

        # Find model ID
        if response.status_code == 200:
            models = json.loads(response.content)
            if len(models) == 0:
                raise HTTPError(
                    f"400 Model with name {self.model_name} doesn't exist.")
                
            model_id = [i['model_id'] for i in models if \
                i['model_name'] == self.model_name]

            if len(model_id) == 1:
                model_id = model_id[0]
            else:
                raise HTTPError(
                    f"400 Model with name {self.model_name} doesn't exist.")

            # Publish model
            response = xplainable.client.__session__.post(
            url=f'{self.hostname}/v1/models/{model_id}/publish'
            )

            if response.status_code == 200:
                print(f"Published model {model_id} ({self.model_name})")
                return
            else:
                raise HTTPError(response)

        elif response.status_code == 401:
            raise HTTPError(f"401 Unauthorised")

        else:
            raise HTTPError(response)

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

    def explain(self):
        data = generate_partition_data(self)
        buttons = ClickResponsiveToggleButtons(description='')

        buttons.options = list(data.keys())
        buttons.value = buttons.options[0]

        def on_change(_):
            canvas.clear_output()
            with canvas:
                display(plot_partition(self, data, buttons.value))

        buttons.on_click(on_change)

        canvas = widgets.Output()
        with canvas:
            display(plot_partition(self, data, buttons.options[0]))
        
        out = widgets.VBox([buttons, canvas])
        out.layout = widgets.Layout(min_height="500px")

        return out


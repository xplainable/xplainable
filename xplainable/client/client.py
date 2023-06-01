""" Copyright Xplainable Pty Ltd, 2023"""

from ..utils.api import get_response_content
from ..utils.encoders import NpEncoder
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from ..gui.screens.preprocessor import Preprocessor
from ..preprocessing import transformers as xtf
from ..utils.exceptions import AuthenticationError
from ..core.models import (XClassifier, XRegressor, PartitionedRegressor,
                           PartitionedClassifier)
from ..gui.components.cards import render_user_avatar
import json
import numpy as np
import pyperclip
import time
from IPython.display import clear_output, display
from ..gui.components import KeyValueTable
import ipywidgets as widgets
from typing import Union


class Client:
    """ A client for interfacing with the xplainable web api (xplainable cloud).

    Access models, preprocessors and user data from xplainable cloud. API keys
    can be generated at https://app.xplainable.io.

    Args:
        api_key (str): A valid api key.
    """

    def __init__(self, api_key):
        self.__api_key = api_key
        self.hostname = 'https://api.xplainable.io'
        self.machines = {}
        self.__session__ = requests.Session()
        self._user = None
        self.avatar = None
        self._init()

    def _init(self):
        """ Authorize access to xplainable API.
        
            Active API Key is required for authorization. 

        Raises:
            HTTPError: If user not authorized.
        """
        # Add token to session headers
        self.__session__.headers['api_key'] = self.__api_key

        # Configure retry strategy
        RETRY_STRATEGY = Retry(
            total=5,
            backoff_factor=1
        )
        # Mount strategy
        ADAPTER = HTTPAdapter(max_retries=RETRY_STRATEGY)
        self.__session__.mount(self.hostname, ADAPTER)

        self._user = self.get_user_data()
        self.avatar = render_user_avatar(self._user)

        self.xplainable_version = None
        self.python_version = None

    def list_models(self) -> list:
        """ Lists all models of the active user's team.

        Returns:
            dict: Dictionary of saved models.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/models'
            )

        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data

    def list_model_versions(self, model_id: int) -> list:
        """ Lists all versions of a model.

        Args:
            model_id (int): The model id

        Returns:
            dict: Dictionary of model versions.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/models/{model_id}/versions'
            )

        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data
    
    def list_preprocessors(self) -> list:
        """ Lists all preprocessors of the active user's team.

        Returns:
            dict: Dictionary of preprocessors.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/preprocessors'
            )

        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data

    def list_preprocessor_versions(self, preprocessor_id: int) -> list:
        """ Lists all versions of a preprocessor.

        Args:
            preprocessor_id (int): The preprocessor id

        Returns:
            dict: Dictionary of preprocessor versions.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/preprocessors/{preprocessor_id}/versions'
            )
        
        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data


    def load_preprocessor(
            self, preprocessor_id: int, version_id: int,
            response_only: bool = False):
        """ Loads a preprocessor by preprocessor_id and version_id.

        Args:
            preprocessor_id (int): The preprocessor id
            version_id (int): The version id
            response_only (bool, optional): Returns the preprocessor metadata.

        Returns:
            xplainable.preprocessing.Preprocessor: The loaded preprocessor
        """

        def build_transformer(stage):
            """Build transformer from metadata"""

            if not hasattr(xtf, stage["name"]):
                raise ValueError(f"{stage['name']} does not exist in the transformers module")

            # Get transformer function
            func = getattr(xtf, stage["name"])

            return func(**stage['params'])
        
        try:
            preprocessor_response = self.__session__.get(
                url=f'{self.hostname}/v1/preprocessors/{preprocessor_id}/versions/{version_id}'
                )

            response = get_response_content(preprocessor_response)

            if response_only:
                return response

            stages = response['stages']
            deltas = response['deltas']
            
        except Exception as e:
            raise ValueError(
            f'Preprocessor with ID {preprocessor_id}:{version_id} does not exist')
            
        xp = Preprocessor()
        xp.pipeline.stages = [{"feature": i["feature"], "name": i["name"], \
            "transformer": build_transformer(i)} for i in stages]
        xp.df_delta = deltas
        xp.state = len(xp.pipeline.stages)

        return xp
    
    def load_classifier(self, model_id: int, version_id: int, model=None):
        """ Loads a binary classification model by model_id

        Args:
            model_id (str): A valid model_id
            version_id (str): A valid version_id
            model (PartitionedClassifier): An existing model to add partitions

        Returns:
            xplainable.PartitionedClassifier: The loaded xplainable classifier
        """

        response = self.__get_model__(model_id, version_id)

        if response['model_type'] != 'binary_classification':
            raise ValueError(f'Model with ID {model_id}:{version_id} is not a binary classification model')

        if model is None:
            partitioned_model = PartitionedClassifier(response['partition_on'])
        else:
            partitioned_model = model

        for p in response['partitions']:
            model = XClassifier()
            model._profile = np.array([
                np.array(i) for i in json.loads(p['profile'])])
            model._calibration_map = p['calibration_map']
            model._support_map = p['support_map']
            model.base_value = p['base_value']
            model.target_map = p['target_map']
            model.feature_map = p['feature_map']
            model.feature_map_inv = {idx: {v: i} for idx, val in p['feature_map'].items() for i, v in val.items()}
            model.columns = p['columns']
            model.id_columns = p['id_columns']
            model.categorical_columns = p['feature_map'].keys()
            model.numeric_columns = [c for c in model.columns if c not \
                                     in model.categorical_columns]
            model.category_meta = p['category_meta']

            partitioned_model.add_partition(model, p['partition'])

        return partitioned_model

    def load_regressor(self, model_id: int, version_id: int, model=None):
        """ Loads a regression model by model_id and version_id

        Args:
            model_id (str): A valid model_id
            version_id (str): A valid version_id
            model (PartitionedRegressor): An existing model to add partitions to

        Returns:
            xplainable.PartitionedRegressor: The loaded xplainable regressor
        """
        response = self.__get_model__(model_id, version_id)

        if response['model_type'] != 'regression':
            raise ValueError(f'Model with ID {model_id}:{version_id} is not a regression model')

        if model is None:
            partitioned_model = PartitionedRegressor(response['partition_on'])
        else:
            partitioned_model = model

        for p in response['partitions']:
            model = XRegressor()
            model._profile = np.array([
                np.array(i) for i in json.loads(p['profile'])])
            model.base_value = p['base_value']
            model.target_map = p['target_map']
            model.feature_map = p['feature_map']
            model.feature_map_inv = {idx: {v: i} for idx, val in p['feature_map'].items() for i, v in val.items()}
            model.columns = p['columns']
            model.id_columns = p['id_columns']
            model.categorical_columns = p['feature_map'].keys()
            model.numeric_columns = [c for c in model.columns if c \
                                     not in model.categorical_columns]
            model.category_meta = p['category_meta']

            partitioned_model.add_partition(model, p['partition'])

        return partitioned_model

    def __get_model__(self, model_id: int, version_id: int):
        try:
            response = self.__session__.get(
                url=f'{self.hostname}/v1/models/{model_id}/versions/{version_id}'
            )
            return get_response_content(response)

        except Exception as e:
            raise ValueError(
            f'Model with ID {model_id}:{version_id} does not exist')


    def get_user_data(self) -> dict:
        """ Retrieves the user data for the active user.

        Returns:
            dict: User data
        """
        
        response = self.__session__.get(
        url=f'{self.hostname}/v1/user'
        )

        if response.status_code ==200:
            return get_response_content(response)
        else:
            raise AuthenticationError("API key has expired or is invalid.")
        
    def create_preprocessor_id(
            self, preprocessor_name: str, preprocessor_description: str) -> int:
        """ Creates a new preprocessor and returns the preprocessor id.

        Args:
            preprocessor_name (str): The name of the preprocessor
            preprocessor_description (str): The description of the preprocessor

        Returns:
            int: The preprocessor id
        """

        payoad = {
            "preprocessor_name": preprocessor_name,
            "preprocessor_description": preprocessor_description
        }
        
        response = self.__session__.post(
            url=f'{self.hostname}/v1/create-preprocessor',
            json=payoad
        )
        
        preprocessor_id = get_response_content(response)
            
        return preprocessor_id
    
    def create_preprocessor_version(
            self, preprocessor_id: int, stages: dict, deltas: dict,
            versions: dict) -> int:
        """ Creates a new preprocessor version and returns the version id.

        Args:
            preprocessor_id (int): The preprocessor id
            stages (dict): The preprocessor stages
            deltas (dict): The preprocessor deltas
            versions (dict): Versions of current environment

        Returns:
            int: The preprocessor version id
        """

        payload = {
            "stages": stages,
            "deltas": deltas,
            "versions": versions
            }

        # Create a new version and fetch id
        response = self.__session__.post(
                url=f'{self.hostname}/v1/preprocessors/{preprocessor_id}/add-version',
            json=payload
        )

        version_id = get_response_content(response)

        return version_id

    def create_model_id(
            self, model_name: str, model_description: str, target: str,
            model_type: str) -> int:
        """ Creates a new model and returns the model id.

        Args:
            model_name (str): The name of the model
            model_description (str): The description of the model
            target (str): The target column name
            model_type (str): The model type

        Returns:
            int: The model id
        """

        payoad = {
            "model_name": model_name,
            "model_description": model_description,
            "model_type": model_type,
            "target_name": target
        }
        
        response = self.__session__.post(
            url=f'{self.hostname}/v1/create-model',
            json=payoad
        )
        
        model_id = get_response_content(response)
            
        return model_id

    def create_model_version(
        self, model_id: int, partition_on: str, ruleset: Union[dict, str],
        health_info: dict, versions: dict) -> int:
        """ Creates a new model version and returns the version id.

        Args:
            model_id (int): The model id
            partition_on (str): The partition column name
            ruleset (dict | str): The feeature ruleset
            health_info (dict): Feature health information
            versions (dict): Versions of current environment

        Returns:
            int: The model version id
        """

        payload = {
            "partition_on": partition_on,
            "ruleset": json.dumps(ruleset, cls=NpEncoder),
            "health_info": json.dumps(health_info, cls=NpEncoder),
            "versions": versions
            }

        # Create a new version and fetch id
        response = self.__session__.post(
                url=f'{self.hostname}/v1/models/{model_id}/add-version',
            json=payload
        )

        version_id = get_response_content(response)

        return version_id

    def log_partition(
            self, model_type: str, partition_name: str, model, model_id: int,
            version_id: int, evaluation: dict = None,
            training_metadata: dict = None) -> None:
        """ Logs a partition to a model version.

        Args:
            model_type (str): The model type
            partition_name (str): The name of the partition column
            model (mixed): The model to log
            model_id (int): The model id
            version_id (int): The version id
            evaluation (dict, optional): Model evaluation data and metrics.
            training_metadata (dict, optional): Model training metadata.

        """

        data = {
            "partition": str(partition_name),
            "profile": json.dumps(model._profile, cls=NpEncoder),
            "feature_importances": json.loads(
                json.dumps(model.feature_importances, cls=NpEncoder)),
            "id_columns": json.loads(
                json.dumps(model.id_columns, cls=NpEncoder)),
            "columns": json.loads(
                json.dumps(model.columns, cls=NpEncoder)),
            "target_map": json.loads(
                json.dumps(model.target_map_inv, cls=NpEncoder)),
            "parameters": json.loads(
                json.dumps(model.params, cls=NpEncoder)),
            "base_value": json.loads(
                json.dumps(model.base_value, cls=NpEncoder)),
            "feature_map": json.loads(
                json.dumps(model.feature_map, cls=NpEncoder)),
            "category_meta": json.loads(
                json.dumps(model.category_meta, cls=NpEncoder)),
            "calibration_map": None,
            "support_map": None
            }

        if model_type == 'binary_classification':
            data.update({
                "calibration_map": json.loads(
                    json.dumps(model._calibration_map, cls=NpEncoder)),
                "support_map": json.loads(
                json.dumps(model._support_map, cls=NpEncoder))
            })

        if evaluation is not None:
            data.update({
                "evaluation": json.loads(json.dumps(evaluation, cls=NpEncoder))
                })

        if training_metadata is not None:
            data.update({
                "training_metadata": json.loads(
                    json.dumps(training_metadata, cls=NpEncoder))
                })

        try:
            response = self.__session__.post(
                url=f'{self.hostname}/v1/models/{model_id}/versions/{version_id}/add-partition',
                json=data
            )
        except Exception as e:
            raise ValueError(e)
        
        if response.status_code != 200:
            raise ConnectionError("Failed to add partition data")

        return

    def deploy(
            self, hostname: str, model_id: int, version_id: int, partition_id: int,
            raw_output: bool=False) -> dict:
        """ Deploys a model partition to xplainable cloud.

        The hostname should be the url of the inference server. For example:
        https://inference.xplainable.io

        Args:
            hostname (str): The host name for the inference server
            model_id (int): The model id
            version_id (int): The version id
            partition_id (int): The partition id
            raw_output (bool, optional): returns a dictionary

        Returns:
            dict: deployment status and details.
        """
        
        url = f'{self.hostname}/v1/models/{model_id}/versions/{version_id}/partitions/{partition_id}/deploy'
        response = self.__session__.get(url)
        
        if response.status_code == 200:

            deployment_id = response.json()

            data = {
                "deployment_id": deployment_id,
                "status": "active",
                "location": "sydney",
                "endpoint": f"{hostname}/v1/predict"
            }

            if raw_output:
                return data

            table = KeyValueTable(
                data,
                transpose=False,
                padding="0px 20px 0px 5px",
                table_width='auto',
                header_color='#e8e8e8',
                border_color='#dddddd',
                header_font_color='#20252d',
                cell_font_color= '#374151'
                )

            def on_click(b):
                try:
                    self.generate_deploy_key(
                         description='generated by python client',
                         deployment_id=deployment_id,
                         surpress_output=True
                     )
                    b.description = "Copied to clipboard!"
                    b.disabled = True
                    
                except Exception as e:
                    b.description = "Failed. Try Again."
                    b.disabled = True
                    time.sleep(2)
                    b.description = "Generate Deploy Key"
                    b.disabled = False
                

            button = widgets.Button(description="Generate Deploy Key")
            button.on_click(on_click)

            output = widgets.HBox([table.html_widget, button])
            display(output)

        else:
            return {"message": f"Failed with status code {response.status_code}"}
        
    def generate_deploy_key(
            self, description: str, deployment_id: int, 
            days_until_expiry: float = 90, surpress_output: bool = False
            ) -> None:
        """ Generates a deploy key for a model deployment.

        Args:
            description (str): Description of the deploy key use case.
            deployment_id (int): The deployment id.
            days_until_expiry (float): The number of days until the key expires.
            surpress_output (bool): Surpress output. Defaults to False.

        Returns:
            None: No key is returned. The key is copied to the clipboard.
        """

        url = f'{self.hostname}/v1/create-deploy-key'
        
        params = {
            'description': description,
            'deployment_id': deployment_id,
            'days_until_expiry': days_until_expiry
        }
        
        response = self.__session__.get(
            url=url,
            params=params
            )

        deploy_key = response.json()

        if deploy_key:
            pyperclip.copy(deploy_key)
            if not surpress_output:
                print("Deploy key copied to clipboard!")
                time.sleep(2)
                clear_output()
        else:
            return response.status_code

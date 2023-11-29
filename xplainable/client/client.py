""" Copyright Xplainable Pty Ltd, 2023"""
import json
import numpy as np
import pandas as pd
import pyperclip
import time
import inspect
import ast
from ..utils.api import get_response_content
from ..utils.encoders import NpEncoder, force_json_compliant
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from ..preprocessing.pipeline import XPipeline
from ..preprocessing import transformers as xtf
from ..utils.exceptions import AuthenticationError
from ..utils.dualdict import FeatureMap, TargetMap
from ..utils.helpers import get_df_delta
from ..utils.encoders import profile_parse
from ..quality.scanner import XScan
from ..metrics.metrics import evaluate_classification, evaluate_regression
from ..core.models import (XClassifier, XRegressor, PartitionedRegressor, PartitionedClassifier, ConstructorParams)

from ..config import OUTPUT_TYPE


class Client:
    """ A client for interfacing with the xplainable web api (xplainable cloud).

    Access models, preprocessors and user data from xplainable cloud. API keys
    can be generated at https://beta.xplainable.io.

    Args:
        api_key (str): A valid api key.
    """

    def __init__(self, api_key, hostname='https://api.xplainable.io'):
        self.__api_key = api_key
        self.hostname = hostname
        self.machines = {}
        self.__session = requests.Session()
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
        self.__session.headers['api_key'] = self.__api_key

        # Configure retry strategy
        RETRY_STRATEGY = Retry(
            total=5,
            backoff_factor=1
        )
        # Mount strategy
        ADAPTER = HTTPAdapter(max_retries=RETRY_STRATEGY)
        self.__session.mount(self.hostname, ADAPTER)

        session_data = self.get_user_data()
        
        self.__org_id = session_data.pop('organisation_id')
        self.__team_id = session_data.pop('team_id')
        self.__ext = f'organisations/{self.__org_id}/teams/{self.__team_id}'
        self._user = session_data
        
        try:
            import ipywidgets
            from ..gui.components.cards import render_user_avatar
            self.avatar = render_user_avatar(self._user)
        except ImportError:
            pass

        self.xplainable_version = None
        self.python_version = None

    def list_models(self) -> list:
        """ Lists all models of the active user's team.

        Returns:
            dict: Dictionary of saved models.
        """

        response = self.__session.get(
            url=f'{self.hostname}/v1/{self.__ext}/models'
            )

        data = get_response_content(response)
        
        # For better readability
        [i.pop('user') for i in data]
        [i.pop('contributors') for i in data]

        return data

    def list_model_versions(self, model_id: int) -> list:
        """ Lists all versions of a model.

        Args:
            model_id (int): The model id

        Returns:
            dict: Dictionary of model versions.
        """

        response = self.__session.get(
            url=f'{self.hostname}/v1/{self.__ext}/models/{model_id}/versions'
            )

        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data
    
    def list_preprocessors(self) -> list:
        """ Lists all preprocessors of the active user's team.

        Returns:
            dict: Dictionary of preprocessors.
        """

        response = self.__session.get(
            url=f'{self.hostname}/v1/{self.__ext}/preprocessors'
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

        response = self.__session.get(
            url=f'{self.hostname}/v1/{self.__ext}/preprocessors/{preprocessor_id}/versions'
            )
        
        data = get_response_content(response)
        [i.pop('user') for i in data]

        return data


    def load_preprocessor(
            self, preprocessor_id: int, version_id: int,
            gui_object: bool = False, response_only: bool = False):
        """ Loads a preprocessor by preprocessor_id and version_id.

        Args:
            preprocessor_id (int): The preprocessor id
            version_id (int): The version id
            response_only (bool, optional): Returns the preprocessor metadata.

        Returns:
            xplainable.preprocessing.pipeline.Pipeline: The loaded pipeline
        """

        def build_transformer(stage):
            """Build transformer from metadata"""

            if not hasattr(xtf, stage["name"]):
                raise ValueError(f"{stage['name']} does not exist in the transformers module")

            # Get transformer function
            func = getattr(xtf, stage["name"])

            return func(**stage['params'])
        
        try:
            preprocessor_response = self.__session.get(
                url=f'{self.hostname}/v1/{self.__ext}/preprocessors/{preprocessor_id}/versions/{version_id}'
                )

            response = get_response_content(preprocessor_response)

            if response_only:
                return response

        except Exception as e:
            raise ValueError(
            f'Preprocessor with ID {preprocessor_id}:{version_id} does not exist')
        
        stages = response['stages']
        deltas = response['deltas']

        pipeline = XPipeline()
        pipeline.stages = [{"feature": i["feature"], "name": i["name"], \
                "transformer": build_transformer(i)} for i in stages]
        
        if not gui_object:
            return pipeline

        else:
            from ..gui.screens.preprocessor import Preprocessor
            pp = Preprocessor()
            pp.pipeline = pipeline
            pp.df_delta = deltas
            pp.state = len(pipeline.stages)

            return pp
    
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
                np.array(i) for i in json.loads(p['profile'])], dtype=object)
            
            model._profile = profile_parse(model._profile)
            
            model._calibration_map = {
                int(i): v for i, v in p['calibration_map'].items()}
            
            model._support_map = {
                int(i): v for i, v in p['support_map'].items()}
            
            model.base_value = p['base_value']
            model.target_map = TargetMap({int(i): v for i, v in p['target_map'].items()}, True)
            model.feature_map = {k: FeatureMap(v) for k, v in p['feature_map'].items()}
            
            model.columns = p['columns']
            model.id_columns = p['id_columns']

            model.categorical_columns = p['feature_map'].keys()
            model.numeric_columns = [c for c in model.columns if c not in model.categorical_columns]

            if 'constructs' in p:
                model.constructs_from_json(p['constructs'])

            model.category_meta = {
                i: {ii: {int(float(k)): v for k, v in vv.items()} for ii, vv \
                    in v.items()} for i, v in p['category_meta'].items()}

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
            model._profile = np.array([np.array(i) for i in json.loads(p['profile'])])
            model._profile = profile_parse(model._profile)
            model.base_value = p['base_value']
            model.feature_map = {k: FeatureMap(v) for k, v in p['feature_map'].items()}
            #model.parameters = ConstructorParams(p['parameters'])

            model.columns = p['columns']
            model.id_columns = p['id_columns']

            model.categorical_columns = p['feature_map'].keys()
            model.numeric_columns = [c for c in model.columns if c not in model.categorical_columns]

            if 'constructs' in p:
                model.constructs_from_json(p['constructs'])

            model.category_meta = {
                i: {ii: {int(float(k)): v for k, v in vv.items()} for ii, vv \
                    in v.items()} for i, v in p['category_meta'].items()}

            partitioned_model.add_partition(model, p['partition'])

        return partitioned_model

    def __get_model__(self, model_id: int, version_id: int):
        try:
            response = self.__session.get(
                url=f'{self.hostname}/v1/{self.__ext}/models/{model_id}/versions/{version_id}'
            )
            return get_response_content(response)

        except Exception as e:
            raise ValueError(f'Model with ID {model_id}:{version_id} does not exist')

    def get_user_data(self) -> dict:
        """ Retrieves the user data for the active user.

        Returns:
            dict: User data
        """
        
        response = self.__session.get(
            url=f'{self.hostname}/v1/client-connect'
        )

        if response.status_code == 200:
            return get_response_content(response)
        
        else:
            raise AuthenticationError(
                f"{response.status_code} Unauthenticated. "
                f"{response.json()['detail']}"
            )

    def create_preprocessor_id(
            self, preprocessor_name: str, preprocessor_description: str) -> str:
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

        response = self.__session.post(
            url=f'{self.hostname}/v1/{self.__ext}/create-preprocessor',
            json=payoad
        )
        
        preprocessor_id = get_response_content(response)
            
        return preprocessor_id
    
    def create_preprocessor_version(
            self, preprocessor_id: str, pipeline: list, df: pd.DataFrame = None
            ) -> str:
        """ Creates a new preprocessor version and returns the version id.

        Args:
            preprocessor_id (int): The preprocessor id
            pipeline (xplainable.preprocessing.pipeline.Pipeline): pipeline

        Returns:
            int: The preprocessor version id
        """

        # Structure the stages and deltas
        stages = []
        deltas = []
        if df is not None:
            before = df.copy()
            deltas.append({"start": json.loads(before.head(10).to_json(
                orient='records'))})
            delta_gen = pipeline.transform_generator(before)

        for stage in pipeline.stages:
            step = {
                'feature': stage['feature'],
                'name': stage['name'],
                'params': stage['transformer'].__dict__
            }

            stages.append(step)

            if df is not None:
                after = delta_gen.__next__()
                delta = get_df_delta(before.copy(), after.copy())
                deltas.append(delta)
                before = after.copy()

        # Get current versions
        versions = {
                "xplainable_version": self.xplainable_version,
                "python_version": self.python_version
            }

        # Create payload
        payload = {
            "stages": stages,
            "deltas": deltas,
            "versions": versions
            }
        
        # Create a new version and fetch id
        url = (
            f'{self.hostname}/v1/{self.__ext}/preprocessors/'
            f'{preprocessor_id}/add-version'
            )
        
        response = self.__session.post(url=url, json=payload)

        version_id = get_response_content(response)

        return version_id
        
    def _detect_model_type(self, model):

        if 'Partitioned' in model.__class__.__name__:
            model = model.partitions['__dataset__']

        cls_name = model.__class__.__name__

        if cls_name == "XClassifier":
            model_type = "binary_classification"

        elif cls_name == "XRegressor":
            model_type = "regression"

        else:
            raise ValueError(
                f'Model type {cls_name} is not supported')
        
        return model_type, model.target

    def create_model_id(
            self, model, model_name: str, model_description: str) -> str:
        """ Creates a new model and returns the model id.

        Args:
            model_name (str): The name of the model
            model_description (str): The description of the model
            model (XClassifier | XRegressor): The model to create.

        Returns:
            int: The model id
        """

        model_type, target = self._detect_model_type(model)

        payoad = {
            "model_name": model_name,
            "model_description": model_description,
            "model_type": model_type,
            "target_name": target,
            "algorithm": model.__class__.__name__
        }
        
        response = self.__session.post(
            url=f'{self.hostname}/v1/{self.__ext}/create-model',
            json=payoad
        )
        
        model_id = get_response_content(response)
            
        return model_id

    def create_model_version(
            self, model, model_id: str, x: pd.DataFrame, y: pd.Series) -> str:
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

        # ruleset = generate_ruleset(
            #     self.df,
            #     self.model.partitions['__dataset__'].target,
            #     self.model.partitions['__dataset__'].id_columns
            #     )

        # Get current versions
        versions = {
                "xplainable_version": self.xplainable_version,
                "python_version": self.python_version
            }

        partition_on = model.partition_on if 'Partitioned' in \
            model.__class__.__name__ else None

        payload = {
            "partition_on": partition_on,
            "versions": versions,
            "partitions": []
            }

        partitioned_models = ['PartitionedClassifier', 'PartitionedRegressor']
        independent_models = ['XClassifier', 'XRegressor']

        # get all partitions
        if model.__class__.__name__ in partitioned_models:
            for p, m in model.partitions.items():
                if p == '__dataset__':
                    part_x = x
                    part_y = y

                else:
                    part_x = x[x[partition_on].astype(str) == str(p)]
                    part_y = y[y.index.isin(part_x.index)]

                pdata = self._get_partition_data(m, p, part_x, part_y)
                payload['partitions'].append(pdata)
        
        elif model.__class__.__name__ in independent_models:
            pdata = self._get_partition_data(model, '__dataset__', x, y)
            payload['partitions'].append(pdata)
        
        # Create a new version and fetch id
        url = f'{self.hostname}/v1/{self.__ext}/models/{model_id}/add-version'
        response = self.__session.post(
            url=url, json=force_json_compliant(payload))

        version_id = get_response_content(response)

        return version_id

    def _get_partition_data(
            self, model, partition_name: str, x: pd.DataFrame,
            y: pd.Series) -> dict:
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

        model_type, _ = self._detect_model_type(model)

        data = {
            "partition": str(partition_name),
            "profile": json.dumps(model._profile, cls=NpEncoder),
            "feature_importances": json.loads(
                json.dumps(model.feature_importances, cls=NpEncoder)),
            "id_columns": json.loads(
                json.dumps(model.id_columns, cls=NpEncoder)),
            "columns": json.loads(
                json.dumps(model.columns, cls=NpEncoder)),
            "parameters": model.params.to_json(),
            "base_value": json.loads(
                json.dumps(model.base_value, cls=NpEncoder)),
            "feature_map": json.loads(
                json.dumps({k: fm.forward for k, fm in model.feature_map.items()}, cls=NpEncoder)),
            "target_map": json.loads(
                json.dumps(model.target_map.reverse, cls=NpEncoder)),
            "category_meta": json.loads(
                json.dumps(model.category_meta, cls=NpEncoder)),
            # "constructs": model.constructs_to_json(),
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

            evaluation = model.metadata.get('evaluation', {})
            if evaluation == {}:
                y_prob = model.predict_score(x)

                if model.target_map:
                    y = y.map(model.target_map)

                evaluation = {
                    'train': evaluate_classification(y, y_prob)
                }
                
        elif model_type == 'regression':
            evaluation = model.metadata.get('evaluation', {})
            if evaluation == {}:
                y_pred = model.predict(x)
                evaluation = {
                            'train': evaluate_regression(y, y_pred)
                        }
        
        data["evaluation"] = json.dumps(evaluation, cls=NpEncoder)

        training_metadata = {
            i: v for i, v in model.metadata.items() if i != "evaluation"}
        
        data["training_metadata"] = json.dumps(training_metadata, cls=NpEncoder)
        
        if x is not None:
            scanner = XScan()
            scanner.scan(x)

            results = []
            for i, v in scanner.profile.items():
                feature_info = {
                    "feature": i,
                    "description": '',
                    "type": v['type'],
                    "health_info": json.loads(json.dumps(v, cls=NpEncoder))
                }
                results.append(feature_info)

            data["health_info"] = json.dumps(results, cls=NpEncoder)

        return data
    
    def list_deployments(self):
        """ Lists all deployments of the active user's team.

        Returns:
            dict: Dictionary of deployments.
        """

        response = self.__session.get(
            url=f'{self.hostname}/v1/{self.__ext}/deployments'
            )

        deployments = get_response_content(response)

        return deployments

    def deploy(
            self, model_id: str, version_id: str,
            hostname: str='https://inference.xplainable.io',
            location: str='syd', raw_output: bool=True) -> dict:
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
        
        url = (
            f'{self.hostname}/v1/{self.__ext}/models/{model_id}/versions/'
            f'{version_id}/deploy'
        )

        body = {
            "location": location
        }

        response = self.__session.put(url, json=body)
        
        if response.status_code == 200:

            deployment_id = response.json()['deployment_id']

            data = {
                "deployment_id": deployment_id,
                "status": "inactive",
                "location": location,
                "endpoint": f"{hostname}/v1/predict"
            }

            return data

        else:
            return {
                "message": f"Failed with status code {response.status_code}"}

    def activate_deployment(self, deployment_id):
        """ Activates a model deployment.

        Args:
            deployment_id (str): The deployment id
        """

        url = (
            f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/activate'
        )

        response = self.__session.patch(url)

        if response.status_code == 200:
            return response.json()

        else:
            return {
                "message": f"Failed with status code {response.status_code}"}
    
    def deactivate_deployment(self, deployment_id):
        """ Deactivates a model deployment.

        Args:
            deployment_id (str): The deployment id
        """

        url = (
            f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/deactivate'
        )

        response = self.__session.patch(url)

        if response.status_code == 200:
            return response.json()

        else:
            return {
                "message": f"Failed with status code {response.status_code}"}
        
    def generate_deploy_key(
            self, description: str, deployment_id: str, 
            days_until_expiry: float = 90, clipboard: bool = True,
            surpress_output: bool = False
            ) -> None:
        """ Generates a deploy key for a model deployment.

        Args:
            description (str): Description of the deploy key use case.
            deployment_id (str): The deployment id.
            days_until_expiry (float): The number of days until the key expires.
            surpress_output (bool): Surpress output. Defaults to False.

        Returns:
            None: No key is returned. The key is copied to the clipboard.
        """

        url = f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/create-deploy-key'
        
        params = {
            'description': description,
            'days_until_expiry': days_until_expiry
        }
        
        response = self.__session.put(
            url=url,
            json=params
            )

        deploy_key = response.json()

        if deploy_key:
            if not clipboard:
                return deploy_key
            pyperclip.copy(deploy_key['deploy_key'])
            if not surpress_output:
                print("Deploy key copied to clipboard!")
        else:
            raise ConnectionError(
                f"Falied to generate deploy key. Code: {response.status_code}")
        
    def generate_example_deployment_payload(self, deployment_id):
        """ Generates an example deployment payload for a deployment.

        Args:
            deployment_id (str): The deployment id.
        """

        url = f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/payload'

        response = self.__session.get(url)

        return response.json()
    
    @staticmethod
    def __parse_function(func):
        """ Parses a function to a middleware function. """
        if not callable(func):
            raise Exception("Function must be callable")

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise Exception("Function must take one parameter")

        # Parse the source code to an AST
        source = inspect.getsource(func)
        parsed_ast = ast.parse(source)

        # Rename the function in the AST
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                node.name = "middleware"
                break

        # Store the modified source
        modified_source = ast.unparse(parsed_ast)

        # Compile the AST back to code and execute in a new namespace
        local_vars = {}
        exec(compile(
            parsed_ast, filename="<ast>", mode="exec"),
            func.__globals__, local_vars)

        middleware = local_vars['middleware']
        middleware.source = modified_source
        return middleware
        
    def add_deployment_middleware(
        self, deployment_id, func, name, description=None):
        """ Adds or replaces a middleware function to a deployment.

        Args:
            deployment_id (str): The deployment id
            func (function): The middleware function
        """

        url = (
            f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/add-middleware'
        )

        # Convert function to expected name
        if func.__name__ != 'middleware':
            func = self.__parse_function(func)
            source_code = func.source
        
        else:
            source_code = inspect.getsource(func)

        body = {
            "code_block": source_code,
            "name": name,
            "description": description
        }

        response = self.__session.put(
            url=url,
            json=body
            )

        return response.json()
    
    def delete_deployment_middleware(self, deployment_id):
        """ Deletes a middleware function from a deployment.

        Args:
            deployment_id (str): The deployment id
        """

        url = (
            f'{self.hostname}/v1/{self.__ext}/deployments/{deployment_id}/middleware'
        )

        response = self.__session.delete(url)

        return {"status_code": response.status_code}

    def _gpt_report(
            self, model_id, version_id, target_description='',
            project_objective='', max_features=15, temperature=0.5):

        url = (
            f'{self.hostname}/v1/{self.__ext}/models/{model_id}/versions/'
            f'{version_id}/generate-report'
        )

        params = {
            'target_description': target_description,
            'project_objective': project_objective,
            'max_features': max_features,
            'temperature': temperature
        }

        response = self.__session.put(
            url=url,
            json=params,
            )
        
        content = get_response_content(response)

        return content

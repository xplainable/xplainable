from ..utils.api import get_response_content
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from ..gui.screens.preprocessor import Preprocessor
from ..preprocessing import transformers as tf

class Client:
    """ Client for interfacing with the xplainable web api.
    """

    def __init__(self, api_key, use_ray=False):
        self.__api_key = api_key
        self.hostname = 'https://api.xplainable.io'
        self.machines = {}
        self.__session__ = requests.Session()
        self.init()

    def init(self):
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

        url = f'{self.hostname}/v1/ping'
        response = self.__session__.get(url)

        if response.status_code == 200:
            return

            # url = f'{self.hostname}/v1/machines'
            # try:
            #     response = self.__session__.get(url)
            #     self.machines = {
            #         i['machine_name']: i['machine_id'] for \
            #             i in get_response_content(response)}
            
            # except Exception as e:
            #     raise ConnectionError('User has no available machines')
            
            # return
        
        else:
            raise PermissionError(
                "Not authenticated. API key invalid or expired")

    def list_models(self):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/models'
            )

        return get_response_content(response)

    def list_versions(self, model_id):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = self.__session__.get(
            url=f'{self.hostname}/v1/models/{model_id}/versions'
            )

        return get_response_content(response)

    def load_preprocessor(self, preprocessor_id, version_id='latest'):

        def build_transformer(stage):
            """Build transformer from metadata"""

            params = str(stage['params'])
            trans = eval(f'tf.{stage["name"]}(**{params})')

            return trans
        
        try:
            meta_response = self.__session__.get(
                    f'{self.hostname}/v1/preprocessors/{preprocessor_id}')

            preprocessor_meta = get_response_content(meta_response)

            versions_response = self.__session__.get(
                f'{self.hostname}/v1/preprocessors/{preprocessor_id}/versions')

            versions = get_response_content(versions_response)
            
            if version_id == 'latest':
                version_id = versions[-1][0]

            preprocessor_response = self.__session__.get(
                url=f'{self.hostname}/v1/preprocessors/{preprocessor_id}/versions/{version_id}'
                )

            stages = get_response_content(preprocessor_response)['stages']
            
        except Exception as e:
            raise ValueError(
            f'Preprocessor with ID {preprocessor_id}:{version_id} does not exist')
            
        xp = Preprocessor()
        xp.preprocessor_name = preprocessor_meta['preprocessor_name']
        xp.description = preprocessor_meta['preprocessor_description']
        xp.pipeline.stages = [{"feature": i["feature"], "name": i["name"], \
            "transformer": build_transformer(i)} for i in stages]

        xp.state = len(xp.pipeline.stages)

        return xp
    
    def load_model(self, model_id, version_id='latest'):
        """ Loads a model by model_id

        Args:
            model_id (str): A valid model_id

        Returns:
            xplainable.model: The loaded xplainable model
        """

        try:
            meta_response = self.__session__.get(
                    f'{self.hostname}/v1/models/{model_id}')

            model_meta = get_response_content(meta_response)

            
            versions_response = self.__session__.get(
                f'{self.hostname}/v1/models/{model_id}/versions')

            versions = get_response_content(versions_response)
            
            if version_id == 'latest':
                version_id = versions[-1]['version_id']

            partition_on = [v['partition_on'] for v in versions if \
                v['version_id'] == version_id][0]

            model_response = self.__session__.get(
                url=f'{self.hostname}/v1/models/{model_id}/versions/{version_id}'
                )

            model_data = {
                i['partition']: i for i in get_response_content(model_response)}
            
            model_data['__dataset__']['data']['partition_on'] = partition_on

        except Exception as e:
            raise ValueError(
            f'Model with ID {model_id}:{version_id} does not exist')
            
        
        if model_meta['model_type'] == 'binary_classification':
            from xplainable.models.classification import XClassifier
            model = XClassifier(model_name=model_meta['model_name'])

        elif model_meta['model_type'] == 'regression':
            from xplainable.models.regression import XRegressor
            model = XRegressor(model_name=model_meta['model_name'])
        
        model._load_metadata(model_data)

        return model

    def get_user_data(self):
        """ Retrieves the user data for the active user.

        Returns:
            dict: User data
        """
        
        response = self.__session__.get(
        url=f'{self.hostname}/v1/user'
        )

        return get_response_content(response)

    def save_model(self, model):
        pass
        
import json
import requests
from urllib3.exceptions import HTTPError
from authlib.jose import jwt
from xplainable.utils.api import get_response_content
import xplainable

__session__ = requests.Session()


class Client:
    """ Client for interfacing with the xplainable web api.
    """

    def __init__(self, token):
        self.token = token
        self.compute_hostname = None
        self.api_hostname = 'https://api.xplainable.io'

        self.init()


    def init(self):
        """ Authorize access to xplainable API.
        
            Active bearer token is required for authorization. 

        Raises:
            HTTPError: If user not authorized.
        """
        # Add token to session headers
        __session__.headers['authorization'] = f'Bearer {self.token}'

        # get response data
        response = __session__.get(f'{self.api_hostname}/get-organisation-machine')
        machine_ip = get_response_content(response)

        self.compute_hostname = f'http://{machine_ip}'
        xplainable.__client__ = self

        print("initialised")

    def list_models(self):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = __session__.get(
            url=f'{self.api_hostname}/models'
            )

        return get_response_content(response)

    def list_versions(self, model_id):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = __session__.get(
            url=f'{self.api_hostname}/models/{model_id}/versions'
            )

        return get_response_content(response)
        
    def load_model(self, model_id, version_id='latest'):
        """ Loads a model by model_id

        Args:
            model_id (str): A valid model_id

        Returns:
            xplainable.model: The loaded xplainable model
        """

        model_response = __session__.get(
            url=f'{self.api_hostname}/models/{model_id}'
            )

        model_data = get_response_content(model_response)

        if model_data is None:
            raise ValueError(f'Model with ID {model_id} does not exist')
            
        model_name = model_data['model_name']
        model_type = model_data['model_type']

        response = __session__.get(
            url=f'{self.api_hostname}/models/{model_id}/versions/{version_id}'
            )

        data = get_response_content(response)
        meta_data = data['data']['data']

        model = None
        if model_type == 'binary_classification':
            from .models.classification import XClassifier
            model = XClassifier(model_name=model_name)

        elif model_type == 'regression':
            from .models.regression import XRegressor
            model = XRegressor(model_name=model_name)
        
        model._load_metadata(meta_data)

        return model

    def get_user_data(self):
        """ Retrieves the user data for the active user.

        Returns:
            dict: User data
        """
        
        response = __session__.get(
        url=f'{self.api_hostname}/api/user'
        )

        return get_response_content(response)

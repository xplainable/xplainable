import json
from getpass import getpass
import requests
from urllib3.exceptions import HTTPError

__session__ = requests.Session()


class Client:
    """ Client for interfacing with the xplainable web api.
    """

    def __init__(self, hostname):
        self.hostname = hostname


    def authenticate(self, token=None):
        """ Login to xplainable with token or email and password.
        
            Email address or an active bearer token is required
            to login. 

        Args:
            email (str): Email of account holder
            token (str): Bearer token

        Raises:
            HTTPError: If user not authenticated
            ValueError: If no email or token specified
        """
        # Add token to session headers
        __session__.headers['authorization'] = f'Bearer {token}'

        response = __session__.get(
            url=f'{self.hostname}/api/user'
        )

        if response.status_code == 200:

            user = json.loads(response.content)['email']
            
            print(f"Authenticated as user {user}")

        elif response.status_code == 401:
            raise HTTPError("401 Invalid access token")

        else:
            raise HTTPError(response)


    def list_models(self):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = __session__.get(
            url=f'{self.hostname}/models'
            )

        if response.status_code == 200:
            return json.loads(response.content)

        elif response.status_code == 401:
            raise HTTPError(f"401 Unauthorised")

        else:
            raise HTTPError(response)

        
    def load_model(self, model_name):
        """ Loads a model by model_name

        Args:
            model_name (str): A valid model name

        Returns:
            xplainable.model: The loaded xplainable model
        """


        response = __session__.get(
            url=f'{self.hostname}/models/{model_name}'
            )

        if response.status_code == 200:

            data = json.loads(response.content)
            model_name = data['model_name']
            model_type = data['model_type']
            meta_data = data['data']['data']

            model = None
            if model_type == 'binary_classification':
                from .models.classification import XClassifier
                model = XClassifier(
                    model_name=model_name, hostname=self.hostname)

            elif model_type == 'regression':
                from .models.regression import XRegressor
                model = XRegressor(
                    model_name=model_name, hostname=self.hostname)
            
            model._load_metadata(meta_data)

            return model

        elif response.status_code == 401:
            raise HTTPError(f"401 Unauthorised")

        else:
            raise HTTPError(response)


    def get_user_data(self):
        """ Retrieves the user data for the active user.

        Returns:
            dict: User data
        """
        
        response = __session__.get(
        url=f'{self.hostname}/api/user'
        )

        if response.status_code == 200:
            user_details = json.loads(response.content)
            return user_details

        elif response.status_code == 401:
            raise HTTPError(f"401 Unauthorised")

        else:
            raise HTTPError(response)

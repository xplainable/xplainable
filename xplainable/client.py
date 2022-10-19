import json
import requests
from urllib3.exceptions import HTTPError
from authlib.jose import jwt

__session__ = requests.Session()

def get_response_content(response):
    if response.status_code == 200:
        return json.loads(response.content)

    elif response.status_code == 401:
        raise HTTPError(f"401 Unauthorised")

    else:
        raise HTTPError(response)

class Client:
    """ Client for interfacing with the xplainable web api.
    """

    def __init__(self, token):
        self.token = token
        self.hostname = None

        self.init()


    def init(self):
        """ Authorize access to xplainable API.
        
            Active bearer token is required for authorization. 

        Raises:
            HTTPError: If user not authorized.
        """
        # Add token to session headers
        __session__.headers['authorization'] = f'Bearer {self.token}'

        response = __session__.get(
            url='https://dev-k-xn6t1r.us.auth0.com/.well-known/jwks.json')

        # get response data
        jwks = get_response_content(response)
            
        try:
            key = f'''-----BEGIN CERTIFICATE-----\n{jwks["keys"][0]["x5c"][0]}\n-----END CERTIFICATE-----'''
            claims = jwt.decode(self.token, key=key)
            vmip = claims['https://www.xplainable.io/app_metadata']['vm_ip'][0]
            self.hostname = f"http://{vmip}"
            print(f"Initialised")

        except Exception as e:
            raise HTTPError(f"401 Invalid access token. {e}")

    def list_models(self):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = __session__.get(
            url=f'{self.hostname}/models'
            )

        return get_response_content(response)

    def list_perspectives(self, model_id):
        """ Lists models of active user.

        Returns:
            dict: Dictionary of trained models.
        """

        response = __session__.get(
            url=f'{self.hostname}/models/{model_id}/perspectives'
            )

        return get_response_content(response)
        
    def load_model(self, model_id, perspective_id='latest'):
        """ Loads a model by model_id

        Args:
            model_id (str): A valid model_id

        Returns:
            xplainable.model: The loaded xplainable model
        """

        model_response = __session__.get(
            url=f'{self.hostname}/models/{model_id}'
            )

        model_data = get_response_content(model_response)

        if model_data is None:
            raise ValueError(f'Model with ID {model_id} does not exist')
            
        model_name = model_data['model_name']
        model_type = model_data['model_type']

        response = __session__.get(
            url=f'{self.hostname}/models/{model_id}/perspectives/{perspective_id}'
            )

        data = get_response_content(response)
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

import json
from getpass import getpass
import requests
import webbrowser
from datetime import datetime
from IPython.display import display, HTML

__session__ = requests.Session()


class Client:

    def __init__(self, hostname):
        self.hostname = hostname

    def create_account(self):
        webbrowser.open(self.hostname)

    def login(self, email):

        body = {
            'username': email,
            'password': getpass()
        }
        
        response = __session__.post(
            url=f'{self.hostname}/login',
            data=body
        )

        if response.status_code == 200:
            content = json.loads(response.content)
            __session__.headers['authorization'] = f'''Bearer {
                content['access_token']}'''

            print("Login successful")
            return self

        else:
            return json.loads(response.content)

    def reset_password(self):

        body = {
            'old_password': getpass('Old password: '),
            'new_password': getpass('New password: '),
            'confirm_new_password': getpass('Confirm new password: ')
        }

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = __session__.post(
            url=f'{self.hostname}/reset-password',
            data=body,
            headers=headers
        )

        return json.loads(response.content)

    def list_models(self):

        response = __session__.get(
            url=f'{self.hostname}/models'
            )

        if response.status_code != 200:
            try:
                return json.loads(response.content)
            except Exception as e:
                return response.content

        else:
            return json.loads(response.content)

    def load_model(self, model_name):
        response = __session__.get(
            url=f'{self.hostname}/models/{model_name}'
            )

        if response.status_code != 200:
            try:
                return json.loads(response.content)
            except Exception as e:
                return response.content

        else:
            data = json.loads(response.content)
            model_name = data['model_name']
            model_type = data['model_type']
            meta_data = data['data']['data']

            model = None
            if model_type == 'binary_classification':
                from .models.classification import XClassifier

                model = XClassifier(model_name, self.hostname)
                model._load_metadata(meta_data)

            elif model_type == 'regression':
                from .models.regression import XRegressor
                model = XRegressor(model_name, self.hostname)
                model._load_metadata(meta_data)

            return model


    def get_user_data(self):
        
        response = __session__.get(
            url=f'{self.hostname}/get-user-data'
            )

        if response.status_code != 200:
            try:
                return json.loads(response.content)
            except Exception as e:
                return response.content

        else:
            user_details = json.loads(response.content)
            return user_details

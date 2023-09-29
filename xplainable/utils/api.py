import xplainable

from urllib3.exceptions import HTTPError
import json

def get_response_content(response):
    if response.status_code == 200:
        return json.loads(response.content)

    elif response.status_code == 401:
        err_string = "401 Unauthorised"
        content = json.loads(response.content)
        if 'detail' in content:
            err_string = err_string + f" ({content['detail']})"
        
        raise HTTPError(err_string)

    else:
        raise HTTPError(response.status_code, json.loads(response.content))

def ping_server(hostname):
    response = xplainable.client.__session.get(
        f'{hostname}/v1/compute/ping',
        timeout=1
        )
    content = json.loads(response.content)
    if content == True:
        return True
    else:
        return False

def ping_gateway(hostname):
    response = xplainable.client.__session.get(
        f'{hostname}/v1/ping',
        timeout=1
        )
    content = json.loads(response.content)
    if content == True:
        return True
    else:
        return False


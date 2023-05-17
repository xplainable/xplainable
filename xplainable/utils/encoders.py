import numpy as np
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def force_json_compliant(data, fill_value=None):
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = force_json_compliant(data[i])
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = force_json_compliant(v)
    elif isinstance(data, float):
        if np.isnan(data):
            data = fill_value
        elif np.isinf(data):
            data = fill_value
    return data


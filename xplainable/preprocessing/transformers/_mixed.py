from ._base import XBaseTransformer, TransformError
import pandas.api.types as pdtypes
from ipywidgets import interactive
import re
import pandas as pd
import ipywidgets as widgets

class SetDType(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['numeric', 'categorical']

    def __init__(self, to_type=None):
        super().__init__()
        self.to_type = to_type

    def __call__(self, ser, *args, **kwargs):
      
        def _set_params(to_type, *args, **kwargs):
            self.to_type = to_type

        def get_widget(col):
            if pdtypes.is_float_dtype(ser):
                options=["float", "integer", "string"]
                value = 'float'

            elif pdtypes.is_integer_dtype(ser):
                options=["float", "integer", "string"]
                value = 'integer'

            elif pdtypes.is_string_dtype(ser):
                options=["string"]
                if all(ser.str.isdigit()):
                    options += ["float", "integer"]
                value = 'string'
            
            elif pdtypes.is_datetime64_dtype(ser):
                options = ["date", "string"]
                value = "date"

            elif pdtypes.is_bool_dtype(ser):
                options = ["boolean", "string", "integer", "float"]
                value = "boolean"

            return widgets.Dropdown(options=options, value=value)

        col_widget = {'to_type': get_widget(ser)}  

        return interactive(_set_params, **col_widget)

    def _operations(self, ser):

        mapp = {
            'integer': int,
            'float': float,
            'string': str
        }
            
        return ser.astype(mapp[self.to_type])


class Shift(XBaseTransformer):
    """ Shifts a series up of down n steps

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Attributes for ipywidgets
    supported_types = ['categorical', 'numeric']

    def __init__(self, step=0):
        super().__init__()
        self.step = step

    def __call__(self, *args, **kwargs):
        
        def _set_params(
            step = widgets.IntText(value=0, min=-1000, max=1000)
            ):
            self.step = step
        
        return interactive(_set_params)

    def _operations(self, ser):
        ser = ser.shift(self.step)

        return ser



""" Copyright Xplainable Pty Ltd, 2023"""

from .base import XBaseTransformer
import pandas.api.types as pdtypes
import pandas as pd


class SetDType(XBaseTransformer):
    """Changes the data type of a specified column."""

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['numeric', 'categorical']

    def __init__(self, to_type=None):
        super().__init__()
        self.to_type = to_type

    def __call__(self, ser, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
      
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
                #options=["string"]
                options = ["float", "integer", "string"]
                #if all(ser.str.isdigit()):
                #    options += ["float", "integer"]
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

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Changes the data type of a specified column.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        if self.to_type == 'string':
            return ser.astype(str)
            
        ser = pd.to_numeric(ser, errors='coerce')

        if self.to_type == 'integer':
            # If missing value are present, cannot cast to int
            try:
                ser = ser.astype(int)
            except Exception:
                pass

        return ser


class Shift(XBaseTransformer):
    """ Shifts a series up or down n steps.

    Args:
        step (str): The number of steps to shift.
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical', 'numeric']

    def __init__(self, step=0):
        super().__init__()
        self.step = step

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        def _set_params(
            step = widgets.IntText(value=0, min=-1000, max=1000)
            ):
            self.step = step
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Shifts a series up or down n steps.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        ser = ser.shift(self.step)

        return ser

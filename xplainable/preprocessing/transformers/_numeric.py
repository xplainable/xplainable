from ._base import XBaseTransformer, TransformError
from ipywidgets import interactive
import ipywidgets as widgets
from pandas.api.types import is_numeric_dtype, is_string_dtype
import numpy as np


class MinMaxScale(XBaseTransformer):
    """ Scales a series between 0 and 1.

    Attributes:
        min (float): The minimum values from the fitted series.
        max (float): The maximum values from the fitted series.

    """

    # Attributes for ipywidgets
    supported_types = ['numeric']

    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def _operations(self, ser):
        
        return (ser - self.min_value) / self.max_value

    def fit(self, ser):
        """ Extracts the min and max value from a series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            self
        """

        # Store min and max from training data
        self.min_value = ser.min()
        self.max_value = ser.max()

        return self

class LogTransform(XBaseTransformer):
    """ Log transforms a given series.
    """

    # Attributes for ipywidgets
    supported_types = ['numeric']

    def __init__(self):
        pass

    def _operations(self, ser):
        return np.log(ser, where=(ser.values != 0))

    def _inverse_operations(self, ser):
        return np.exp(ser)


class Clip(XBaseTransformer):
    """ Clips numeric values to a specified range

    Args:
        lower (float): The lower threshold.
        upper (float): The upper threshold value.

    """

    supported_types = ['numeric']

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def __call__(self, column, *args, **kwargs):
        
        minn = column.min()
        maxx = column.max()
        
        
        def _set_params(n=widgets.FloatRangeSlider(min=minn,max=maxx)):
            self.lower, self.upper = n
        
        return interactive(_set_params)

    def _operations(self, ser):

        if not is_numeric_dtype(ser):
            raise TypeError(f'Series must be numeric type.')

        # Apply value replacement
        return np.clip(ser, self.lower, self.upper)


class FillMissingNumeric(XBaseTransformer):
    """ Fills missing values with a specified value.

    Args:
        fill_with (str): ['mean', 'median', 'mode'] or raw text.

    Attributes:
        fill_with (str): The selected fill instruction.
        fill_value (): The calculated fill value.
    """

    supported_types = ['numeric']

    def __init__(self, fill_with='mean', fill_value=None):
        super().__init__()
        self.fill_with = fill_with
        self.fill_value = fill_value

    def __call__(self, *args, **kwargs):
        
        def _set_params(
            fill_with = widgets.Dropdown(options=["mean", "median", "mode"])
        ):
            self.fill_with = fill_with
        
        return interactive(_set_params)

    def _operations(self, ser):

        return ser.fillna(self.fill_value)

    def fit(self, ser):
        """ Calculates the fill_value from a given series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            self
        """

        # Calculate fill_value if mean, median or mode
        if self.fill_with == 'mean':
            self.fill_value = np.nanmean(ser)

        elif self.fill_with == 'median':
            self.fill_value = np.nanmedian(ser)

        elif self.fill_with == 'mode':
            self.fill_value = ser.mode()

        # Maintain fill value type (if int)
        if ser.dtype == int:
            self.fill_value = int(self.fill_value)

        return self
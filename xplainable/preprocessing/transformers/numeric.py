""" Copyright Xplainable Pty Ltd, 2023"""

from .base import XBaseTransformer
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd


class MinMaxScale(XBaseTransformer):
    """Scales a numeric series between 0 and 1."""

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['numeric']

    def __init__(self, min_value=None, max_value=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Scale a numeric series between 0 and 1.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        
        return (ser - self.min_value) / self.max_value

    def fit(self, ser: pd.Series) -> 'MinMaxScale':
        """ Extracts the min and max value from a series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            MinMaxScale: The fitted transformer.
        """

        # Store min and max from training data
        self.min_value = ser.min()
        self.max_value = ser.max()

        return self

class LogTransform(XBaseTransformer):
    """ Log transforms a given numeric series."""

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['numeric']

    def __init__(self):
        pass

    def transform(self, ser: pd.Series) -> pd.Series:
        """ 

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        return np.log(ser, where=(ser.values != 0))

    def inverse_transform(self, ser: pd.Series) -> pd.Series:
        """ 

        Args:
            ser (pd.Series): The series to inverse transform.

        Returns:
            pd.Series: The inverse transformed series.
        """
        return np.exp(ser)


class Clip(XBaseTransformer):
    """ Clips numeric values to a specified range.

    Args:
        lower (float): The lower threshold value.
        upper (float): The upper threshold value.

    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['numeric']

    def __init__(self, lower=None, upper=None):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def __call__(self, column, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        minn = column.min()
        maxx = column.max()
        
        
        def _set_params(n=widgets.FloatRangeSlider(min=minn,max=maxx)):
            self.lower, self.upper = n
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ 

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        if not is_numeric_dtype(ser):
            raise TypeError(f'Series must be numeric type.')

        # Apply value replacement
        return np.clip(ser, self.lower, self.upper)


class FillMissingNumeric(XBaseTransformer):
    """ Fills missing values with a specified strategy.

    Args:
        fill_with (str): The strategy ['mean', 'median', 'mode'].
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['numeric']

    def __init__(self, fill_with='mean', fill_value=None):
        super().__init__()
        self.fill_with = fill_with
        self.fill_value = fill_value

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        def _set_params(
            fill_with = widgets.Dropdown(options=["mean", "median", "mode"])
        ):
            self.fill_with = fill_with
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ 

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        return ser.fillna(self.fill_value)

    def fit(self, ser: pd.Series) -> 'FillMissingNumeric':
        """ Calculates the fill value from a series.

        Args:
            ser (pandas.Series): The series to analyse.

        Returns:
            FillMissingNumeric: The fitted transformer.
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

import pandas as pd
import numpy as np
import re
from ipywidgets import interactive
import ipywidgets as widgets
from pandas.api.types import is_numeric_dtype, is_string_dtype


class TransformError(Exception):
    """Raise when error in transformation"""
    pass


class XBaseTransformer():

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def _operations(self):
        """Override with transform operations"""
        pass

    def _inverse_operations(self):
        """Override with inverse operations"""
        pass

    def fit(self, *args, **kwargs):
        """Override with fit method"""
        return self

    def transform(self, ser):
        """ Applies feature transformation.

        Args:
            ser (pandas.Series): The series in which to transform.

        Returns:
            pandas.Series: The transformed series.
        """

        # Apply type change
        try:
            ser = self._operations(ser)
        except Exception as e:
            raise TransformError(
                f'Could not apply {self.__class__.__name__} transformation for {ser.name} because:\n {e}')

        return ser

    def fit_transform(self, ser):
        """ Fit and transforms data

        Args:
            ser (pandas.Series): The series in which to transform.

        Returns:
            pandas.Series: The transformed series.
        """

        return self.fit(ser).transform(ser)

    def inverse_transform(self, ser):
        """ Applies feature inverse transformation.

        Args:
            ser (pandas.Series): The series in which to transform.

        Returns:
            pandas.Series: The transformed series.
        """

        # Apply type change
        try:
            ser = self._inverse_operations(ser)
        except Exception as e:
            raise TransformError(
                f'Could not apply {self.__class__.__name__} inverse transformation for {ser.name} because:\n {e}')

        return ser



class ReplaceWith(XBaseTransformer):
    """ Replaces specified value in series

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Attributes for ipywidgets
    supported_types = ['categorical']

    def __init__(self, target=None, replace_with=None):
        super().__init__()
        self.target = target
        self.replace_with = replace_with

    def __call__(self, *args, **kwargs):
        
        def _set_params(target = '', replace_with = ''):
            self.target = target
            self.replace_with = replace_with
        
        return interactive(_set_params)

    def _operations(self, ser):
        
        return ser.str.replace(self.target, self.replace_with)






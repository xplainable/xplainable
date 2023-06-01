""" Copyright Xplainable Pty Ltd, 2023"""

from ...utils.exceptions import TransformError
from typing import Union
import pandas as pd
import functools


class XBaseTransformer():
    """ Base class for all transformers.

    This base class is used as a template for all xplainable transformers.
    It contains the basic methods that all transformers should have, and is
    used to enforce a consistent API across all transformers.

    the __call__ method is used to allow the transformers to be called inside
    the xplainable gui in jupyter, but does not need to be called.
    """

    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        pass

    def raise_errors(func):
        """ Decorator to raise detailed errors in transformer functions.

        This decorator is used to wrap the transformer methods and raise any
        errors that occur during processing. This is done to allow
        the gui to catch the errors and display them.
        """
        @functools.wraps(func) # This preserves the name and docstrings
        def wrapper(self, x):
            try:
                return func(self, x)
            except Exception as e:
                raise TransformError(
                    f'Could not run method {func.__name__} for class {self.__class__.__name__} because:\n {e}')
        return wrapper
    
    @raise_errors
    def fit(self, *args, **kwargs):
        """ No fit is required for this transformer.
        
        This is a default fit method in case no fit is needed. This method is
        used to allow the transformer to be used in a pipeline, and is intended
        to be overridden by transformers that require fitting.

        Decorators:
            raise_errors (decorator): Raises detailed errors.
        """

        return self

    @raise_errors
    def transform(self, x: Union[pd.Series, pd.DataFrame]):
        """ Placeholder for transformation operation. Intended to be overridden.

        The input parameter is either a pd.Series or a pd.DataFrame, depending
        on the transformer. Documentation for each individual transformer should
        specify which type of input is expected in this method when it is being
        overridden.

        Args:
            x (pd.Series | pd.DataFrame): To be specified by transformer.

        Decorators:
            raise_errors (decorator): Raises detailed errors.
        """

        return x
    
    @raise_errors
    def inverse_transform(self, x: Union[pd.Series, pd.DataFrame]):
        """ No inverse transform is available for this transformer.

        This is a default inverse method in case no inverse transform is
        available.
        
        The input parameter is either a pd.Series or a pd.DataFrame, depending
        on the transformer. Documentation for each individual transformer should
        specify which type of input is expected in this method when it is being
        overridden.

        Args:
            x (pd.Series | pd.DataFrame): To be specified by transformer.

        Decorators:
            raise_errors (decorator): Raises detailed errors.
        """

        return x

    def fit_transform(self, x: Union[pd.Series, pd.DataFrame]):
        """ Fit and transforms data on a series or dataframe.

        Args:
            x (pd.Series | pd.DataFrame): Series or df to fit & transform.

        Returns:
            pandas.Series: The transformed series or df.
        """

        return self.fit(x).transform(x)

    

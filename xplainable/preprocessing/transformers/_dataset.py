from ._base import XBaseTransformer, TransformError
import pandas.api.types as pdtypes
from ipywidgets import interactive
from IPython.display import display
import re
import numpy as np
import ipywidgets as widgets


class DropCols(XBaseTransformer):
    """ Drops a specified column from a dataset.

    Args:
        column (str): The column to be dropped.

    Attributes:
        column (str): The column to be dropped.
    """

    supported_types = ['dataset']

    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(columns=widgets.SelectMultiple(options=dataset.columns)):

            self.columns = list(columns)

        return interactive(_set_params)

    def _operations(self, df):
        
        df = df.copy()

        # Apply column dropping
        for c in self.columns:
            if c in df.columns:
                df = df.drop(columns=[c])

        return df

class DropNaNs(XBaseTransformer):
    """ Drops nan rows from a dataset.
    """

    supported_types = ['dataset']

    def __init__(self):
        super().__init__()

    def _operations(self, df):

        return df.copy().dropna()


class AddCols(XBaseTransformer):
    """ Adds multiple numeric columns into single feature.

    Args:
        columns (list): column names to join
    """
    
    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, columns=[], alias: str = None, drop: bool = False):
        super().__init__()
        self.columns = columns
        self.alias = alias if alias else " + ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_add=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_add)
            self.alias = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        if not all([pdtypes.is_numeric_dtype(df[col]) for col in self.columns]):
            raise TypeError("Cannot add string and numeric columns")

        df = df.copy()

        df[self.alias] = df[self.columns].sum(axis=1)
        if self.drop:
            df = df.drop(columns=self.columns)

        return df

class MultiplyCols(XBaseTransformer):
    """ Multiplies multiple numeric columns into single feature.

    Args:
        columns (list): column names to join
    """

    supported_types = ['dataset']

    def __init__(self, columns=[], alias: str = None, drop: bool = False):
        super().__init__()
        self.columns = columns
        self.alias = alias if alias else " * ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_multiply=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_multiply)
            self.alias = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        if not all([pdtypes.is_numeric_dtype(df[col]) for col in self.columns]):
            raise TypeError("Cannot multiply string and numeric columns")

        df = df.copy()

        df[self.alias] = df[self.columns].prod(axis=1)
        if self.drop:
            df = df.drop(columns=self.columns)

        return df

class ConcatCols(XBaseTransformer):
    """ Concatenates multiple numeric columns into single feature.

    Args:
        columns (list): column names to join
    """

    supported_types = ['dataset']

    def __init__(self, columns=[], alias: str = None, drop: bool = False):
        super().__init__()
        self.columns = columns
        self.alias = alias if alias else " + ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_concat=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_concat)
            self.alias = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        for col in self.columns:
            if not pdtypes.is_string_dtype(df[col]):
                df[col] = df[col].astype(str)

        df = df.copy()

        df[self.alias] = df[self.columns].agg('-'.join, axis=1)
        if self.drop:
            df = df.drop(columns=self.columns)

        return df


class ChangeNames(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, col_names={}):
        super().__init__()
        self.col_names = col_names

    def __call__(self, df, *args, **kwargs):
      
        def _set_params(*args, **kwargs):
            args = locals()
            self.col_names = dict(args)['kwargs']

        col_names = {col: widgets.Text(col) for col in df.columns}  

        return interactive(_set_params, **col_names)

    def _operations(self, df):
        df = df.copy()
        return df.rename(columns=self.col_names)


class OrderBy(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, order_by=None, ascending=True):
        super().__init__()
        self.order_by = order_by
        self.ascending = ascending

    def __call__(self, df, *args, **kwargs):
      
        def _set_params(
            order_by = widgets.SelectMultiple(description='Order by: ', options=[None]+list(df.columns)),
            direction = widgets.ToggleButtons(options=['ascending', 'descending'])
        ):

            self.order_by = list(order_by)
            self.ascending = True if direction == 'ascending' else False

        return interactive(_set_params)

    def _operations(self, df):

        return df.sort_values(self.order_by, ascending=self.ascending)


class GroupbyShift(XBaseTransformer):
    """ Shifts a series up of down n steps

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, target=None, step=0, as_new=None, group_by=None, order_by=None, descending=None):
        super().__init__()
        self.target = target
        self.step = step
        self.as_new = as_new

        self.group_by = group_by
        self.order_by = order_by
        self.descending = descending

    def __call__(self, df, *args, **kwargs):
        
        def _set_params(
            group_by = widgets.SelectMultiple(description='Group by: ', options=[None]+list(df.columns)),
            order_by = widgets.SelectMultiple(description='Order by: ', options=[None]+list(df.columns)),
            descending = widgets.Checkbox(value=False),
            target=widgets.Dropdown(options=[None]+list(df.columns)),
            step = widgets.IntText(value=0, min=-1000, max=1000),
            as_new = widgets.Checkbox(value=False)
            ):

            self.target = target
            self.group_by = list(group_by)
            self.order_by = list(order_by)
            self.descending = descending
            self.step = step

            self.as_new = target

            # build new col name if as_new
            if as_new and target is not None:
                col_name = f'{target}_shift_{step}'
                if len(self.group_by) > 0:
                    col_name += "_gb_"
                    col_name += '_'.join(self.group_by)

                if len(self.order_by) > 0:
                    col_name += "_ob_"
                    col_name += '_'.join(self.order_by)
            
                self.as_new = col_name

        return interactive(_set_params)

    def _operations(self, df):

        # Order values if
        if self.order_by and self.order_by[0] is not None:
            df = df.sort_values(self.order_by, ascending=not self.descending)

        if self.group_by and self.group_by[0] is not None:
            df[self.as_new] = df.groupby(self.group_by)[self.target].shift(self.step)
        else:
            df[self.as_new] = df[self.target].shift(self.step)

        return df


class FillMissing(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, fill_with={}, fill_values={}):
        super().__init__()
        self.fill_with = fill_with
        self.fill_values = fill_values

    def __call__(self, df, *args, **kwargs):
      
        def _set_params(*args, **kwargs):
            args = locals()
            self.fill_with = dict(args)['kwargs']

        def get_widget(col):
            if pdtypes.is_numeric_dtype(df[col]):
                return widgets.Dropdown(options=["mean", "median", "mode"])
            else:
                return widgets.Text(value='missing')

        col_widgets = {col: get_widget(col) for col in df.columns}  

        return interactive(_set_params, **col_widgets)

    def _operations(self, df):
        
        for i, v in self.fill_values.items():
            df[i] = df[i].fillna(v)

        return df

    def fit(self, df):
        """ Calculates the fill_value from a given series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            self
        """

        for i, v in self.fill_with.items():
            # Calculate fill_value if mean, median or mode
            if v == 'mean':
                self.fill_values[i] = round(np.nanmean(df[i]), 4)

            elif v == 'median':
                self.fill_values[i] = np.nanmedian(df[i])

            elif v == 'mode':
                self.fill_values[i] = df[i].mode()

            else:
                self.fill_values[i] = v

            # Maintain fill value type (if int)
            if pdtypes.is_integer_dtype(df[i]):
                self.fill_values[i] = int(self.fill_values[i])

        return self


class SetDTypes(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, types={}):
        super().__init__()
        self.types = types

    def __call__(self, df, *args, **kwargs):
      
        def _set_params(*args, **kwargs):
            args = locals()
            self.types = dict(args)['kwargs']

        def get_widget(col):
            if pdtypes.is_float_dtype(df[col]):
                options=["float", "integer", "string"]
                value = 'float'

            elif pdtypes.is_integer_dtype(df[col]):
                options=["float", "integer", "string"]
                value = 'integer'

            elif pdtypes.is_string_dtype(df[col]):
                options=["string"]
                if all(df[col].str.isdigit()):
                    options += ["float", "integer"]
                value = 'string'
            
            elif pdtypes.is_datetime64_dtype(df[col]):
                options = ["date", "string"]
                value = "date"

            elif pdtypes.is_bool_dtype(df[col]):
                options = ["boolean", "string", "integer", "float"]
                value = "boolean"

            w = widgets.Dropdown(
                options=options,
                value=value,
                style={'description_width': 'initial'})

            return w

        col_widgets = {col: get_widget(col) for col in df.columns}  

        return interactive(_set_params, **col_widgets)

    def _operations(self, df):

        mapp = {
            'integer': int,
            'float': float,
            'string': str
        }

        for i, v in self.types.items():
            df[i] = df[i].astype(mapp[v])
            
        return df


class TextSplit(XBaseTransformer):
    """ Remove specified values from string.

    Args:
        numbers (bool, optional): Removes numbers from string.
        characters (bool, optional): Removes characters from string.
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, target=None, separator=None, max_splits=0):

        self.target = target
        self.separator = separator
        self.max_splits = max_splits

    def __call__(self, df, *args, **kwargs):
        
        def _set_params(
            target=widgets.Dropdown(options=[None]+[i for i in df.columns if pdtypes.is_string_dtype(df[i])]),
            separator = widgets.Text(value=""),
            max_splits = widgets.IntText(range=[0,10])
        ):
            self.target = target
            self.separator = separator
            self.max_splits = max([max_splits, 0])

        return interactive(_set_params)

    def _operations(self, df):

        new_cols = df[self.target].astype(str).str.split(self.separator, expand=True, n=self.max_splits)
        new_cols.columns = [f'{self.target}_{i}' for i in new_cols]
        df[new_cols.columns] = new_cols
        df = df.drop(columns=[self.target])

        return df


class ChangeCases(XBaseTransformer):
    """ Changes the case of a string.

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self, columns=[], case='lower'):
        super().__init__()
        self.columns = columns
        self.case = case

    def __call__(self, df, *args, **kwargs):
        
        def _set_params(
            columns = widgets.SelectMultiple(description='Columns: ', options=[None]+[i for i in df.columns if pdtypes.is_string_dtype(df[i])]),
            case = ["lower", "upper"]):
            
            self.columns = list(columns)
            self.case = case
        
        return interactive(_set_params)

    def _operations(self, df):

        for col in self.columns:

            if self.case == 'lower':
                df[col] = df[col].str.lower()

            elif self.case == 'upper':
                df[col] = df[col].str.upper()

            else:
                raise ValueError("case change must be either 'lower' or 'upper'")

        return df
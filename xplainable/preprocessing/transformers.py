from ctypes import c_int16
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


class ChangeType(XBaseTransformer):
    """ Changes the type of a given series.

    Args:
        target_type (str): The type to change the series to.

    Attributes:
        target_type (str): The type to change the series to.
    """

    # Attributes for ipywidgets
    supported_types = ['numeric', 'categorical', 'date']

    def __init__(self, target_type='string'):
        super().__init__()
        self.target_type = target_type


    def __call__(self, *args, **kwargs):
        
        def _set_params(target_type = ['string', 'numeric', 'float',
        'integer', 'date']):
            self.target_type = target_type
        
        return interactive(_set_params)
            

    def _operations(self, ser):
        """ Changes the type of a series to a target type.

        Args:
            ser (pandas.Series): The series in which to transform

        Raises:
            TypeError: If the transformation is invalid for the data type.

        Returns:
            pd.Series: The transformed series.
        """

        ser = ser.copy()
        target_type = self.target_type

        # Attempt feature transformations
        if target_type == 'string':
            try:
                return ser.astype(str)
            except Exception as e:
                raise ValueError(e)

        elif target_type == 'numeric':
            try:
                return pd.to_numeric(ser, errors='coerce')
            except Exception as e:
                raise ValueError(e)

        elif target_type == 'float':
            try:
                return ser.astype(float)
            except Exception as e:
                raise ValueError(e)

        elif target_type == 'integer':
            try:
                if ser.isna().sum() > 0:
                    ser[~ser.isna()] = ser[~ser.isna()].astype(int)
                else:
                    ser = ser.astype(int)
                return ser
            except Exception as e:
                raise ValueError(e)

        elif target_type == 'date':
            try:
                return pd.to_date(ser)
            except Exception as e:
                raise ValueError(e)

        else:
            raise ValueError(f"{target_type} no in accepted types")


class MinMaxScale(XBaseTransformer):
    """ Scales a series between 0 and 1.

    Attributes:
        min (float): The minimum values from the fitted series.
        max (float): The maximum values from the fitted series.

    """

    # Attributes for ipywidgets
    supported_types = ['numeric']

    def __init__(self):
        super().__init__()
        self.min = None
        self.max = None

    def _operations(self, ser):
        
        return (ser - self.min) / self.max

    def fit(self, ser):
        """ Extracts the min and max value from a series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            self
        """

        # Store min and max from training data
        self.min = ser.min()
        self.max = ser.max()

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


class TextRemove(XBaseTransformer):
    """ Remove specified values from string.

    Args:
        numbers (bool, optional): Removes numbers from string.
        characters (bool, optional): Removes characters from string.
        uppercase (bool, optional): Removes uppercase characters from string.
        special (bool, optional): Removes lowercase characters from string.
        whitespace (bool, optional): Removes whitespace from string.
        text (str, optional): Removes specific text match from string.
        custom_regex (str, optional): Removes matching regex text from string.
    """

    # Attributes for ipywidgets
    supported_types = ['categorical']

    def __init__(self, numbers=False, characters=False, uppercase=False, \
        lowercase=False, special=False, whitespace=False, text=None, custom_regex=None):

        self.numbers = numbers
        self.characters = characters
        self.uppercase = uppercase
        self.lowercase = lowercase
        self.special = special
        self.whitespace = whitespace
        self.text = text
        self.custom_regex = custom_regex

    def __call__(self, *args, **kwargs):
        
        def _set_params(numbers=False, characters=False, uppercase=False, \
        lowercase=False, special=False, whitespace=False, text='', regex=''):
            self.numbers = numbers
            self.characters = characters
            self.uppercase = uppercase
            self.lowercase = lowercase
            self.special = special
            self.whitespace = whitespace
            if text == '':
                self.text = None
            else:
                self.text = text
            if regex == '':
                self.custom_regex = None
            else:
                self.custom_regex = regex

        w = interactive(_set_params)
        checkboxes_a = widgets.VBox(w.children[:3])
        checkboxes_b = widgets.VBox(w.children[3:6])
        checkboxes_a.layout = widgets.Layout(margin='0 0 0 -50px')
        checkboxes_b.layout = widgets.Layout(margin='0 0 0 -150px')

        checkboxes = widgets.HBox([checkboxes_a, checkboxes_b])

        textboxes = widgets.VBox(w.children[6:8])
        textboxes.layout = widgets.Layout(margin='0 0 0 -125px')

        box = widgets.VBox([checkboxes, textboxes])

        return box

    def _operations(self, ser):
        matches = []
        if self.numbers:
            matches.append(r'[0-9]')

        if self.characters:
            matches.append(r'[a-zA-Z]')

        else:
            if self.uppercase:
                matches.append(r'[A-Z]')

            if self.lowercase:
                matches.append(r'[a-z]')

        if self.special:
            matches.append(r'[^a-zA-Z0-9 ]')

        if self.whitespace:
            matches.append(r' ')

        if self.custom_regex:
            matches.append(self.custom_regex)

        if len(matches) > 0:
            regex = re.compile("|".join(matches))
            ser[~ser.isna()] = ser[~ser.isna()].apply(lambda x: regex.sub('', x))

        if self.text:
            ser = ser.str.replace(self.text, "")

        return ser


class CaseChange(XBaseTransformer):
    """ Changes the case of a string.

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Attributes for ipywidgets
    supported_types = ['categorical']

    def __init__(self, case='lower'):
        super().__init__()
        self.case = case

    def __call__(self, *args, **kwargs):
        
        def _set_params(case = ["lower", "upper"]):
            self.case = case
        
        return interactive(_set_params)

    def _operations(self, ser):
        if self.case == 'lower':
            return ser.str.lower()

        if self.case == 'upper':
            return ser.str.upper()

        else:
            raise ValueError("case change must be either 'lower' or 'upper'")

class FillMissing(XBaseTransformer):
    """ Fills missing values with a specified value.

    Args:
        fill_with (str): ['mean', 'median', 'mode'] or raw text.

    Attributes:
        fill_with (str): The selected fill instruction.
        fill_value (): The calculated fill value.
    """

    supported_types = ['numeric', 'categorical']

    def __init__(self, fill_with='missing'):
        super().__init__()
        self.fill_with = fill_with
        self.fill_value = None

    def __call__(self, *args, **kwargs):
        
        def _set_params(fill_with = "missing"):
            self.fill_with = fill_with
        
        return interactive(_set_params)

    def _operations(self, ser):
        # Converts "" into np.nan to be filled
        if is_string_dtype(ser):
            ser = ser.apply(lambda x: np.nan if x == "" else x)

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
            self.fill_value = ser.mean()

        elif self.fill_with == 'median':
            self.fill_value = ser.median()

        elif self.fill_with == 'mode':
            self.fill_value = ser.mode()

        # Otherwise will with string
        else:
            self.fill_value = self.fill_with

        # Maintain fill value type (if int)
        if ser.dtype == int:
            self.fill_value = int(self.fill_value)

        return self


class DetectCategories(XBaseTransformer):
    """ Detects categories from a string columns.

    Args:
        max_categories (int): The maximum number of categories to extract.

    Attributes:
        max_categories (str): The maximum number of categories to extract.
        category_list (list): The fitted category list.
    """

    supported_types = ['categorical']

    def __init__(self, max_categories=10):
        super().__init__()
        self.max_categories = int(max_categories)
        self.category_list = []

    def __call__(self, *args, **kwargs):
        
        def _set_params(max_categories=(2, 50)):
            self.max_categories = max_categories
        
        return interactive(_set_params)

    def _operations(self, ser):

        if not is_string_dtype(ser):
            raise TypeError(f'Cannot detect categories for non-text field.')

        cl = self.category_list

        # Map top categories if exists else 'other'
        ser = ser.fillna('missing')
        ser = ser.str.split().apply(
            lambda x: list(
                set([y for y in x if y in cl]))[0] if (
                    len(list(set([y for y in x if y in
                                    cl]))) > 0) else 'other')

        return ser

    def fit(self, ser):
        """ Identifies the top categories from a text series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            self
        """

        # Get the top n categories based on count
        ser = ser.fillna('missing')
        self.category_list = list(ser.str.split().explode(
        ).value_counts().head(self.max_categories).index)

        return self


class Condense(XBaseTransformer):
    """ Condenses a feature into categories that make up x pct of obserations.

    Args:
        pct (int): The minumum pct of observations the categories should cover.

    Attributes:
        pct (str): The minumum pct of observations the categories should cover.
        categories (list): The calculated category list.
    """

    supported_types = ['categorical']

    def __init__(self, pct=0.8):
        super().__init__()
        self.pct = pct
        self.categories = []

    def __call__(self, *args, **kwargs):
        
        def _set_params(pct=(0, 100)):
            self.pct = pct / 100
        
        return interactive(_set_params)

    def fit(self, ser):
        """ Determines the categories that make up x pct of obserations.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Raises:
            TypeError: If the series is not of type string.
        """

        if not is_string_dtype(ser):
            raise TypeError(f'Series must be of type string.')
            
        # calculate the pct of observations the category makes up
        vc = ser.value_counts() / len(ser)

        # Instantiate trackers
        cum_pct = 0
        top_categories = None

        # Iterate through top categories until minimum pct is reached
        for i, v in enumerate(vc):
            cum_pct += v
            top_categories = i + 1
            if cum_pct >= self.pct:
                break

        self.categories = list(vc[:top_categories].index)

        return self      

    def _operations(self, ser):
        if not is_string_dtype(ser):
            raise ValueError(f'Cannot condense categories for non-text field.')

        # Convert non-top categories to 'other
        return (ser.isin(self.categories) * ser).replace("", 'other')


class MergeCategories(XBaseTransformer):
    """ Merges specified categories in a series into one category.

    Args:
        categories (list): The list of categories. First category is target.

    Attributes:
        categories (list): The list of categories. First category is target.
    """

    supported_types = ['categorical']

    def __init__(self, merge_from=[], merge_to=''):
        super().__init__()
        self.merge_from = merge_from
        self.merge_to = merge_to

    def __call__(self, column, *args, **kwargs):

        unq = column.unique()

        
        def _set_params(merge_from=widgets.SelectMultiple(options=unq), merge_to=unq):
            self.merge_from = list(merge_from)
            self.merge_to = merge_to
        
        return interactive(_set_params)

    def _operations(self, ser):

        # Apply merging
        return ser.apply(lambda x: self.merge_to if x in self.merge_from else x)


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


class Replace(XBaseTransformer):
    """ Replaces values in a series with specified values.

    Args:
        target: The target value to replace.
        replace_with: The value to insert in place

    """

    supported_types = ['numeric', 'categorical']

    def __init__(self, target=None, replace_with=''):
        super().__init__()
        self.target = target
        self.replace_with = replace_with

    def __call__(self, column, *args, **kwargs):
        
        
        def _set_params(target=column.unique(), replace_with=''):
            self.target = target
            self.replace_with = replace_with
        
        return interactive(_set_params)

    def _operations(self, ser):
        # Apply value replacement
        return ser.replace(self.target, self.replace_with)


class DropCol(XBaseTransformer):
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

    def __init__(self, column=None):
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
        self.name = alias if alias else " + ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_add=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_add)
            self.name = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        if not all([is_numeric_dtype(df[col]) for col in self.columns]):
            raise TypeError("Cannot add string and numeric columns")

        df = df.copy()

        df[self.name] = df[self.columns].sum(axis=1)
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
        self.name = alias if alias else " * ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_multiply=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_multiply)
            self.name = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        if not all([is_numeric_dtype(df[col]) for col in self.columns]):
            raise TypeError("Cannot multiply string and numeric columns")

        df = df.copy()

        df[self.name] = df[self.columns].prod(axis=1)
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
        self.name = alias if alias else " + ".join([c for c in columns])
        self.drop = drop

    def __call__(self, dataset, *args, **kwargs):
        
        def _set_params(
            columns_to_concat=widgets.SelectMultiple(options=dataset.columns),
            alias='',
            drop_columns=True):

            self.columns = list(columns_to_concat)
            self.name = alias
            self.drop = drop_columns
        
        return interactive(_set_params)

    def _operations(self, df):

        for col in self.columns:
            if not is_string_dtype(df[col]):
                df[col] = df[col].astype(str)

        df = df.copy()

        df[self.name] = df[self.columns].agg('-'.join, axis=1)
        if self.drop:
            df = df.drop(columns=self.columns)

        return df


class ChangeNames(XBaseTransformer):
    """ Changes names of columns in a dataset
    """

    # Attributes for ipywidgets
    supported_types = ['dataset']

    def __init__(self):
        super().__init__()
        self.col_names = {}

    def __call__(self, df, *args, **kwargs):
      
        def _set_params(*args, **kwargs):
            args = locals()
            self.col_names = dict(args)['kwargs']

        col_names = {col: widgets.Text(col) for col in df.columns}  

        return interactive(_set_params, **col_names)

    def _operations(self, df):
        df = df.copy()
        return df.rename(columns=self.col_names)


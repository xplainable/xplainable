""" Copyright Xplainable Pty Ltd, 2023"""

from ...utils import stopwords

from .base import XBaseTransformer
from pandas.api.types import is_string_dtype
import re
import numpy as np
import pandas as pd


class TextRemove(XBaseTransformer):
    """ Remove specified values from a str type series.

    This transformer cannot be inverse_transformed and does not require fitting.

    Args:
        numbers (bool, optional): Removes numbers from string.
        characters (bool, optional): Removes characters from string.
        uppercase (bool, optional): Removes uppercase characters from string.
        lowercase (bool, optional): Removes lowercase characters from string.
        special (bool, optional): Removes special characters from string.
        whitespace (bool, optional): Removes whitespace from string.
        stopwords (bool, optional): Removes stopwords from string.
        text (str, optional): Removes specific text match from string.
        custom_regex (str, optional): Removes matching regex text from string.
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, numbers=False, characters=False, uppercase=False, \
        lowercase=False, special=False, whitespace=False, stopwords=False, \
            text=None, custom_regex=None):
        super().__init__()
        
        self.numbers = numbers
        self.characters = characters
        self.uppercase = uppercase
        self.lowercase = lowercase
        self.special = special
        self.whitespace = whitespace
        self.stopwords = stopwords
        self.text = text
        self.custom_regex = custom_regex

    def __call__(self, *args, **kwargs):
        import ipywidgets as widgets
        from ipywidgets import interactive
        from ...utils import xwidgets
        
        def _set_params(
            numbers=widgets.ToggleButton(value=False),
            characters=widgets.ToggleButton(value=False),
            uppercase=widgets.ToggleButton(value=False),
            lowercase=widgets.ToggleButton(value=False),
            special=widgets.ToggleButton(value=False),
            whitespace=widgets.ToggleButton(value=False),
            stopwords=widgets.ToggleButton(value=False),
            text=xwidgets.Text(''),
            regex=xwidgets.Text('')
        ):

            self.numbers = numbers
            self.characters = characters
            self.uppercase = uppercase
            self.lowercase = lowercase
            self.special = special
            self.whitespace = whitespace
            self.stopwords = stopwords
            if text == '':
                self.text = None
            else:
                self.text = text
            if regex == '':
                self.custom_regex = None
            else:
                self.custom_regex = regex

        w = interactive(_set_params)
        left = widgets.VBox(w.children[:4])
        right = widgets.VBox(w.children[4:7])
        buttons = widgets.HBox([left, right])
        
        text = widgets.VBox(
            w.children[7:], layout=widgets.Layout(margin='20px 0 0 0'))
        
        elements = widgets.VBox([buttons, text])

        return elements

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Removes specified values from a str type series.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
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
            ser[~ser.isna()] = ser[~ser.isna()].apply(
                lambda x: regex.sub('', x))

        if self.text:
            ser = ser.str.replace(self.text, "")

        if self.stopwords:
            ser = ser.apply(lambda x: x if type(x) != str else " ".join(
                [i for i in x.split() if i not in stopwords]))

        return ser


class ChangeCase(XBaseTransformer):
    """ Changes the case of a string.

    Args:
        case (str): 'upper' or 'lower'
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, case='lower'):
        super().__init__()
        self.case = case

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        
        def _set_params(case = ["lower", "upper"]):
            self.case = case
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Changes the case of a string.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        if self.case == 'lower':
            return ser.apply(
                lambda x: str(x).lower() if type(x) in [str, bool] else x)

        if self.case == 'upper':
            return ser.apply(
                lambda x: str(x).upper() if type(x) in [str, bool] else x)

        else:
            raise ValueError("case change must be either 'lower' or 'upper'")


class DetectCategories(XBaseTransformer):
    """ Auto-detects categories from a string column.

    Args:
        max_categories (int): The maximum number of categories to extract.
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, max_categories=10, category_list=[]):
        super().__init__()
        self.max_categories = int(max_categories)
        self.category_list = category_list

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        
        def _set_params(max_categories=(2, 50)):
            self.max_categories = max_categories
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Detects categories from a string column.

        Args:
            ser (pd.Series): The series to transform.

        Raises:
            TypeError: If the series is not of type string.

        Returns:
            pd.Series: The transformed series.
        """

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

    def fit(self, ser: pd.Series) -> 'DetectCategories':
        """ Identifies the top categories from a text series.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Returns:
            DetectCategories: The fitted transformer.
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
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, pct=0.8, categories=[]):
        super().__init__()
        self.pct = pct
        self.categories = categories

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        
        def _set_params(pct=(0, 100)):
            self.pct = pct / 100
        
        return interactive(_set_params)

    def fit(self, ser: pd.Series) -> 'Condense':
        """ Determines the categories that make up x pct of obserations.

        Args:
            ser (pandas.Series): The series in which to analyse.

        Raises:
            TypeError: If the series is not of type string.

        Returns:
            Condense: The fitted transformer.
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

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Condenses a feature into categories that make up x pct of obserations.

        Args:
            ser (pd.Series): The series to transform.

        Raises:
            ValueError: If the series is not of type string.
        
        Returns:
            pd.Series: The transformed series.
        """
        if not is_string_dtype(ser):
            raise ValueError(f'Cannot condense categories for non-text field.')

        # Convert non-top categories to 'other
        return (ser.isin(self.categories) * ser).replace("", 'other')


class MergeCategories(XBaseTransformer):
    """ Merges specified categories in a series into one category.

    Args:
        merge_from (list): List of categories to merge from.
        merge_to (str): The category to merge to.
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, merge_from=[], merge_to=''):
        super().__init__()
        self.merge_from = merge_from
        self.merge_to = merge_to

    def __call__(self, column, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets

        unq = column.dropna().unique()

        def _set_params(
                merge_from=widgets.SelectMultiple(options=unq), merge_to=unq):
            self.merge_from = list(merge_from)
            self.merge_to = merge_to
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Merges specified categories in a series into one category.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        # Apply merging
        return ser.apply(lambda x: self.merge_to if x in self.merge_from else x)


class ReplaceCategory(XBaseTransformer):
    """ Replaces a category in a series with specified value.

    Args:
        target: The target value to replace.
        replace_with: The value to insert in place.
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, target=None, replace_with=''):
        super().__init__()
        self.target = target
        self.replace_with = replace_with

    def __call__(self, column, *args, **kwargs):
        from ipywidgets import interactive
        
        unq = column.dropna().unique()

        def _set_params(target=unq, replace_with=''):
            self.target = target
            self.replace_with = replace_with
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Replaces a category in a series with specified value.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        # Apply value replacement
        return ser.replace(self.target, self.replace_with)


class FillMissingCategorical(XBaseTransformer):
    """ Fills missing values with a specified value.

    Args:
        fill_with (str): Text to fill with.
    """
    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, fill_with='missing'):
        super().__init__()
        self.fill_with = fill_with

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        
        def _set_params(fill_with = "missing"):
            self.fill_with = fill_with
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Fills missing values with a specified value.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        
        """
        # Converts "" into np.nan to be filled
        ser = ser.apply(lambda x: np.nan if str(x).strip() == "" else x)

        return ser.fillna(self.fill_with)


class MapCategories(XBaseTransformer):
    """ Maps all categories of a string column to new values"""

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, category_values={}):
        super().__init__()
        self.category_values = category_values

    def __call__(self, ser, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
      
        def _set_params(*args, **kwargs):
            args = locals()
            self.category_values = dict(args)['kwargs']

        category_values = {i: widgets.Text(i) for i in ser.dropna().unique()}  

        return interactive(_set_params, **category_values)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Maps all categories of a string column to new values

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        return ser.map(self.category_values)


class TextContains(XBaseTransformer):
    """ Flags series values that contain, start with, or end with a value.

    Args:
        selector (str): The type of search to make.
        value (str): The value to search.

    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, selector=None, value=None):

        self.selector = selector
        self.value = value

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        def _set_params(
            selector = widgets.Dropdown(
                options=["starts with", "ends with", "contains"]),
            value = ''):

            self.selector = selector
            self.value = value

        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Flags series values that contain, start with, or end with a value.

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """

        if self.selector == "starts with":
            return ser.str.startswith(self.value)

        if self.selector == "ends with":
            return ser.str.endswith(self.value)

        else:
            return ser.str.contains(self.value)


class TextTrim(XBaseTransformer):
    """ Drops or keeps first/last n characters of a categorical column.

    Args:
        selector (str): [first, last].
        n (int): Number of characters to identify.
        action (str): [keep, drop] the identified characters.
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, selector=None, n=0, action='keep'):

        self.selector = selector
        self.n = n
        self.action = action

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        def _set_params(
            selector = widgets.Dropdown(options=["first", "last"]),
            n = widgets.IntText(n=1),
            action = widgets.Dropdown(options=['keep', 'drop']),
        ):
            self.selector = selector
            self.n = n
            self.action = action

        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Drops or keeps first/last n characters of a categorical column.

        Args:
            ser (pd.Series): The series to transform.
        
        Returns:
            pd.Series: The transformed series.
        """

        if self.action == 'keep':
            if self.selector == "first":
                return ser.str[:self.n]

            else:
                return ser.str[-self.n:]
        else:
            if self.selector == "first":
                return ser.str[self.n:]

            else:
                return ser.str[:-self.n]


class TextSlice(XBaseTransformer):
    """ Selects slice from categorical column string.

    Args:
        start (int): Starting character.
        end (int): Ending character.
        action (str): [keep, drop] selected slice.
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, start=None, end=None, action='keep'):

        self.start = start
        self.end = end
        self.action = action

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        import ipywidgets as widgets
        
        def _set_params(
            start = widgets.IntText(idx=0),
            end = widgets.IntText(idx=0),
            action = widgets.Dropdown(options=['keep', 'drop'])
        ):
            self.start = start
            self.end = end
            self.action = action

        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Selects slice from categorical column string.

        Args:
            ser (pd.Series): The series to transform.
        
        Returns:
            pd.Series: The transformed series.
        """

        if self.action == 'keep':
            return ser.str[self.start:self.end]
        else:
            return ser.str[:self.start] + ser.str[self.end:]


class ReplaceWith(XBaseTransformer):
    """ Replaces specified value in series

    Args:
        case (str): 'upper' or 'lower'

    Attributes:
        case (str): The case the string will convert to.
    """

    # Informs the embedded GUI which data types this transformer supports.
    supported_types = ['categorical']

    def __init__(self, target=None, replace_with=None):
        super().__init__()
        self.target = target
        self.replace_with = replace_with

    def __call__(self, *args, **kwargs):
        from ipywidgets import interactive
        
        def _set_params(target = '', replace_with = ''):
            self.target = target
            self.replace_with = replace_with
        
        return interactive(_set_params)

    def transform(self, ser: pd.Series) -> pd.Series:
        """ Replaces specified value in series

        Args:
            ser (pd.Series): The series to transform.

        Returns:
            pd.Series: The transformed series.
        """
        
        return ser.str.replace(self.target, self.replace_with)
    
""" Copyright Xplainable Pty Ltd, 2023"""

from pandas.api.types import (is_string_dtype, is_datetime64_dtype,
                              is_numeric_dtype, is_bool_dtype)
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


#Decorator to catch zero division errors
def catch_zero_division(return_value=0):
    """
    A decorator that catches division by zero errors in methods and returns a predefined value.

    Args:
        return_value: The value to return if a ZeroDivisionError is caught. Defaults to np.nan.

    Returns:
        The wrapped function's return value, or `return_value` if a ZeroDivisionError is caught.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ZeroDivisionError:
                return return_value
        return wrapper
    return decorator


class XScan:
    """ Scans a dataframe and returns a profile of each feature."""

    def __init__(self):
        self.profile = {}
        self.target = None

    @staticmethod
    @catch_zero_division(return_value=0)
    def _cardinality(ser):
        """ Measures the cardinality of a feature.

        Args:
            ser (Pandas.Series): Pandas Series containing a single feature.

        Returns:
            float: The cardinality score
        """

        # Drop NA for cardinality calculation
        ser = ser.dropna()

        # If the series is empty after dropping NA, return 0 or an appropriate value
        if len(ser) == 0:
            return 0.0

        return round(ser.nunique() / len(ser), 4)

    @staticmethod
    @catch_zero_division(return_value=0)
    def _is_mixed_case(ser):
        """ Percentage of a text feature that contains mixed cases.

        Args:
            ser (pd.Series): The series to be analysed.

        Returns:
            bool: True if contains mixed cases else False.
        """

        def _is_mixed(s):
            """ Checks if a string contains both numbers and letters.

            Args:
                s (str): The string to be checked.

            Returns:
                bool: True if the string is mixed else False.
            """

            is_mixed = any(char.isupper() for char in str(s).strip()) and \
                not all(char.isupper() for char in str(s).strip())

            return is_mixed

        # Check if cases are mixed
        ser = ser.copy().dropna()

        return ser.apply(_is_mixed).mean()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _mixed_char_num(ser):
        """ Percentage of a text feature that contains numbers and letters.

        Args:
            ser (pd.Series): The series to be analysed.

        Returns:
            float: Percentage of feature that contains numbers and letters.
        """

        def _is_mixed(s):
            """ Checks if a string contains both numbers and letters.

            Args:
                s (str): The string to be checked.

            Returns:
                bool: True if the string is mixed else False.
            """

            is_mixed = any(char.isdigit() for char in str(s).strip()) and \
                not all(char.isdigit() for char in str(s).strip())

            return is_mixed
        
        ser = ser.copy().dropna()

        return ser.apply(_is_mixed).mean()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _skewness_score(ser):
        """ Calculates a numeric feature's skewness.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The skewness score.
        """

        return ser.skew()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _variance_score(ser):
        """ Calculates a numeric feature's variance.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The skewness score.
        """

        return ser.var()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _kurtosis_score(ser):
        """ Calculates a numeric feature's kurtosis.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The kurtosis score.
        """

        return ser.kurtosis()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _unique_values(ser):
        """ Counts unique values of feature.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: Number of unique values.
        """

        return ser.nunique()

    @staticmethod
    @catch_zero_division(return_value=0)
    def _is_missing(ser):
        """ Calculates the pct of missing values

        Args:
            ser (pd.Series): Pandas Series containing the feature.

        Returns:
            float: pct of missing values
        """
        return ser.isna().sum() / len(ser)

    @staticmethod
    @catch_zero_division(return_value=0)
    def _category_imbalance(ser):
        """ Scores a categorical feature's category imbalance.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The imbalance score.
        """

        # attain the pct distribution of each category
        c = ser.value_counts() / len(ser)

        # Calculate the pct of each cat if perfectly imbalanced
        p = 1 / len(c)

        # Get the mean absolute difference from p
        mad = (abs(c - p) / p).mean()

        return mad

    def _detect_type(self, ser):
        """ Detects the machine learning type of a feature.

        Args:
            ser (Pandas.Series): Pandas Series containing a single feature.

        Returns:
            str: The feature type.
        """

        cardinality = self._cardinality(ser)
        missing_pct = self._is_missing(ser)

        if is_numeric_dtype(ser):

            if cardinality == 1 and missing_pct == 0:
                return 'id'

            else:
                return 'numeric'

        elif is_string_dtype(ser):
            if cardinality == 1:

                if not any(ser.str.contains(" ")):
                    return 'id'

                else:
                    return 'nlp'

            elif cardinality > 0.9:
                return 'nlp'

            else:
                return 'categorical'

        elif is_datetime64_dtype(ser):
            return 'date'

        else:
            return 'categorical'

    def _series_is_empty(self, ser):
        if len(ser) == 0:
            return True
        if ser.isna().sum() / len(ser) == 1:
            return True
        else:
            return False

    def _scan_numeric_feature(self, ser):

        if self._series_is_empty(ser):
            return {
                'type': 'empty',
                'missing_pct': 1
                }

        ser_type = self._detect_type(ser)
        if ser_type == 'id':
            return {
                'type': ser_type,
                'missing_pct': 0
                }

        sub_profile = {
            'type': ser_type,
            'missing_pct': self._is_missing(ser),
            'min': np.nanmin(ser),
            '25%': ser.quantile(0.25),
            'mean': np.nanmean(ser),
            'median': np.nanmedian(ser),
            '75%': ser.quantile(0.75),
            'max': np.nanmax(ser),
            'std': ser.std(),
            'skewness': self._skewness_score(ser),
            'kurtosis': self._kurtosis_score(ser),
            'variance': self._variance_score(ser)
        }

        return sub_profile

    def _scan_nlp_feature(self, ser):

        sub_profile = {
            'type': 'nlp',
            'missing_pct': self._is_missing(ser),
            'mixed_case': self._is_mixed_case(ser),
            'mixed_type': self._mixed_char_num(ser)
        }

        return sub_profile

    def _scan_categorical_feature(self, ser):

        if self._series_is_empty(ser):
            return {
                'type': 'empty',
                'missing_pct': 1
                }

        ser_type = self._detect_type(ser)
        if ser_type == 'id':
            return {
                'type': ser_type,
                'missing_pct': 0
                }

        elif ser_type == 'nlp':
            return self._scan_nlp_feature(ser)

        sub_profile = {
            'type': ser_type,
            'missing_pct': self._is_missing(ser),
            'nunique': self._unique_values(ser),
            'mode': ser.mode().values[0] if not ser.mode().empty else np.nan,
            'cardinality': self._cardinality(ser),
            'mixed_case': self._is_mixed_case(ser),
            'mixed_type': self._mixed_char_num(ser),
            'category_imbalance': self._category_imbalance(ser)
        }

        return sub_profile

    def _scan_date_feature(self, ser):

        if self._series_is_empty(ser):
            return {
                'type': 'empty',
                'missing_pct': 1
                }

        sub_profile = {
            'type': self._detect_type(ser),
            'missing_pct': self._is_missing(ser)
        }

        return sub_profile

    def _scan_feature(self, ser):

        # Check if series is empty or all NA first
        if self._series_is_empty(ser):
            return {
                'type': 'empty',
                'missing_pct': 1
            }

        if is_bool_dtype(ser):
            ser = ser.astype(str)
            sub_profile = self._scan_categorical_feature(ser)

        elif is_numeric_dtype(ser):
                sub_profile = self._scan_numeric_feature(ser)
                
        elif is_string_dtype(ser):
            sub_profile = self._scan_categorical_feature(ser)

        elif is_datetime64_dtype(ser):
            sub_profile = self._scan_date_feature(ser)

        else:
            # Handle unknown types by treating them as categorical
            # This includes cases where all values are NA in pandas 2.x
            sub_profile = self._scan_categorical_feature(ser)

        return sub_profile

    def scan(self, df, target=None):
        if target:
            if target not in df.columns:
                raise ValueError(f"{target} not in df")
            df = df.drop(columns=[target])

        # Iterate over each column and check for non-NaN existence before scanning
        for col in tqdm(list(df.columns)):
            if df[col].isna().all():
                print(f"Skipping column {col} because it is completely NaN.")
                continue  # Skip this column as it's entirely NaN

            # Proceed with scanning non-NaN columns
            self.profile[col] = self._scan_feature(df[col])

from pandas.api.types import is_string_dtype, is_datetime64_dtype, is_numeric_dtype
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
import copy

class QScan:
    """ Data quality scanner
    """

    def __init__(self):
        self.profile = {}
        self.target = None

    @staticmethod
    def _cardinality(ser):
        """ Measures the cardinality of a feature.

        Args:
            ser (Pandas.Series): Pandas Series containing a single feature.

        Returns:
            float: The cardinality score
        """

        # drop na for cardinality calculation
        ser = ser.dropna()

        return round(ser.nunique() / len(ser), 4)

    @staticmethod
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

            is_mixed = any(char.isupper() for char in str(s)) and \
                not all(char.isupper() for char in str(s))

            return is_mixed

        # Check if cases are mixed
        ser = ser.copy().dropna()

        return ser.apply(_is_mixed).mean()

    @staticmethod
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

            is_mixed = any(char.isdigit() for char in str(s)) and \
                not all(char.isdigit() for char in str(s))

            return is_mixed
        
        ser = ser.copy().dropna()

        return ser.apply(_is_mixed).mean()

    @staticmethod
    def _skewness_score(ser):
        """ Calculates a numeric feature's skewness.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The skewness score.
        """

        return ser.skew()

    @staticmethod
    def _variance_score(ser):
        """ Calculates a numeric feature's variance.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The skewness score.
        """

        return ser.var()

    @staticmethod
    def _kurtosis_score(ser):
        """ Calculates a numeric feature's kurtosis.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: The kurtosis score.
        """

        return ser.kurtosis()

    @staticmethod
    def _unique_values(ser):
        """ Counts unique values of feature.

        Args:
            ser (Pandas.Series): Pandas Series containing the feature.

        Returns:
            float: Number of unique values.
        """

        return ser.nunique()

    @staticmethod
    def _is_missing(ser):
        """ Calculates the pct of missing values

        Args:
            ser (pd.Series): Pandas Series containing the feature.

        Returns:
            float: pct of missing values
        """
        return ser.isna().sum() / len(ser)

    @staticmethod
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

    def _scan_numeric_feature(self, ser):

        ser_type = self._detect_type(ser)
        if ser_type == 'id':
            return {'type': ser_type}

        sub_profile = {
            'type': ser_type,
            'skewness': self._skewness_score(ser),
            'kurtosis': self._kurtosis_score(ser),
            'variance': self._variance_score(ser),
            'missing_pct': self._is_missing(ser)
        }

        return sub_profile

    def _scan_categorical_feature(self, ser):

        ser_type = self._detect_type(ser)
        if ser_type == 'id':
            return {'type': ser_type}

        sub_profile = {
            'type': ser_type,
            'nunique': self._unique_values(ser),
            'cardinality': self._cardinality(ser),
            'mixed_case': self._is_mixed_case(ser),
            'mixed_type': self._mixed_char_num(ser),
            'missing_pct': self._is_missing(ser),
            'category_imbalance': self._category_imbalance(ser)
        }

        return sub_profile

    def _scan_date_feature(self, ser):

        sub_profile = {
            'type': self._detect_type(ser),
            'missing_pct': self._is_missing(ser)
        }

        return sub_profile

    @ staticmethod
    def _vif(X, max_categories: int = 10):

        X = X.copy()
        X.dropna(inplace=True)
        
        categorical_columns = X.select_dtypes(
            include=['object', 'category']).columns.to_list()

        for col in list(categorical_columns):
            if X[col].nunique() > max_categories:
                categorical_columns.remove(col)

        X = pd.get_dummies(
            X, columns=categorical_columns, drop_first=True)

        X = X.select_dtypes(include=[np.number])

        # Add constant for vif calculation
        X = add_constant(X)

        # calculating VIF for each feature
        vif = {X.columns[i]: variance_inflation_factor(
            X.values, i) for i in range(X.shape[1])}

        vif.pop('const')

        vif_report = {}

        # Add categories to map
        for col in categorical_columns:
            sub = {}
            for i, v in copy.deepcopy(vif).items():
                if i.startswith(col):
                    sub[i.replace(f'{col}_', "")] = v
                    vif.pop(i)
            
            vif_report[col] = sub
            
        vif_report.update(vif)

        return vif_report

    def scan(self, df, target=None):
        
        for col in df.columns:
            ser = df[col]
            if is_numeric_dtype(ser):
                sub_profile = self._scan_numeric_feature(ser)
                
            elif is_string_dtype(ser):
                sub_profile = self._scan_categorical_feature(ser)

            elif is_datetime64_dtype(ser):
                sub_profile = self._scan_date_feature(ser)

            else:
                raise TypeError(f"Feature {col} has unknown type")

            self.profile[col] = sub_profile

        if target:
            # Add variance inflation factor 
            pass

        return self
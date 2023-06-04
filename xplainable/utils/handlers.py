import numpy as np
import pandas as pd
import warnings
from packaging import version


def add_thousands_separator(var):
    if isinstance(var, (float)):
        return '{:,.2f}'.format(var)
    elif isinstance(var, (int)):
        return '{:,.0f}'.format(var)
    else:
        return var
    
def _check_nans(df):
    return not df.isnull().values.any()

def _check_inf(df):
    numeric_df = df.select_dtypes(include=[np.number])
    inf_columns = numeric_df.apply(lambda x: np.isinf(x).any())
    return not inf_columns.any()

def check_df(df):
    assert _check_nans(df), \
        'Dataframe contains NaNs. Please remove them before proceeding.'
    assert _check_inf(df), \
        'Dataframe contains infinite values. Please remove them before proceeding.'
    
def check_critical_versions():
    """ This is implemented to ensure critical dependencies are imported."""
    
    # Tornado
    try:
        import tornado
    except ImportError:
        warnings.warn(
            "Tornado is not installed, but is required for this package.")
        return

    if version.parse(tornado.version) > version.parse('6.1'):
        warnings.warn(
            """
            Your version of Tornado is greater than 6.1, which is known to
            crash the Jupyter kernel when training models. Please consider
            downgrading to Tornado 6.1
            """)
    
    return
    
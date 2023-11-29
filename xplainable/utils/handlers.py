import numpy as np
import pandas as pd


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
    assert _check_inf(df), \
        'Dataframe contains infinite values. Please remove them before proceeding.'
        
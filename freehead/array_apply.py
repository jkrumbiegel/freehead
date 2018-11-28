import pandas as pd
import numpy as np
from collections import OrderedDict


def array_apply(df: pd.DataFrame, func, add_inplace=False):
    """
    Allows df.apply but with a numpy array as the result, which then stays an array and isn't auto expanded wrongly into
    a series by pandas. Useful if you deal with multiple data arrays for each trial of an experiment.

    :param df: A pandas DataFrame
    :param func: A function to apply, or an OrderedDict of functions to apply
    :param add_to_df:
    :return:
    """
    if not add_inplace:
        if callable(func):
            result = df.apply(lambda r: arr_series(func(r)), axis=1)

        elif isinstance(func, OrderedDict) or isinstance(func, dict):
            result = pd.DataFrame(index=df.index)
            for name, f in func.items():
                result[name] = df.apply(lambda r: arr_series(f(r)), axis=1)
        else:
            Exception('func needs to be a callable or a dict with callables')
        return result
    else:
        if isinstance(func, OrderedDict):
            for name, f in func.items():
                # directly add new series to given dataframe
                df[name] = df.apply(lambda r: arr_series(f(r)), axis=1)
        else:
            Exception('func needs to be an OrderedDict with callables')
        return None


def arr_series(arrs, name=None):
    # wraps a single array in a list so the auto expansion doesn't start
    if isinstance(arrs, np.ndarray):
        return pd.Series([arrs], name=name)

    elif isinstance(arrs, list):
        return pd.Series(arrs, name=name)

    else:
        # value could be an int, or float, or string etc, let's wrap it in a list too
        return pd.Series([arrs])

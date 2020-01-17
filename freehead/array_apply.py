import pandas as pd
import numpy as np
from collections import OrderedDict


def array_apply(df: pd.DataFrame, func, add_inplace=False, print_log=False):
    """
    Allows df.apply but with a numpy array as the result, which then stays an array and isn't auto expanded wrongly into
    a series by pandas. Useful if you deal with multiple data arrays for each trial of an experiment.

    :param df: A pandas DataFrame
    :param func: A function to apply, or an OrderedDict of functions to apply
    :param add_inplace:
    :return:
    """
    if not add_inplace:
        if callable(func):
            result = df.apply(lambda r: arr_series(func(r)), axis=1)

        elif isinstance(func, OrderedDict) or isinstance(func, dict):
            result = pd.DataFrame(index=df.index)

            n_funcs = len(func)
            for i, (name, f) in enumerate(func.items()):
                if print_log:
                    print(f'Computing "{name}" ({i + 1} of {n_funcs})...')
                result[name] = df.apply(lambda r: arr_series(f(r)), axis=1)
        else:
            Exception('func needs to be a callable or a dict with callables')
        return result

    else:
        if isinstance(func, OrderedDict):

            n_funcs = len(func)
            for i, (name, f) in enumerate(func.items()):
                # directly add new series to given dataframe
                if isinstance(name, str) or len(name) == 1:
                    operationtype = 'row'

                elif len(name) == 2:
                    operationtype, name = name

                else:
                    raise Exception(f'Too many elements in tuple {name}. Max 2.')

                if print_log:
                    print(f'Computing "{name}" ({i + 1} of {n_funcs})...')
                if operationtype == 'df':
                    df[name] = f(df)
                elif operationtype == 'row':
                    df[name] = df.apply(lambda r: arr_series(f(r)), axis=1)
                else:
                    raise Exception(f'Unknown operation type {operationtype} for {name}.')
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

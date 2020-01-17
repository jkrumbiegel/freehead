import pandas as pd
import numpy as np
from collections import OrderedDict


def expand_array_column(column, summed_array_lengths):
    concatenated = np.concatenate([cell for cell in column], axis=0)
    if concatenated.ndim > 1:
        flattened = concatenated.reshape(summed_array_lengths, -1)
        return tuple(flattened[:, i] for i in range(flattened.shape[1]))
    else:
        return concatenated


def expand_scalar_column(column, array_lengths):
    return np.repeat(column, array_lengths)


def expand_array_df(df, columns):

    affected_columns = [c if isinstance(c, str) else c[0] for c in columns]

    affected_df = df[affected_columns]

    # find a cell with a numpy array in the first row
    for i, cell in enumerate(affected_df.iloc[0, :]):
        if isinstance(cell, np.ndarray):
            reference_column = i
            break
    else:
        raise Exception('No affected column contains numpy arrays to expand.')

    array_lengths = [len(a) for a in affected_df.iloc[:, reference_column]]
    sum_lengths = np.sum(array_lengths)

    new_columns = OrderedDict()

    for c in columns:
        colname = c if isinstance(c, str) else c[0]
        if isinstance(affected_df[colname].iloc[0], np.ndarray):
            expanded = expand_array_column(affected_df[colname], sum_lengths)
        else:
            expanded = expand_scalar_column(affected_df[colname], array_lengths)

        # expanded array was multidimensional
        if isinstance(expanded, tuple):
            # print('multi')
            # print('colname: ', colname)

            new_colnames = [colname + f'_{i}' for i in range(len(expanded))] if isinstance(c, str) else c[1]
            for name, column in zip(new_colnames, expanded):
                if name != '_':
                    # print('new colname:', name)
                    # print('colshape:', column.shape)
                    new_columns[name] = column

        # expanded array was not multidimensional
        else:
            # print('not multi')
            new_colname = colname if isinstance(c, str) else c[1]
            # print('new colname:', new_colname)
            # print('colshape:', expanded.shape)
            new_columns[new_colname] = expanded

    return pd.DataFrame(new_columns)


# a = [np.zeros(10), np.zeros(7)]
# b = ['a', 'b']
# c = [np.ones((10, 2)), np.ones((7, 2))]
#
# df = pd.DataFrame(data={'a': a, 'b': b, 'c': c, 'd': c})
#
# print(expand_array_df(df, ['a', ('b', 'bbb'), 'c', ('d', ('c1', 'c2'))]))
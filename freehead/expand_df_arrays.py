import pandas as pd
import numpy as np
from collections import OrderedDict


def expand_df_arrays(df, array_lambdas, scalar_lambdas, index_columns):
    """Function to bring dataframes with arrays in cells into long form

    It is assumed that all resulting arrays from one row of the dataframe have the same length of the first dimension.
    Usually, this will be the case because all arrays contain data from the same time stamps, like 1000 samples of one
    trial in the first row, 1500 samples of the next trial in the second row, and so on.

    Args:
        :param df: A pandas dataframe with numpy arrays in some of the cells
        :param array_lambdas: A dict of named lambda functions that operate on arrays from the dataframe rows
        :param scalar_lambdas: A dict of named lambda functions that operate on scalars from the dataframe rows
        :param index_columns: A name or list of names of the columns from the expanded frame that will be the new index.
    Returns:
        expanded_df: A dataframe with arrays resulting from array lambdas expanded out into one row per array step.

    Example:
        import numpy as np
        import pandas as pd
        df = pd.DataFrame({
            'trial': [1, 2, 3],
            'vectors': [np.random.rand(10, 3), np.random.rand(20, 3), np.random.rand(30, 3)],
            'rotations': [np.random.rand(10, 3, 3), np.random.rand(20, 3, 3), np.random.rand(30, 3, 3)],
            'timestamps': [np.arange(10), np.arange(10, 30), np.arange(30, 60)]
        })

        expanded_df = expand_df_arrays(
            df,
            {
                # expand result array with three columns into three named columns
                ('rotated_x', 'rotated_y', 'rotated_z'): lambda r: np.einsum('tij,tj->ti', r['rotations'], r['vectors']),
                'vector_mean': lambda r: np.mean(r['vectors'], axis=1),
                'time': lambda r: r['timestamps']
            },
            {
                # there is one trial number per row, this gets multiplied the required number of times
                'trial': lambda r: r['trial']
            },
            ['trial', 'time']
        )
        expanded_df
    """
    columns = OrderedDict()

    row_counts = None

    for name, lambd in array_lambdas.items():
        results = []

        for i_row, row in df.iterrows():
            result = lambd(row)
            if row_counts is not None and result.shape[0] != row_counts[i_row]:
                raise Exception(
                    f'Array resulting from row {i_row} in column {name} needs length {row_counts[i_row]}'
                    f'but has {result.shape[0]}.')
            results.append(result)

        if row_counts is None:
            # record row counts for first column
            # these are later used to repeat scalars as often as necessary for each array
            row_counts = pd.Series([result.shape[0] for result in results], index=df.index)

        result = np.concatenate(results)

        if len(result.shape) > 2:
            raise Exception('Lambda functions need to return one- or two-dimensional arrays.')

        elif len(result.shape) == 2:
            if not isinstance(name, tuple):
                raise Exception(
                    f'Return array has shape {result.shape}.'
                    f'There need to be {result.shape[1]} names but there is only one ({name})')
            if len(name) != result.shape[1]:
                raise Exception(f'Result array has {result.shape[1]} columns but only {len(name)} names were supplied.')
            for i, subname in enumerate(name):
                columns[subname] = result[:, i]

        elif len(result.shape) == 1:
            if not isinstance(name, str):
                raise Exception(f'Column name is not a string but {type(name)}')
            columns[name] = result

    for name, lambd in scalar_lambdas.items():
        results = []
        for i_row, row in df.iterrows():
            results.append(lambd(row))
        columns[name] = [result for (row_count, result) in zip(row_counts, results) for _ in range(row_count)]

    expanded_df = pd.DataFrame(columns).set_index(index_columns)
    return expanded_df

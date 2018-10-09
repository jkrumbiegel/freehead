import pandas as pd
import numpy as np
from collections import OrderedDict


def create_trial_frame(same_each_block, block_specific=None):

    def length_if_list(obj):
        return len(obj) if isinstance(obj, list) else 1

    same_each_block_sorted = sorted(same_each_block.items(), key=lambda kv: length_if_list(kv[1]), reverse=True)

    # make objects that aren't lists yet into lists
    listified = [(name, value if isinstance(value, list) else [value]) for (name, value) in same_each_block_sorted]

    lengths = np.array([len(value) for (name, value) in listified])
    multipliers = (np.prod(lengths) / lengths).astype(np.int)

    # multiply all lists as often as needed for combinatorics and make an ordered dict out of them
    columns = [(name, value * multiplier) for (multiplier, (name, value)) in zip(multipliers, listified)]

    if block_specific is None:
        block_number_column = ('block', np.zeros(len(columns[0][1]), dtype=np.int))
        df = pd.DataFrame(data=OrderedDict([block_number_column, *columns]))
        return df

    listified_b = [(name, value if isinstance(value, list) else [value]) for (name, value) in block_specific.items()]
    lengths_b = np.array([len(value) for (name, value) in listified_b])

    n_blocks = lengths_b[0]
    if not np.all(lengths_b == n_blocks):
        raise ValueError('Block specific lists need to be all of the same length')

    # multiply columns for blocks
    columns_multiplied = [(name, value * n_blocks) for (name, value) in columns]

    # make block columns
    length_single_block = len(columns[0][1])
    block_specific_columns = [
        (name, [v for v in value for _ in range(length_single_block)])
        for (name, value) in listified_b]
    block_number_column = ('block', [b for b in range(n_blocks) for _ in range(length_single_block)])

    # make one big list with all column name / value tuples
    all_columns = [block_number_column, *block_specific_columns, *columns_multiplied]

    df = pd.DataFrame(data=OrderedDict(all_columns))
    return df

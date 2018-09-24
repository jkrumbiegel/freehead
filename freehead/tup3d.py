import numpy as np


def tup3d(array: np.ndarray, flipyz=False):
    assert(array.ndim == 2 and array.shape[1] == 3)
    if flipyz:
        return array[:, 0], array[:, 2], array[:, 1]
    else:
        return array[:, 0], array[:, 1], array[:, 2]

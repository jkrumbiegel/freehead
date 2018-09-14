import numpy as np


def tup3d(array: np.ndarray):
    assert(array.ndim == 2 and array.shape[1] == 3)
    return array[:, 0], array[:, 1], array[:, 2]

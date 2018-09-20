import numpy as np


def anynan(array: np.ndarray):
    return np.any(np.isnan(array))
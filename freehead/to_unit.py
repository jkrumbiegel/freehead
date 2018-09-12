import numpy as np


def to_unit(array: np.ndarray):

    magnitude = np.sqrt(np.sum(array ** 2, axis=array.ndim - 1))
    unit = array / magnitude[..., None]
    return unit

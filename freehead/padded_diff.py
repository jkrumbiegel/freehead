import numpy as np


def padded_diff(arr, filler=np.nan, axis=0):
    diff = np.diff(arr, axis=axis)
    pad_shape = [s if i != axis else 1 for i, s in enumerate(diff.shape)]
    pad = np.full(pad_shape, filler, dtype=arr.dtype)
    return np.concatenate((pad, diff), axis=axis)
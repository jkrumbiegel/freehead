import numpy as np
from numba import jit


@jit(nopython=True)
def to_yawpitchroll_jit(R, in_degrees=True, eps=1e-16):
    R = R.reshape((-1, 3, 3))
    ypr = np.empty((R.shape[0], 3), dtype=np.float64)
    for i in range(R.shape[0]):
        if np.any(np.isnan(R[i, :, :])):
            ypr[i, :] = np.nan
        else:
            if np.abs(R[i, 2, 1] - 1) <= eps or np.abs(R[i, 2, 1] + 1) <= eps:
                ypr[i, 0] = 0.0
                ypr[i, 1] = np.arcsin(R[i, 2, 1])
                ypr[i, 2] = np.arctan2(R[i, 1, 0], R[i, 0, 0])
            else:
                p = np.arcsin(R[i, 2, 1])
                ypr[i, 1] = p
                ypr[i, 2] = -np.arctan2(R[i, 2, 0] / np.cos(p), R[i, 2, 2] / np.cos(p))
                ypr[i, 0] = -np.arctan2(R[i, 0, 1] / np.cos(p), R[i, 1, 1] / np.cos(p))

    return ypr if not in_degrees else np.rad2deg(ypr)


def to_yawpitchroll(R, in_degrees=True, eps=1e-16):
    # for some reason squeezing the result array in the numba function doesn't work
    return to_yawpitchroll_jit(R, in_degrees=in_degrees, eps=eps).squeeze()



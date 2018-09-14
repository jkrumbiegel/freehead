import numpy as np


def is_rotation_matrix(R):
    assert(R.shape == (3, 3))
    return np.allclose(R @ R.T, np.eye(3))
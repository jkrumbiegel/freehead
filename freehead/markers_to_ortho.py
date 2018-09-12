import numpy as np
import freehead


def markers_to_ortho(
        marker1: np.ndarray,
        marker2: np.ndarray,
        marker3: np.ndarray
):
    marker1 = marker1.squeeze()
    marker2 = marker2.squeeze()
    marker3 = marker3.squeeze()

    side1 = freehead.to_unit(marker2 - marker1)
    side2 = freehead.to_unit(marker3 - marker1)
    normal1 = freehead.to_unit(np.cross(side1, side2)) # should be unit length but let's be sure
    normal2 = freehead.to_unit(np.cross(side1, normal1))

    # these are all Nx3
    v1 = side1
    v2 = normal1
    v3 = normal2

    # single rotation matrix if there is only one rigidbody
    if len(v1.shape) == 1:
        V = np.empty((3, 3), np.float)

        V[:, 0] = v1
        V[:, 1] = v2
        V[:, 2] = v3

        return V

    else:
        length = v1.shape[0]

        # add the orthonormal basis vectors to the array of rotation matrices as column vectors
        V = np.empty((length, 3, 3), np.float)
        # V[marker, row, column]
        V[:, :, 0] = v1
        V[:, :, 1] = v2
        V[:, :, 2] = v3

        return V

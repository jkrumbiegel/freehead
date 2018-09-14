import numpy as np
import freehead
import warnings


def _to_yawpitchroll_single(R, in_degrees):
    if not freehead.is_rotation_matrix(R):
        if np.any(np.isnan(R)):
            warnings.warn('Rotation matrix contains nan, result replaced with nan')
        else:
            warnings.warn('Rotation matrix invalid, result replaced with nan')
        return np.array([np.nan, np.nan, np.nan])

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([yaw, pitch, roll]) if not in_degrees else np.rad2deg(np.array([yaw, pitch, roll]))


def to_yawpitchroll(R, in_degrees=True):
    if len(R.shape) == 2:
        return _to_yawpitchroll_single(R, in_degrees=in_degrees)

    if len(R.shape) == 3:
        euler = np.empty((R.shape[0], 3), np.float)
        for i in range(R.shape[0]):
            euler[i, :] = _to_yawpitchroll_single(R[i, ...].squeeze(),  in_degrees=in_degrees)

        return euler

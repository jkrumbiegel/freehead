import freehead
import numpy as np


def from_yawpitchroll(*args, in_degrees=True):

    if len(args) == 1:
        ypr = args[0]

    elif len(args) == 3:
        ypr = np.array([args[0], args[1], args[2]])

    else:
        raise Exception('Input argument has to be one array with three entries or three numbers.')

    if in_degrees:
        ypr = np.deg2rad(ypr)

    yaw_rotation = freehead.u_theta(np.array([0, 0, 1]), ypr[0])
    pitch_rotation = freehead.u_theta(np.array([0, 1, 0]), ypr[1])
    roll_rotation = freehead.u_theta(np.array([1, 0, 0]), ypr[2])

    return roll_rotation @ pitch_rotation @ yaw_rotation

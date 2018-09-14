from scipy.optimize import minimize, Bounds
import numpy as np
import freehead as fh


def calibrate_pupil(T_head_world, R_head_world, gaze_normals, T_target_world, ini_T_eye_head=np.zeros(3), bounds_mm=np.inf):

    def err_func(parameters):
        R_eye_head = fh.from_yawpitchroll(parameters[0:3])
        T_eye_head = parameters[3:6]

        T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
        eye_to_target = fh.to_unit(T_target_world - T_eye_world)

        gaze_normals_world = np.einsum(
            'tij,tj->ti',
            R_head_world,
            np.einsum('ij,tj->ti', R_eye_head, gaze_normals))

        angles = np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target))
        return np.sum(angles)

    ini_parameters = np.concatenate((np.zeros(3), ini_T_eye_head))

    bounds = Bounds(
        lb=np.array([-np.inf, -np.inf, -np.inf, ini_T_eye_head[0] - bounds_mm, ini_T_eye_head[1] - bounds_mm, ini_T_eye_head[2] - bounds_mm]),
        ub=np.array([np.inf, np.inf, np.inf, ini_T_eye_head[0] + bounds_mm, ini_T_eye_head[1] + bounds_mm, ini_T_eye_head[2] + bounds_mm])
    )

    return minimize(err_func, ini_parameters, bounds=bounds)

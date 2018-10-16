from scipy.optimize import minimize, Bounds, differential_evolution
import numpy as np
import freehead as fh


def calibrate_pupil(T_head_world, R_head_world, gaze_normals, T_target_world, ini_T_eye_head=np.zeros(3), bounds_mm=np.inf, global_opt=False):

    def err_func(parameters):
        R_eye_head = fh.from_yawpitchroll(parameters[0:3])
        T_eye_head = parameters[3:6]

        T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
        eye_to_target = fh.to_unit(T_target_world - T_eye_world)

        gaze_normals_world = np.einsum(
            'tij,tj->ti',
            R_head_world,
            np.einsum('ij,tj->ti', R_eye_head, gaze_normals))

        angles = np.rad2deg(np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target)))
        return np.mean(angles)

    ini_parameters = np.concatenate((np.zeros(3), ini_T_eye_head))

    if global_opt:
        lb = [-360, -180, -180, ini_T_eye_head[0] - bounds_mm, ini_T_eye_head[1] - bounds_mm, ini_T_eye_head[2] - bounds_mm]
        ub = [360, 180, 180, ini_T_eye_head[0] + bounds_mm, ini_T_eye_head[1] + bounds_mm, ini_T_eye_head[2] + bounds_mm]
        bounds = [(l, u) for (l, u) in zip(lb, ub)]
    else:
        bounds = Bounds(
            lb=np.array([-np.inf, -np.inf, -np.inf, ini_T_eye_head[0] - bounds_mm, ini_T_eye_head[1] - bounds_mm, ini_T_eye_head[2] - bounds_mm]),
            ub=np.array([np.inf, np.inf, np.inf, ini_T_eye_head[0] + bounds_mm, ini_T_eye_head[1] + bounds_mm, ini_T_eye_head[2] + bounds_mm])
        )

    if global_opt:
        return differential_evolution(err_func, bounds)
    else:
        return minimize(err_func, ini_parameters, bounds=bounds)


def calibrate_pupil_rotation(T_head_world, R_head_world, gaze_normals, T_target_world, T_eye_head):

    def err_func(ypr):
        R_eye_head = fh.from_yawpitchroll(ypr)

        T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
        eye_to_target = fh.to_unit(T_target_world - T_eye_world)

        gaze_normals_world = np.einsum(
            'tij,tj->ti',
            R_head_world,
            np.einsum('ij,tj->ti', R_eye_head, gaze_normals))

        angles = np.rad2deg(np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target)))
        return np.sum(angles)

    return minimize(err_func, np.zeros(3, dtype=np.float))


def calibrate_pupil_translation(T_head_world, R_head_world, gaze_normals, T_target_world, R_eye_head, T_eye_head_ini):

    def err_func(T_eye_head):

        T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
        eye_to_target = fh.to_unit(T_target_world - T_eye_world)

        gaze_normals_world = np.einsum(
            'tij,tj->ti',
            R_head_world,
            np.einsum('ij,tj->ti', R_eye_head, gaze_normals))

        angles = np.rad2deg(np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target)))
        return np.sum(angles)

    return minimize(err_func, T_eye_head_ini)


def calibrate_pupil_nonlinear(
        T_head_world,
        R_head_world,
        gaze_normals,
        T_target_world,
        ini_T_eye_head=np.zeros(3),
        ini_ypr=np.zeros(3),
        ini_polynom_params=np.array([0, 0, 1.0, 1.0, 0, 0]),
        leave_T_eye_head=False):

    if leave_T_eye_head:

        def err_func(params):
            T_eye_head = ini_T_eye_head
            R_eye_head = fh.from_yawpitchroll(params[0:3])
            aa = params[3:5]
            bb = params[5:7]
            cc = params[7:9]

            T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
            eye_to_target = fh.to_unit(T_target_world - T_eye_world)

            gaze_normals_head = np.einsum('ij,tj->ti', R_eye_head, gaze_normals)
            gaze_normals_head_transformed = fh.normals_nonlinear_angular_transform(gaze_normals_head, aa, bb, cc)

            gaze_normals_world = np.einsum(
                'tij,tj->ti',
                R_head_world,
                gaze_normals_head_transformed)

            angles = np.rad2deg(np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target)))
            return angles.mean()

        ini_params = np.concatenate((ini_ypr, ini_polynom_params))

    else:

        def err_func(params):

            R_eye_head = fh.from_yawpitchroll(params[0:3])
            aa = params[3:5]
            bb = params[5:7]
            cc = params[7:9]
            T_eye_head = params[9:12]

            T_eye_world = np.einsum('tij,j->ti', R_head_world, T_eye_head) + T_head_world
            eye_to_target = fh.to_unit(T_target_world - T_eye_world)

            gaze_normals_head = np.einsum('ij,tj->ti', R_eye_head, gaze_normals)
            gaze_normals_head_transformed = fh.normals_nonlinear_angular_transform(gaze_normals_head, aa, bb, cc)

            gaze_normals_world = np.einsum(
                'tij,tj->ti',
                R_head_world,
                gaze_normals_head_transformed)

            angles = np.rad2deg(np.arccos(np.einsum('ti,ti->t', gaze_normals_world, eye_to_target)))
            return angles.mean()

        ini_params = np.concatenate((ini_T_eye_head, ini_ypr, ini_polynom_params))

    return minimize(err_func, ini_params)

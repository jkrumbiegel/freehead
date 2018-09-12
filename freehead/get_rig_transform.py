from scipy.optimize import minimize
import numpy as np
import freehead


def get_rig_transform(probe_tips, indices, in_degrees=True):
    probed_leds = freehead.LED_POSITIONS[indices, :]

    def err_func(parameters):
        ypr = parameters[0:3]
        T_leds = parameters[3:6]
        R_leds = freehead.yawpitchroll(ypr, in_degrees=in_degrees)
        # apply inverse transform
        calculated_positions = np.einsum('ij,tj->ti', np.linalg.pinv(R_leds), probe_tips - T_leds)
        position_error = np.sum((calculated_positions - probed_leds) ** 2)
        return position_error

    initial_guess = np.zeros(6)

    return minimize(err_func, initial_guess)
from scipy.interpolate import interp1d
import numpy as np


def interpolate_a_onto_b_time(a, a_time, b_time, kind='linear'):
    interpolator = interp1d(a_time, a, axis=0, bounds_error=False, fill_value=np.nan, kind=kind)
    return [interpolator(b_time)]
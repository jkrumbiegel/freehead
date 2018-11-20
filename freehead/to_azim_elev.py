import numpy as np


def to_azim_elev(vectors):
    """
    Gives back azimuth (horizontal) and elevation (vertical) angle for vectors where +x is right, +y is forward and
    +z is up
    :param vectors: Vectors array, last dimension needs to have three components, x, y, z
    :return:
    """

    return np.arctan2(vectors[..., [0, 2]], vectors[..., 1, None])

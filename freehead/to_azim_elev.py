import numpy as np


def to_azim_elev(vectors):
    """
    Gives back azimuth (horizontal) and elevation (vertical) angle for vectors where +x is right, +y is forward and
    +z is up
    :param vectors: Vectors array, last dimension needs to have three components, x, y, z
    :return:
    """

    azimuth = np.arctan2(vectors[..., 1], vectors[..., 0])
    elevation = np.arctan2(vectors[..., 2], np.sqrt((vectors[..., 1] ** 2) + (vectors[..., 0] ** 2)))

    if vectors.ndim == 1:
        return np.array([azimuth, elevation])
    else:
        return np.stack((azimuth, elevation), axis=-1)

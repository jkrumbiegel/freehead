import numpy as np


def u_theta(u, theta):

    if not u.shape == (3,):
        raise ValueError('u is not a three element vector')

    if not np.isclose(np.sum(u ** 2), np.array([1])):
        raise ValueError('u is not a unit vector')

    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    return np.array(
        [
            [
                ctheta + u[0] * u[0] * (1 - ctheta),
                u[0] * u[1] * (1 - ctheta) - u[2] * stheta,
                u[0] * u[2] * (1 - ctheta) + u[1] * stheta],

            [
                u[1] * u[0] * (1 - ctheta) + u[2] * stheta,
                ctheta + u[1] * u[1] * (1 - ctheta),
                u[1] * u[2] * (1 - ctheta) - u[0] * stheta],

            [
                u[2] * u[0] * (1 - ctheta) - u[1] * stheta,
                u[2] * u[1] * (1 - ctheta) + u[0] * stheta,
                ctheta + u[2] * u[2] * (1 - ctheta)]
        ]
    )

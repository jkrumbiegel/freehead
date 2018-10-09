import numpy as np
import freehead as fh


def normals_nonlinear_angular_transform(normals, *polynom_params):

    if normals.ndim == 1:
        normals = normals[None, :]

    if len(polynom_params) == 1:
        aa = polynom_params[0:2]
        bb = polynom_params[2:4]
        cc = polynom_params[4:6]

    elif len(polynom_params) == 3:
        aa = polynom_params[0]
        bb = polynom_params[1]
        cc = polynom_params[2]

    else:
        raise Exception('Polynomial parameters must either be three arrays (aa, bb, cc) or one array (aabbcc).')

        # uses x right, y forward, z up assumption
    elev_azim = np.arctan2(normals[:, [0, 2]], normals[:, 1, None])
    elev_azim_transformed = aa * (elev_azim ** 2) + bb * elev_azim + cc

    xz_transformed = np.tan(elev_azim_transformed)
    xyz = np.empty(normals.shape, np.float)
    xyz[:, [0, 2]] = xz_transformed
    xyz[:, 1] = 1
    xyz_normalized = fh.to_unit(xyz).squeeze()

    return xyz_normalized

import freehead as fh
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import pickle
import os


probe_data_raw = loadmat('/Users/juliuskrumbiegel/Dropbox/Uni/Mind and Brain/Rolfslab Master/Test Data/2018_08_21/digitizer_calibration_spheres.mat')['optotrak_data']
probe_data = np.reshape(probe_data_raw, (-1, 4, 3))

i_start = 3330
i_end = 6000

probe = fh.Rigidbody(probe_data[i_start, :, :])

rotations, reference_points = probe.solve(probe_data[i_start: i_end, ...])

ini_tip_vector = np.array([0, -80, 0])

def err_func(vector):
    tip_positions = reference_points.squeeze() + np.einsum('nij,j->ni', rotations, vector)
    error = ((tip_positions - tip_positions.mean(axis=0)[None, :]) ** 2).sum()
    return error

result = minimize(err_func, ini_tip_vector)

tip_positions = reference_points.squeeze() + np.einsum('nij,j->ni', rotations, result.x)


fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')
ax.plot(*fh.tup3d(probe_data[i_start: i_end, 0, :].squeeze()))
ax.plot(*fh.tup3d(tip_positions))
plt.legend(['marker', 'tip'])

ax2 = fig.add_subplot(211, projection='3d')
ax2.plot(*fh.tup3d(tip_positions))
plt.legend(['tip'])

tip_position_ini = probe.ref_points + result.x


calibrated_probe = fh.Rigidbody(probe_data[i_start, :, :], ref_points=tip_position_ini)

rotations_calib, tips_calib = calibrated_probe.solve(probe_data)

#%%
fig2 = plt.figure()
ax3 = fig2.add_subplot(211, projection='3d')
ax3.plot(*fh.tup3d(probe_data[:, 0, :].squeeze()))
ax3.scatter(*fh.tup3d(tips_calib.squeeze()), alpha=0.01)
plt.legend(['marker', 'tip'])

#%%
probe_calibration = {
    'markers': probe_data[i_start, :, :],
    'ref_point': tip_position_ini
    }

folder_path = os.path.dirname(os.path.abspath(__file__))
pickle.dump(probe_calibration, open(folder_path + '/../datafiles/four_marker_probe_calibrated.pickle', 'wb'))

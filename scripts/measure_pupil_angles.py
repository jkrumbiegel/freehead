import freehead as fh
import numpy as np
import pygame
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize

pygame.init()

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

pthread = fh.PupilThread()
pthread.start()
pthread.started_running.wait()

probe = fh.FourMarkerProbe()

with open('screen_corners_triangulation_result.pickle', 'rb') as file:
    screen_corners = pickle.load(file)

try:
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

    while True:
        fh.wait_for_keypress(pygame.K_SPACE)
        r_probe, p_eye = probe.solve(othread.current_sample[15:27].reshape((4, 3)))
        if fh.anynan(r_probe):
            continue
        else:
            break
  
    xx, yy = np.meshgrid(np.linspace(0, screen.get_width(), 6, dtype=np.int), np.linspace(0, screen.get_height(), 6, dtype=np.int))
    
    pupil_samples = []
    
    for i in range(xx.size):
        print(i)
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 0, 0), (xx.flat[i], yy.flat[i]), 10, 0)
        pygame.display.update()
        fh.wait_for_keypress(pygame.K_SPACE)
        pupil_samples.append(pthread.current_sample.copy())
        
    pygame.quit()
    othread.should_stop.set()
    pthread.should_stop.set()
except:
    pygame.quit()
    othread.should_stop.set()
    pthread.should_stop.set()


#%%
def screen_to_space(coords):
    origin = screen_corners['upper left']
    
    right_vector = fh.to_unit(screen_corners['upper right'] - origin)
    down_vector = fh.to_unit(screen_corners['lower left'] - origin)
    
    width = np.linalg.norm(screen_corners['upper right'] - origin)
    height = np.linalg.norm(screen_corners['lower left'] - origin)
    
    return (coords / np.array([1920, 1080]) * np.array([width, height]) @ np.vstack((right_vector, down_vector))) + origin
    
points_on_screen = screen_to_space(np.vstack((xx.flat, yy.flat)).T)
eye_to_points = points_on_screen - p_eye

angles_from_eye = np.rad2deg(np.arctan2(eye_to_points[:, [0, 2]], eye_to_points[:, 1, None]))

plt.figure()
plt.scatter(angles_from_eye[:, 0], angles_from_eye[:, 1], c=np.arange(angles_from_eye.shape[0]))
plt.xlabel('azimuth')
plt.ylabel('elevation')

#%%
gaze_normals = np.vstack([p[3:6] for p in pupil_samples])

# find rotation matrix
def err_func(ypr):
    R = fh.from_yawpitchroll(ypr)
    gaze_rotated = (R @ gaze_normals.T).T
    azim_elev =  np.rad2deg(np.arctan2(gaze_rotated[:, [0, 2]], gaze_rotated[:, 1, None]))
    return ((angles_from_eye - azim_elev) ** 2).sum()

optim_result = minimize(err_func, np.zeros(3))
R_eye = fh.from_yawpitchroll(optim_result.x)

gaze_corrected = (R_eye @ gaze_normals.T).T
#%%

azim_elev = np.rad2deg(np.arctan2(gaze_corrected[:, [0, 2]], gaze_corrected[:, 1, None]))

#%%

def err_func_nonlinear(ypr_aa_bb_cc):
    ypr = ypr_aa_bb_cc[0:3]
    R = fh.from_yawpitchroll(ypr)
    gaze_rotated = (R @ gaze_normals.T).T
    azim_elev =  np.rad2deg(np.arctan2(gaze_rotated[:, [0, 2]], gaze_rotated[:, 1, None]))
    aa = ypr_aa_bb_cc[None, 3:5]
    bb = ypr_aa_bb_cc[None, 5:7]
    cc = ypr_aa_bb_cc[None, 7:9]
    azim_elev_nonlinear = aa * (azim_elev ** 2) + bb * azim_elev + cc
    return ((angles_from_eye - azim_elev_nonlinear) ** 2).sum()

optim_result_nonlinear = minimize(err_func_nonlinear, np.zeros(9))

R_eye_nonlinear = fh.from_yawpitchroll(optim_result_nonlinear.x)
aa = optim_result_nonlinear.x[3:5]
bb = optim_result_nonlinear.x[5:7]
cc = optim_result_nonlinear.x[7:9]

gaze_corrected_nonlinear = (R_eye_nonlinear @ gaze_normals.T).T
azim_elev_uncorrected_nonlinear = np.rad2deg(np.arctan2(gaze_corrected_nonlinear[:, [0, 2]], gaze_corrected_nonlinear[:, 1, None]))
azim_elev_nonlinear = aa * (azim_elev_uncorrected_nonlinear ** 2) + bb * azim_elev_uncorrected_nonlinear + cc

#%%

def err_func_nonlinear_direct(ypr_aa_bb_cc):
    ypr = ypr_aa_bb_cc[0:3]
    aa = ypr_aa_bb_cc[None, 3:5]
    bb = ypr_aa_bb_cc[None, 5:7]
    cc = ypr_aa_bb_cc[None, 7:9]
    R = fh.from_yawpitchroll(ypr)
    gaze_rotated = (R @ gaze_normals.T).T
    gaze_transformed = gaze_rotated.copy()
    gaze_transformed[:, [0, 2]] = aa * (gaze_transformed[:, [0, 2]] ** 2) + bb * gaze_transformed[:, [0, 2]] + cc
    gaze_transformed = fh.to_unit(gaze_transformed)
    azim_elev_nonlinear = np.rad2deg(np.arctan2(gaze_transformed[:, [0, 2]], gaze_transformed[:, 1, None]))
    return ((angles_from_eye - azim_elev_nonlinear) ** 2).sum()

optim_result_nonlinear_2 = minimize(err_func_nonlinear_direct, np.zeros(9))

R_eye_nonlinear_2  = fh.from_yawpitchroll(optim_result_nonlinear_2 .x)
aa_2  = optim_result_nonlinear_2 .x[3:5]
bb_2  = optim_result_nonlinear_2 .x[5:7]
cc_2  = optim_result_nonlinear_2 .x[7:9]

gaze_corrected_nonlinear_2 = (R_eye_nonlinear_2  @ gaze_normals.T).T
gaze_transformed_2 = gaze_corrected_nonlinear_2.copy()
gaze_transformed_2[:, [0, 2]] = aa_2 * (gaze_transformed_2[:, [0, 2]] ** 2) + bb_2 * gaze_transformed_2[:, [0, 2]] + cc_2
gaze_transformed_2 = fh.to_unit(gaze_transformed_2)
azim_elev_nonlinear_2 = np.rad2deg(np.arctan2(gaze_transformed_2[:, [0, 2]], gaze_transformed_2[:, 1, None]))

#%%


plt.figure()
plt.scatter(angles_from_eye[:, 0], angles_from_eye[:, 1], c=np.arange(angles_from_eye.shape[0]), cmap='winter', label='real')
plt.scatter(azim_elev[:, 0], azim_elev[:, 1], marker='x', c=np.arange(azim_elev.shape[0]), cmap='autumn', label='pupil')
plt.scatter(azim_elev_nonlinear[:, 0], azim_elev_nonlinear[:, 1], marker='s', c=np.arange(azim_elev.shape[0]), cmap='cool', label='pupil nonlinear')
plt.scatter(azim_elev_nonlinear_2[:, 0], azim_elev_nonlinear_2[:, 1], marker='+', c=np.arange(azim_elev.shape[0]), cmap='cool', label='pupil nonlinear 2')

plt.xlabel('azimuth')
plt.ylabel('elevation')
plt.legend()
plt.show()
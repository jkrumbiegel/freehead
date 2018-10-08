import freehead as fh
import numpy as np
import pygame
import os
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt
import vispy.scene
from vispy.scene import visuals
from scipy.interpolate import interp1d

pygame.init()
pygame.display.set_mode((300, 300))

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

pthread = fh.PupilThread()
pthread.start()
pthread.started_running.wait()

athread = fh.ArduinoThread()
athread.start()
athread.started_running.wait()

#%%
probe = fh.FourMarkerProbe()

def extract_probe(ot_data):
    return ot_data[..., 15:27].reshape((-1, 4, 3)).squeeze()

def extract_helmet(ot_data):
    return ot_data[..., 3:15].reshape((-1, 4, 3)).squeeze()

def extract_gaze(p_data):
    return p_data[3:6]

if not 'rig_led_positions' in locals():
    n_rig_calib_samples = 5
    led_positions = np.full((n_rig_calib_samples, 3), np.nan)
    rig_calibration_indices = np.round(np.linspace(0, 254, n_rig_calib_samples)).astype(np.int)
    
    for i, ci in enumerate(rig_calibration_indices):
        athread.write_uint8(ci, 100, 0, 0)
        print('Press space to record led with index', ci)
        while True:
            fh.wait_for_keypress(pygame.K_SPACE)
            probe_rotation, probe_tip = probe.solve(extract_probe(othread.current_sample.copy()))
            if fh.anynan(probe_rotation):
                print('Probe not visible, try again.')
                continue
            led_positions[i, :] = probe_tip[:]
            break
          
    athread.write_uint8(255, 0, 0, 0)
    
    rig = fh.LedRig(rig_calibration_indices)
    _, rig_led_positions = rig.solve(led_positions)
else:
    print('rig_led_positions calibration already present. Delete if you want to recalibrate.')

#%%
head_measurement_points = [
        'head straight',
        'nasion',
        'inion',
        'right ear',
        'left ear',
        'right eye']

for i, measurement_point in enumerate(head_measurement_points):
    print('Press space to measure: ' + measurement_point)
    while True:
        fh.wait_for_keypress(pygame.K_SPACE)
        current_sample = othread.current_sample.copy()
        helmet_leds = extract_helmet(current_sample)

        if i == 0:
            if np.any(np.isnan(helmet_leds)):
                print('Helmet LEDs not all visible. Try again.')
                continue
        
            helmet = fh.Rigidbody(helmet_leds)
            break
        else:
            _, probe_tip = probe.solve(extract_probe(current_sample))
            if np.any(np.isnan(probe_tip)):
                print('Probe not visible. Try again.')
                continue
            helmet_rotation, _ = helmet.solve(helmet_leds)
            if fh.anynan(helmet_rotation):
                print('Helmet could not be solved. Try again.')
                continue
            helmet.add_reference_points(helmet_leds, probe_tip)
            break

right_eye = helmet.ref_points[5, :].copy()


#%% calibration of rotation using multiple points and no head movement
n_calibration_points = 9
calibration_indices = np.linspace(0, 254, n_calibration_points).astype(np.int)
R_head_world = np.full((n_calibration_points, 3, 3), np.nan)
T_head_world = np.full((n_calibration_points, 3), np.nan)
gaze_normals = np.full((n_calibration_points, 3), np.nan)
T_target_world = rig_led_positions[calibration_indices, :]

for i, c_index in enumerate(calibration_indices):
    athread.write_uint8(c_index, 100, 0, 0)
    print('Press space to record eye calibration point at led index ', c_index)
    while True: 
        fh.wait_for_keypress(pygame.K_SPACE)
        rotation, ref_points = helmet.solve(extract_helmet(othread.current_sample.copy()))
        if np.any(np.isnan(rotation)):
            print('NaN values in optotrak data, repeat')
            continue
        gaze_normal = extract_gaze(pthread.current_sample.copy())
        if np.any(np.isnan(gaze_normal)):
            print('NaN values in pupil data, repeat')
            continue
        break
    R_head_world[i, :, :] = rotation
    T_head_world[i, :] = ref_points[0, :]
    gaze_normals[i, :] = gaze_normal
    
athread.write_uint8(255, 0, 0, 0)

T_eye_head_ini = ((helmet.ref_points[5, :] - helmet.ref_points[0, :])
    + 15 * fh.to_unit(helmet.ref_points[2, :] - helmet.ref_points[1, :])) # add 15 mm of nasion to inion to measured eye position
        

calibration_result_rotation = fh.calibrate_pupil_rotation(
        T_head_world,
        R_head_world,
        gaze_normals,
        T_target_world,
        T_eye_head = T_eye_head_ini)

R_eye_head = fh.from_yawpitchroll(calibration_result_rotation.x)

#%% calibration of translation using single point technique
calibration_point = 127
duration = 15

athread.write_uint8(calibration_point, 100, 0, 0)
print('Press space to start eye calibration.')
fh.wait_for_keypress(pygame.K_SPACE)

othread.reset_data_buffer()
pthread.reset_data_buffer()

start_time = time.monotonic()
while time.monotonic() - start_time < duration:
    time.sleep(0.1)
    
athread.write_uint8(255, 0, 0, 0)

odata = othread.get_shortened_data().copy()
pdata = pthread.get_shortened_data().copy()
gaze_normals = pdata[:, 3:6]

f_interpolate = interp1d(odata[:, 30], odata[:, 3:15], axis=0)
odata_interpolated = f_interpolate(pdata[:, 2]).reshape((-1, 4, 3))

R_head_world, ref_points = helmet.solve(odata_interpolated)
T_head_world = ref_points[:, 0]

confidence_enough = pdata[:, 6] > 0.6
rotations_valid = ~np.any(np.isnan(R_head_world).reshape((-1, 9)), axis=1)
chosen_mask = np.logical_and(confidence_enough, rotations_valid)

T_target_world = np.tile(rig_led_positions[calibration_point, :], (chosen_mask.sum(), 1))
#%%

calibration_result_translation = fh.calibrate_pupil_translation(
        T_head_world[chosen_mask],
        R_head_world[chosen_mask],
        gaze_normals[chosen_mask],
        T_target_world,
        R_eye_head,
        T_eye_head_ini)

T_eye_head = calibration_result_translation.x
#%%
calibration_result_both = fh.calibrate_pupil(
        T_head_world[chosen_mask],
        R_head_world[chosen_mask],
        gaze_normals[chosen_mask],
        T_target_world,
        T_eye_head_ini,
        bounds_mm=10
        )

R_eye_head = fh.from_yawpitchroll(calibration_result_both.x[0:3])
T_eye_head = calibration_result_both.x[3:6]

# replace eye point and calculate reference markers again
helmet.ref_points[5, :] = helmet.ref_points[0, :] + T_eye_head
R_head_world, ref_points = helmet.solve(odata_interpolated)
#%%

from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
window = QtWidgets.QWidget()

layout = QtWidgets.QVBoxLayout()

canvas_widget = QtWidgets.QWidget()
layout.addWidget(canvas_widget)

slider = QtWidgets.QSlider(Qt.Horizontal)

layout.addWidget(slider)

window.setLayout(layout)

canvas = vispy.scene.SceneCanvas(parent=canvas_widget, create_native=True, vsync=True, show=False)
view = canvas.central_widget.add_view()

calibration_point_mask = np.arange(255) == calibration_point
rig_scatter = visuals.Markers()
rig_scatter.set_data(rig_led_positions[~calibration_point_mask, :], edge_color=(1, 1, 1, 1), face_color=(1, 1, 1, 1), size=1)
view.add(rig_scatter)

rig_calib_scatter = visuals.Markers()
rig_calib_scatter.set_data(rig_led_positions[calibration_point_mask, :], edge_color=(1, 0, 0, 1), face_color=(1, 0, 0, 1), size=1)
view.add(rig_calib_scatter)

view.camera = 'turntable'
view.camera.center = rig_led_positions[calibration_point, :]

axis = visuals.XYZAxis(parent=view.scene)

start_points = T_head_world + np.einsum('tij,j->ti', R_head_world, T_eye_head)
ray_length = 3000
end_points = start_points + ray_length * np.einsum('tij,tj->ti', np.einsum('tij,jk->tik', R_head_world, R_eye_head), gaze_normals)

rays = np.full((start_points.shape[0] * 3, 3), np.nan)
rays[0::3, :] = start_points
rays[1::3, :] = end_points

ray_lines = visuals.Line()
ray_lines.set_data(color=(0.5, 0.25, 0, 1))
view.add(ray_lines)

error_line = visuals.Line()
error_line.set_data(color=(1, 0, 0, 1))
view.add(error_line)

head_scatter = visuals.Markers()
view.add(head_scatter)

def update_plot(i):
    ray_lines.set_data(rays[i * 3: i*3 + 3, :])
    head_scatter.set_data(ref_points[i, ...], edge_color=(0.5, 0.25, 0, 1), face_color=(0.5, 0.25, 0, 1), size=3)
    
    A = start_points[i, ...]
    B = end_points[i, ...]
    P = rig_led_positions[calibration_point, ...]

    error_point = A + np.dot(P - A, B - A) / np.dot(B - A, B - A) * (B - A)
    error_line.set_data(np.vstack((P, error_point)))
    
    
def slider_changed(value):
    update_plot(value)

slider.setMinimum(0)
slider.setMaximum(start_points.shape[0])
slider.valueChanged.connect(slider_changed)

window.show()
window.resize(400, 300)
window.update()

#%%
settings = {
    'trials_per_shift': 2,
    'shift_magnitudes': np.arange(-5, 6) * 4,
    'default_fixation_led_index': 50,
    'default_target_led_index': 200,
    'max_random_led_offset': 10,
    'before_fixation_color': (1, 0, 0),
    'during_fixation_color': (0, 1, 0),
    'before_response_target_color': (0, 1, 0),
    'during_response_target_color': (0, 0, 1),
    'pupil_min_confidence': 0.25,
    'fixation_threshold': 1.5,
    'fixation_duration': 5,
    'fixation_head_velocity_threshold': 900,
    'saccade_threshold': 2,
    'maximum_saccade_latency': 0.5,
    'maximum_target_reaching_duration': 0.5,
    'after_landing_fixation_threshold': 4,
    'after_landing_fixation_duration': 0.5,
    'with_blanking': True,
    'blanking_duration': 0.25
}

experiment = fh.LedShiftExperiment(othread, pthread, athread, helmet, R_eye_head, rig_led_positions, settings)
for i in range(experiment.trial_matrix.shape[0]):
    experiment.run_trial()

athread.write_uint8(255, 0, 0, 0)


# %%        
othread.should_stop.set()
othread.join()

pthread.should_stop.set()
pthread.join()

pygame.quit()

athread.should_stop.set()
athread.join()

print('done')
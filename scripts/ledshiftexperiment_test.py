import freehead as fh
import numpy as np
import pygame
import os
import pickle

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


probe = fh.FourMarkerProbe()

def extract_probe(ot_data):
    return ot_data[..., 15:27].reshape((-1, 4, 3)).squeeze()

def extract_helmet(ot_data):
    return ot_data[..., 3:15].reshape((-1, 4, 3)).squeeze()

def extract_gaze(p_data):
    return p_data[3:6]

if not 'rig_led_positions' in locals():
    led_rig_indices = [0, 70, 141, 213, 254] # these are the array indices
    probe_tips = np.full((len(led_rig_indices), 3), np.nan)
    for i, led_rig_index in enumerate(led_rig_indices):
        athread.write_uint8(led_rig_index, 100, 0, 0)
        print('Press space to record led with index', led_rig_index)
        while True:
            fh.wait_for_keypress(pygame.K_SPACE)
            current_sample = othread.current_sample.copy()
            rotation, tip = probe.solve(extract_probe(current_sample))
            if np.any(np.isnan(tip)):
                print('Probe not visible, try again.')
                continue
            probe_tips[i, :] = tip
            break
    athread.write_uint8(255, 0, 0, 0)
    
    led_rig_transform_result = fh.get_rig_transform(probe_tips, led_rig_indices)
    
    R_rig = fh.from_yawpitchroll(led_rig_transform_result.x[0:3])
    T_rig = led_rig_transform_result.x[3:6]
    
    rig_led_positions = np.einsum('ij,tj->ti', R_rig, fh.LED_POSITIONS) + T_rig
else:
    print('rig_led_positions calibration already present. Delete if you want to recalibrate.')


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
        
        if np.any(np.isnan(helmet_leds)):
            print('Helmet LEDs not all visible. Try again.')
            continue

        if i == 0:
            helmet = fh.Rigidbody(helmet_leds)
            break
        else:
            _, probe_tip = probe.solve(extract_probe(current_sample))
            if np.any(np.isnan(probe_tip)):
                print('Probe not visible. Try again.')
                continue
            helmet.add_reference_points(helmet_leds, probe_tip)
            break

#%%
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

calibration_result = fh.calibrate_pupil(
        T_head_world,
        R_head_world,
        gaze_normals,
        T_target_world,
        ini_T_eye_head = helmet.ref_points[5, :] - helmet.ref_points[0, :],
        bounds_mm=15)

R_eye_head = fh.from_yawpitchroll(calibration_result.x[0:3])
T_eye_head = calibration_result.x[3:6]

helmet.ref_points[5, :] = T_eye_head + helmet.ref_points[0, :]

led_fixation_no_offset = 50
led_target_no_offset = 200
should_fix_color = (10, 0, 0)
is_fix_color = (0, 10, 0)
should_target_color = (0, 10, 0)
trial_done_color = (0, 0, 10)
fix_threshold_eye = 2
fix_threshold_head = 5
fixation_duration = 0.8
 
n_per_offset = 20
offsets = np.arange(-5, 6) * 2
offset_array = np.repeat(offsets, n_per_offset)
np.random.shuffle(offset_array)

fix_phase = 0
saccade_phase = 1
landing_phase = 2

# %%
o_data_list = []
p_data_list = []
response_list = []
target_shift_list = []
start_led_list = []
target_led_list = []
saccade_i_list = []

settings = {
    'trials_per_shift': 20,
    'shift_magnitudes': np.arange(-5, 6) * 4,
    'default_fixation_led_index': 50,
    'default_target_led_index': 200,
    'max_random_led_offset': 10,
    'before_fixation_color': (5, 0, 0),
    'during_fixation_color': (0, 5, 0),
    'before_response_target_color': (0, 5, 0),
    'during_response_target_color': (0, 0, 5),
    'pupil_min_confidence': 0.25,
    'fixation_threshold': 2,
    'fixation_duration': 0.3,
    'maximum_saccade_latency': 0.4,
    'maximum_saccade_duration': 0.2,
    'after_landing_fixation_threshold': 4,
    'after_landing_fixation_duration': 0.3
}

experiment = fh.LedShiftExperiment(othread, pthread, athread, helmet, R_eye_head, rig_led_positions, settings)

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


data = {
    'trial_data': experiment.trial_data,
    'settings': settings,
    'trial_matrix': experiment.trial_matrix,
    'R_eye_head': R_eye_head,
    'rig_leds': rig_led_positions,
    'helmet_ref_markers': helmet.reference_markers,
    'helmet_ref_points': helmet.ref_points,
}

identifier = 'led_shift_experiment'
counter = 0
while True:
    filename = identifier + '.pickle' if counter == 0 else identifier + '_' + counter + '.pickle'
    if os.path.isfile(filename):
        counter += 1
    else:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


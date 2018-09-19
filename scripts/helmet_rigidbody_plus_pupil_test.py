import freehead as fh
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
import serial
import struct

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
        bounds_mm=12)

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

last_i = None
for s in range(offset_array.size):
    print('trial ', s)
    othread.reset_data_buffer()
    pthread.reset_data_buffer()
    
    phase = fix_phase
    
    was_fixating_eye = False
    t_started_fixating = False
    dt_fixation = 0
    is_saccading = False
    
    offset = np.random.randint(-10, 10)
    led_fixation = led_fixation_no_offset + offset
    led_target = led_target_no_offset + offset
    target_shift = offset_array[s]
    
    athread.write_uint8(led_fixation, *should_fix_color)
    
    while True:
        current_i = pthread.i_current_sample
        if current_i == last_i:
            time.sleep(0)
            continue
        last_i = current_i
        pdata = pthread.current_sample.copy()
        gaze_normals = extract_gaze(pdata)
        if np.any(np.isnan(gaze_normals)) or pdata[6] < 0.25:
            if phase != fix_phase:
                athread.write_uint8(led_fixation, *should_fix_color)
                phase = fix_phase       
            was_fixating_eye = False
            t_started_fixating = False
            dt_fixation = 0
            continue
        R_head_world, ref_points = helmet.solve(extract_helmet(othread.current_sample.copy()))
        if np.any(np.isnan(R_head_world)):
            if phase != fix_phase:
                athread.write_uint8(led_fixation, *should_fix_color)
                phase = fix_phase        
            was_fixating_eye = False
            t_started_fixating = False
            dt_fixation = 0
            continue
        T_eye_world = ref_points[5, :]
        
        if phase == fix_phase:
            eye_to_fixpoint = fh.to_unit(rig_led_positions[led_fixation, :] - T_eye_world)
            gaze_normals_world = R_head_world @ R_eye_head @ gaze_normals
            eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
            is_fixating_eye = eye_to_fixpoint_angle <= fix_threshold_eye

            if not is_fixating_eye and was_fixating_eye:
                athread.write_uint8(led_fixation, *should_fix_color)
                was_fixating_eye = False
                
            if is_fixating_eye and not was_fixating_eye:
                athread.write_uint8(led_fixation, *is_fix_color)
                was_fixating_eye = True
                t_started_fixating = time.monotonic()
            if is_fixating_eye and was_fixating_eye:
                if dt_fixation < fixation_duration:
                    dt_fixation = time.monotonic() - t_started_fixating
                else:
                    athread.write_uint8(led_target, *should_target_color)
                    phase = saccade_phase
                    saccade_i = current_i
                    
        elif phase == saccade_phase:
            eye_to_fixpoint = fh.to_unit(rig_led_positions[led_fixation, :] - T_eye_world)
            gaze_normals_world = R_head_world @ R_eye_head @ gaze_normals
            eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
            is_fixating_eye = eye_to_fixpoint_angle <= fix_threshold_eye
            
            if not is_fixating_eye:
                athread.write_uint8(led_target + target_shift, *should_target_color)
                phase = landing_phase
                was_fixating_eye = False
                dt_fixation = 0
                
        elif phase == landing_phase:
            eye_to_target = fh.to_unit(rig_led_positions[led_target + target_shift, :] - T_eye_world)
            gaze_normals_world = R_head_world @ R_eye_head @ gaze_normals
            eye_to_target_angle = np.rad2deg(np.arccos(eye_to_target @ gaze_normals_world))
            is_fixating_eye = eye_to_target_angle <= fix_threshold_eye
            
            if is_fixating_eye and not was_fixating_eye:
                t_started_fixating = time.monotonic()
                was_fixating_eye = True
            if is_fixating_eye and was_fixating_eye:
                if dt_fixation < fixation_duration:
                    dt_fixation = time.monotonic() - t_started_fixating
                else:
                    # trial done
                    athread.write_uint8(led_target + target_shift, 0, 0, 120)
                    response_list.append(fh.wait_for_keypress(pygame.K_LEFT, pygame.K_RIGHT))
                    o_data_list.append(othread.get_shortened_data())
                    p_data_list.append(pthread.get_shortened_data())
                    start_led_list.append(led_fixation)
                    target_led_list.append(led_target)
                    saccade_i_list.append(saccade_i)
                    break
                
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

# %%

responses_right = np.array(response_list) == 275
offset_matrix = offset_array[None, :] == offsets[:, None]
where_right = np.logical_and(offset_matrix, responses_right)
n_right = where_right.sum(axis=1)
p_right = n_right / n_per_offset

plt.plot(offsets, p_right, '-s')
plt.ylim(-0.1, 1.1)
plt.yticks(np.linspace(0, 1, 11))
plt.xticks(offsets)
plt.ylabel('p(right)')
plt.xlabel('offset in leds')
plt.title('No blanking, angle 150 leds / ca 52 deg')
plt.show()

data = {
    'response list': response_list,
    'o_data_list': o_data_list,
    'p_data_list': p_data_list,
    'start_led_list': start_led_list,
    'target_led_list': target_led_list,
    'saccade_i_list': saccade_i_list,
    'offset_array': offset_array,
    'n_per_offset': n_per_offset,
    'helmet_ref_markers': helmet_ref_markers,
    'helmet_ref_points': helmet_ref_points
}
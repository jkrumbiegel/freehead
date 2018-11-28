import freehead as fh
import numpy as np
import pygame
import os
import pickle
from datetime import datetime
#%%
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

settings = {
    'left_to_right': [True, False],
    'amplitude': [40, 70, 100, 130, 160],
    'shift_percent_approx': [-30, -20, -10, 0, 10, 20, 30],
    # 'fixation_led': 50,
    'before_fixation_color': (10, 0, 0),
    'during_fixation_color': (30, 0, 0),
    'before_response_target_color': (30, 0, 0),
    'during_response_target_color': (10, 0, 0),
    'pupil_min_confidence': 0.15,
    'fixation_threshold': 1.5,
    'fixation_duration': 1,
    'fixation_head_velocity_threshold': 10,
    'saccade_threshold': 2,
    'maximum_saccade_latency': 0.4,
    'maximum_target_reaching_duration': 0.5,
    'after_landing_fixation_threshold': 3,
    'after_landing_fixation_duration': 0.5,
}

trial_frame = fh.create_trial_frame(
        settings,
        block_specific={
            'blanking_duration': [0]
        },
        trial_lambdas={
            'shift': lambda df: int(df['amplitude'] * df['shift_percent_approx'] / 100),
            'fixation_led': lambda df: 22 + np.random.randint(-20, 21)
        })

print('left-most led:', trial_frame['fixation_led'].min())
print('right-most led:', (trial_frame['amplitude'] + trial_frame['shift'] + trial_frame['fixation_led']).max())
print('number of trials:', len(trial_frame))
#%%
subject_prefix = input('Subject prefix: ')
print('Remember to select the pygame window')
experiment = fh.LedShiftExperiment(othread, pthread, athread, rig_led_positions, trial_frame)
experiment_df = experiment.run() 

#%%
fh.save_experiment_files(experiment_df, trial_frame, rig_led_positions, subject_prefix)
# %%        
othread.should_stop.set()
othread.join()

pthread.should_stop.set()
pthread.join()

pygame.quit()

athread.should_stop.set()
athread.join()

print('done')

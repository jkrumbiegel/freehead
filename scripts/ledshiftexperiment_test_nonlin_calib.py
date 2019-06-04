import freehead as fh
import numpy as np
import pygame
import sys
#%% initialize background threads
# pygame.init()
pygame.display.init()
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

#%% record led rig
probe = fh.FourMarkerProbe()
fh.focus_pygame_window()


def extract_probe(ot_data):
    return ot_data[..., 15:27].reshape((-1, 4, 3)).squeeze()


if 'rig_led_positions' not in locals():
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
    'amplitude': [45, 75, 105, 135, 165],
    'shift_percent_approx': [-300/15, -200/15, -100/15, 0, 100/15, 200/15, 300/15],
    'before_fixation_color': (0, 15, 0),
    'during_fixation_color': (25, 0, 0),
    'before_response_target_color': (25, 0, 0),
    'during_response_target_color': (0, 0, 15),
    'pupil_min_confidence': 0,
    'fixation_threshold': 2,
    'fixation_duration': 0.8,
    'fixation_head_velocity_threshold': 30,
    'saccade_threshold': 2,
    'maximum_saccade_latency': 0.8,
    'maximum_target_reaching_duration': 0.8,
    'landing_fixation_threshold': 3,
    'after_landing_fixation_duration': 0.5,
    'inter_trial_interval': 0.7,
}

trial_frame = fh.create_trial_frame(
        settings,
        block_specific={
            # 'blanking_duration': [0, 0.25] * 4  # no blank first
            'blanking_duration': [0.25, 0] * 4  # blank first
        },
        trial_lambdas={
            'shift': lambda df: int(df['amplitude'] * df['shift_percent_approx'] / 100),
            'fixation_led': lambda df: 38 + np.random.randint(0, 10)
        })

print('left-most led:', trial_frame['fixation_led'].min())
print('right-most led:', (trial_frame['amplitude'] + trial_frame['shift'] + trial_frame['fixation_led']).max())
print('number of trials:', len(trial_frame))
#%%
subject_prefix = input('Subject prefix: ')
fh.focus_pygame_window()
experiment = fh.LedShiftExperiment(othread, pthread, athread, rig_led_positions, trial_frame)

sys.setswitchinterval(0.0001)
experiment_df = experiment.run()
sys.setswitchinterval(0.005)

fh.save_experiment_files(experiment_df, trial_frame, rig_led_positions, subject_prefix)

pygame.quit()

othread.should_stop.set()
othread.join()

pthread.should_stop.set()
pthread.join()

athread.should_stop.set()
athread.join()

print('done')

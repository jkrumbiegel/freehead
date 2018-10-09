import freehead as fh
import numpy as np
import time
import enum
import pygame
import pandas as pd
from collections import OrderedDict
from scipy.interpolate import interp1d


class Phase(enum.Enum):
    BEFORE_FIXATION = 0
    DURING_FIXATION = 1
    BEFORE_SACCADE = 2
    DURING_SACCADE = 3
    AFTER_LANDING = 4


class TrialResult(enum.Enum):
    COMPLETED = 0
    FAILED = 1
    CALIBRATE = 2


class LedShiftExperiment:

    def __init__(
            self,
            othread: fh.OptotrakThread,
            pthread: fh.PupilThread,
            athread: fh.ArduinoThread,
            rig_leds: np.ndarray,
            trial_frame: pd.DataFrame,
            calib_duration: 10,
    ):
        self.trial_data = []

        self.othread = othread
        self.pthread = pthread
        self.athread = athread
        self.rig_leds = rig_leds
        self.trial_frame = trial_frame
        self.calib_duration = calib_duration

        self.helmet = None
        self.R_eye_head = None
        self.nonlinear_parameters = None


        # self.trials_per_shift = settings['trials_per_shift']
        # self.shift_magnitudes = settings['shift_magnitudes']
        self.blocks = self.trial_frame['block'].unique()
        self.remaining_trials = np.arange(len(self.trial_frame))

    def run(self):

        self.calibrate()

        for block in self.blocks:
            self.run_block(block)

    def run_block(self, block):
        block_frame = self.trial_frame[self.trial_frame['block'] == block]
        remaining_block_trials = block_frame.index.values
        while remaining_block_trials.size > 0:
            random_index = np.random.randint(0, remaining_block_trials.size)
            random_trial_number = remaining_block_trials[random_index]

            (trial_result, trial_data) = self.run_trial(block_frame[random_trial_number])

            if trial_result == TrialResult.COMPLETED:
                remaining_block_trials = np.delete(remaining_block_trials, random_index)
                # do something with trial data

            elif trial_result == TrialResult.FAILED:
                pass

            elif trial_result == TrialResult.CALIBRATE:
                self.calibrate()

    def run_trial(self, trial_frame):

        self.othread.reset_data_buffer()
        self.pthread.reset_data_buffer()

        # set up variables for one trial
        shift = trial_frame['shift'].iloc[0]
        amplitude = trial_frame['amplitude'].iloc[0]
        fixation_led = trial_frame['fixation_led'].iloc[0]
        target_led = fixation_led + amplitude
        shifted_target_led = target_led + shift
        fixation_threshold = trial_frame['fixation_threshold'].iloc[0]
        fixation_head_velocity_threshold = trial_frame['fixation_head_velocity_threshold'].iloc[0]
        saccade_threshold = trial_frame['saccade_threshold'].iloc[0]
        after_landing_fixation_threshold = trial_frame['after_landing_fixation_threshold'].iloc[0]

        pupil_min_confidence = trial_frame['pupil_min_confidence'].iloc[0]

        before_fixation_color = trial_frame['before_fixation_color'].iloc[0]
        during_fixation_color = trial_frame['during_fixation_color'].iloc[0]
        before_response_target_color = trial_frame['before_response_target_color'].iloc[0]
        during_response_target_color = trial_frame['during_response_target_color'].iloc[0]

        fixation_duration = trial_frame['fixation_duration'].iloc[0]
        blanking_interval = trial_frame['blanking_interval'].iloc[0]
        maximum_target_reaching_duration = trial_frame['maximum_target_reaching_duration'].iloc[0]
        maximum_saccade_latency = trial_frame['maximum_saccade_latency'].iloc[0]
        after_landing_fixation_duration = trial_frame['after_landing_fixation_duration'].iloc[0]
        
        t_started_fixating = None
        i_started_fixating = None
        t_target_appeared = None
        i_target_appeared = None
        t_saccade_started = None
        i_saccade_started = None
        t_saccade_landed = None
        i_saccade_landed = None
        t_blanking_ended = None
        i_blanking_ended = None
        response = None

        # show the fixation led
        self.athread.write_uint8(fixation_led, *before_fixation_color)
        phase = Phase.BEFORE_FIXATION

        t_trial_started = time.monotonic()
        last_i = None
        R_head_world = np.full((3, 3), np.nan)
        # this loop runs during data collection in the trial
        # if trial_successful is true when you break out of it, the trial's parameters and timings are saved
        trial_successful = False
        while True:

            # check that a new pupil sample is available
            current_i = self.pthread.i_current_sample
            if current_i == last_i:
                time.sleep(0)
                continue
            last_i = current_i
            pdata = self.pthread.current_sample.copy()

            gaze_normals = pdata[3:6]
            confidence = pdata[6]

            odata = self.othread.current_sample.copy()
            helmet_leds = odata[3:15].reshape((4, 3))
            last_R_head_world = R_head_world
            R_head_world, helmet_ref_points = self.helmet.solve(helmet_leds)
            T_eye_world = helmet_ref_points[5, :]

            # if helmet rigidbody couldn't be solved or pupil data is bad
            if fh.anynan(R_head_world) or fh.anynan(gaze_normals) or confidence < pupil_min_confidence:
                if phase == Phase.BEFORE_FIXATION:
                    continue
                elif phase == Phase.DURING_FIXATION:
                    phase = Phase.BEFORE_FIXATION
                    self.athread.write_uint8(fixation_led, *before_fixation_color)
                    continue
                else:
                    # in later phases, the target was already visible, start a new one
                    break

            current_head_angular_velocity = 0 if np.allclose(last_R_head_world, R_head_world) else np.rad2deg(
                np.arccos(
                    (np.trace(last_R_head_world @ R_head_world.T) - 1) / 2
                )
            ) * self.othread.server_config['optotrak']['collection_frequency']

            if phase == Phase.BEFORE_FIXATION:

                eye_to_fixpoint = fh.to_unit(self.rig_leds[fixation_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ (
                            fh.normals_nonlinear_angular_transform(
                                self.R_eye_head @ gaze_normals, self.nonlinear_parameters))
                eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                is_fixating = eye_to_fixpoint_angle <= fixation_threshold
                is_holding_still = current_head_angular_velocity <= fixation_head_velocity_threshold

                if is_fixating and is_holding_still:
                    self.athread.write_uint8(fixation_led, *during_fixation_color)
                    t_started_fixating = time.monotonic()
                    i_started_fixating = current_i
                    phase = Phase.DURING_FIXATION

            elif phase == Phase.DURING_FIXATION:

                eye_to_fixpoint = fh.to_unit(self.rig_leds[fixation_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ (
                            fh.normals_nonlinear_angular_transform(
                                self.R_eye_head @ gaze_normals, self.nonlinear_parameters))
                eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                is_fixating = eye_to_fixpoint_angle <= fixation_threshold
                is_holding_still = current_head_angular_velocity <= fixation_head_velocity_threshold

                if not (is_fixating and is_holding_still):
                    self.athread.write_uint8(fixation_led, *before_fixation_color)
                    phase = Phase.BEFORE_FIXATION
                    # if fixation is lost here, don't start a completely new trial, that would be wasteful because
                    # the target wasn't even shown
                else:
                    if time.monotonic() - t_started_fixating >= fixation_duration:
                        i_target_appeared = current_i
                        t_target_appeared = time.monotonic()
                        self.athread.write_uint8(target_led, *before_response_target_color)
                        phase = Phase.BEFORE_SACCADE

            elif phase == Phase.BEFORE_SACCADE:

                if time.monotonic() - t_target_appeared > maximum_saccade_latency:
                    # abort trial because saccade latency was too long
                    break

                eye_to_fixpoint = fh.to_unit(self.rig_leds[fixation_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ (
                    fh.normals_nonlinear_angular_transform(
                        self.R_eye_head @ gaze_normals, self.nonlinear_parameters))
                eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                has_started_saccade = eye_to_fixpoint_angle > saccade_threshold

                if has_started_saccade:
                    i_saccade_started = current_i
                    t_saccade_started = time.monotonic()
                    if blanking_interval == 0:
                        self.athread.write_uint8(shifted_target_led, *before_response_target_color)
                    else:
                        self.athread.write_uint8(255, 0, 0, 0)

                    phase = Phase.DURING_SACCADE

            elif phase == Phase.DURING_SACCADE:

                if time.monotonic() - t_saccade_started >= blanking_interval:
                    if t_blanking_ended is None:
                        t_blanking_ended = time.monotonic()
                        i_blanking_ended = current_i
                        self.athread.write_uint8(shifted_target_led, *before_response_target_color)

                    if time.monotonic() - t_blanking_ended > maximum_target_reaching_duration:
                        # abort trial because saccade to target took too long
                        break

                    eye_to_shifted_target = fh.to_unit(self.rig_leds[shifted_target_led, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ (
                            fh.normals_nonlinear_angular_transform(
                                self.R_eye_head @ gaze_normals, self.nonlinear_parameters))
                    eye_to_shifted_target_angle = np.rad2deg(
                        np.arccos(eye_to_shifted_target @ gaze_normals_world))
                    is_fixating_target = eye_to_shifted_target_angle <= fixation_threshold

                    if is_fixating_target:
                        i_saccade_landed = current_i
                        t_saccade_landed = time.monotonic()
                        phase = Phase.AFTER_LANDING

            elif phase == Phase.AFTER_LANDING:

                if time.monotonic() - t_saccade_landed > after_landing_fixation_duration:
                    self.athread.write_uint8(shifted_target_led, *during_response_target_color)
                    response = fh.wait_for_keypress(pygame.K_LEFT, pygame.K_RIGHT)
                    trial_successful = True
                    break

                eye_to_shifted_target = fh.to_unit(self.rig_leds[shifted_target_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ (
                    fh.normals_nonlinear_angular_transform(
                        self.R_eye_head @ gaze_normals, self.nonlinear_parameters))
                eye_to_shifted_target_angle = np.rad2deg(np.arccos(eye_to_shifted_target @ gaze_normals_world))
                is_fixating_target_after_landing = eye_to_shifted_target_angle <= after_landing_fixation_threshold

                if not is_fixating_target_after_landing:
                    # abort trial because after saccade fixation was not long enough
                    break

        # sampling loop over
        if trial_successful:

            trial_data = OrderedDict([
                ('o_data', self.othread.get_shortened_data()),
                ('p_data', self.pthread.get_shortened_data()),
                ('t_trial_started', t_trial_started),
                ('i_started_fixating', i_started_fixating),
                ('t_started_fixating', t_started_fixating),
                ('i_target_appeared', i_target_appeared),
                ('t_target_appeared', t_target_appeared),
                ('t_saccade_started', t_saccade_started),
                ('i_saccade_started', i_saccade_started),
                ('t_saccade_landed', t_saccade_landed),
                ('i_saccade_landed', i_saccade_landed),
                ('i_blanking_ended', i_blanking_ended),
                ('t_blanking_ended', t_blanking_ended),
                ('response', response),
            ])

            return TrialResult.COMPLETED, trial_data

        else:
            return TrialResult.FAILED, None

    def calibrate(self):

        calibration_point = 127

        while True:

            self.athread.write_uint8(calibration_point, 50, 0, 0)

            print('Calibration pending. Press space to start.')
            fh.wait_for_keypress(pygame.K_SPACE)

            self.othread.reset_data_buffer()
            self.pthread.reset_data_buffer()

            # show calibration led

            print('\nCalibration starting.\n')

            start_time = time.monotonic()
            while time.monotonic() - start_time < self.calib_duration:
                time.sleep(0.1)

            self.athread.write_uint8(255, 0, 0, 0)

            print('\nCalibration over. Optimizing parameters...\n')

            odata = self.othread.get_shortened_data().copy()
            pdata = self.pthread.get_shortened_data().copy()
            gaze_normals = pdata[:, 3:6]

            f_interpolate = interp1d(odata[:, 30], odata[:, 3:15], axis=0)
            odata_interpolated = f_interpolate(pdata[:, 2]).reshape((-1, 4, 3))

            R_head_world, ref_points = self.helmet.solve(odata_interpolated)
            T_head_world = ref_points[:, 0]

            confidence_enough = pdata[:, 6] > 0.6
            rotations_valid = ~np.any(np.isnan(R_head_world).reshape((-1, 9)), axis=1)
            chosen_mask = np.logical_and(confidence_enough, rotations_valid)

            T_target_world = np.tile(self.rig_leds[calibration_point, :], (chosen_mask.sum(), 1))

            ini_T_eye_head = self.helmet.ref_points[5, :]

            calibration_result = fh.calibrate_pupil_nonlinear(
                T_head_world, R_head_world, gaze_normals, T_target_world, ini_T_eye_head=ini_T_eye_head)

            print('Optimization done.\n')
            print('Error: ', calibration_result.fval, '\n')
            print('Parameters: ', calibration_result.x, '\n')
            print('Accept calibration? Yes: Space, No: Escape')
            key = fh.wait_for_keypress(pygame.K_SPACE, pygame.K_ESCAPE)

            if key == pygame.K_SPACE:
                self.helmet.ref_points[5, :] = self.helmet.ref_points[0, :] + calibration_result.x[0:3]
                self.R_eye_head = fh.from_yawpitchroll(calibration_result.x[4:6])
                self.nonlinear_parameters = calibration_result.x[6:12]
                break

            elif key == pygame.K_ESCAPE:
                continue
















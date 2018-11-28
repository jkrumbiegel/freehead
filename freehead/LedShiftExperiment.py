import freehead as fh
import numpy as np
import time
import enum
import pygame
import pandas as pd
from collections import OrderedDict
from scipy.interpolate import interp1d
from typing import Optional


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
            calib_duration=10,
            after_reset_wait=15
    ):
        self.trial_data = []

        self.othread = othread
        self.pthread = pthread
        self.athread = athread
        self.rig_leds = rig_leds
        self.trial_frame = trial_frame
        self.calib_duration = calib_duration
        self.after_reset_wait = after_reset_wait

        self.helmet = None
        self.R_eye_head = None
        self.nonlinear_parameters = None

        self.blocks = self.trial_frame['block'].unique()
        self.remaining_trials = np.arange(len(self.trial_frame))

    def run(self) -> pd.DataFrame:

        self.create_helmet()
        self.calibrate()

        experiment_dataframe = None
        block_lengths = self.trial_frame['block'].value_counts(sort=False).values
        block_borders = np.concatenate(([0], np.cumsum(block_lengths)))

        for block in self.blocks:
            self.pause_experiment()

            block_dataframe = self.run_block(block)
            block_dataframe['trial_in_block'] = block_dataframe.index
            # make a new index so the trial numbers are correct when appending. trials are in random order depending on
            # when they were successfully finished
            block_dataframe.index = pd.Series(np.arange(block_borders[block], block_borders[block + 1]))

            if experiment_dataframe is None:
                experiment_dataframe = block_dataframe
            else:
                experiment_dataframe = experiment_dataframe.append(block_dataframe)

        self.play_finish_animation()
        return experiment_dataframe

    def run_block(self, block) -> pd.DataFrame:
        block_frame = self.trial_frame[self.trial_frame['block'] == block]
        remaining_block_trials = block_frame.index.values
        block_dataframe = None
        trial_in_block = 0
        while remaining_block_trials.size > 0:
            random_index = np.random.randint(0, remaining_block_trials.size)
            random_trial_number = remaining_block_trials[random_index]

            (trial_result, trial_data) = self.run_trial(block_frame.loc[random_trial_number])

            if trial_result == TrialResult.COMPLETED:
                remaining_block_trials = np.delete(remaining_block_trials, random_index)

                trial_data.update({'trial_number': random_trial_number, 'block': block})
                trial_data.move_to_end('block', last=False)
                trial_data.move_to_end('trial_number', last=False)

                if block_dataframe is None:
                    block_dataframe = pd.DataFrame(trial_data, index=[trial_in_block])
                else:
                    block_dataframe = block_dataframe.append(pd.DataFrame(trial_data, index=[trial_in_block]))
                trial_in_block += 1

            elif trial_result == TrialResult.FAILED:
                pass

            elif trial_result == TrialResult.CALIBRATE:
                self.create_helmet()
                self.calibrate()

        return block_dataframe

    def run_trial(self, trial_frame: pd.DataFrame) -> (TrialResult, Optional[OrderedDict]):

        self.othread.reset_data_buffer()
        self.pthread.reset_data_buffer()

        # set up variables for one trial
        left_to_right = trial_frame['left_to_right']
        shift = trial_frame['shift'] if left_to_right else -trial_frame['shift']
        amplitude = trial_frame['amplitude'] if left_to_right else -trial_frame['amplitude']
        fixation_led = trial_frame['fixation_led'] if left_to_right else 254 - trial_frame['fixation_led']
        target_led = fixation_led + amplitude
        shifted_target_led = target_led + shift
        fixation_threshold = trial_frame['fixation_threshold']
        fixation_head_velocity_threshold = trial_frame['fixation_head_velocity_threshold']
        saccade_threshold = trial_frame['saccade_threshold']
        after_landing_fixation_threshold = trial_frame['after_landing_fixation_threshold']

        pupil_min_confidence = trial_frame['pupil_min_confidence']

        before_fixation_color = trial_frame['before_fixation_color']
        during_fixation_color = trial_frame['during_fixation_color']
        before_response_target_color = trial_frame['before_response_target_color']
        during_response_target_color = trial_frame['during_response_target_color']

        fixation_duration = trial_frame['fixation_duration']
        blanking_duration = trial_frame['blanking_duration']
        maximum_target_reaching_duration = trial_frame['maximum_target_reaching_duration']
        maximum_saccade_latency = trial_frame['maximum_saccade_latency']
        after_landing_fixation_duration = trial_frame['after_landing_fixation_duration']
        inter_trial_interval = trial_frame['inter_trial_interval']
        
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

        # turn off all leds
        self.athread.write_uint8(255, 0, 0, 0)
        # do the inter trial interval here, too many exit points
        time.sleep(inter_trial_interval)
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

            # do calibration if escape was pressed
            if fh.was_key_pressed(pygame.K_ESCAPE):
                return TrialResult.CALIBRATE, None

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
            if fh.anynan(R_head_world) or fh.anynan(gaze_normals) or (confidence < pupil_min_confidence and phase != Phase.DURING_SACCADE):
                if phase == Phase.BEFORE_FIXATION:
                    continue
                elif phase == Phase.DURING_FIXATION:
                    if fh.anynan(R_head_world):
                        print('head was not visible during fixation')
                    if fh.anynan(gaze_normals):
                        print('nan values in gaze normals during fixation')
                    if confidence < pupil_min_confidence:
                        print('pupil confidence was too low during fixation')
                    phase = Phase.BEFORE_FIXATION
                    self.athread.write_uint8(fixation_led, *before_fixation_color)
                    continue
                else:
                    if fh.anynan(R_head_world):
                        print('head was not visible')
                    if fh.anynan(gaze_normals):
                        print('nan values in gaze normals')
                    if confidence < pupil_min_confidence:
                        print('pupil confidence was too low')
                    break

            current_head_angular_velocity = 0 if np.allclose(last_R_head_world, R_head_world) else np.rad2deg(
                np.arccos(
                    (np.trace(last_R_head_world @ R_head_world.T) - 1) / 2
                )
            ) * self.othread.server_config['optotrak']['collection_frequency']

            if phase == Phase.BEFORE_FIXATION:

                eye_to_fixpoint = fh.to_unit(self.rig_leds[fixation_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ fh.normals_nonlinear_angular_transform(self.R_eye_head @ gaze_normals, self.nonlinear_parameters)
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
                gaze_normals_world = R_head_world @ fh.normals_nonlinear_angular_transform(self.R_eye_head @ gaze_normals, self.nonlinear_parameters)
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
                    print('maximum saccade latency exceeded')
                    break

                eye_to_fixpoint = fh.to_unit(self.rig_leds[fixation_led, :] - T_eye_world)
                gaze_normals_world = R_head_world @ fh.normals_nonlinear_angular_transform(self.R_eye_head @ gaze_normals, self.nonlinear_parameters)
                eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                has_started_saccade = eye_to_fixpoint_angle > saccade_threshold

                if has_started_saccade:
                    i_saccade_started = current_i
                    t_saccade_started = time.monotonic()
                    if blanking_duration == 0:
                        self.athread.write_uint8(shifted_target_led, *before_response_target_color)
                    else:
                        self.athread.write_uint8(255, 0, 0, 0)

                    phase = Phase.DURING_SACCADE

            elif phase == Phase.DURING_SACCADE:

                if time.monotonic() - t_saccade_started >= blanking_duration:
                    if t_blanking_ended is None:
                        t_blanking_ended = time.monotonic()
                        i_blanking_ended = current_i
                        self.athread.write_uint8(shifted_target_led, *before_response_target_color)

                    if time.monotonic() - t_blanking_ended > maximum_target_reaching_duration:
                        print('maximum target reaching duration was exceeded')
                        break

                    eye_to_shifted_target = fh.to_unit(self.rig_leds[shifted_target_led, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ fh.normals_nonlinear_angular_transform(
                        self.R_eye_head @ gaze_normals, self.nonlinear_parameters)
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
                    response_key = fh.wait_for_keypress(pygame.K_LEFT, pygame.K_RIGHT)
                    response = 'left' if response_key == pygame.K_LEFT else 'right'
                    trial_successful = True
                    break

                # this part might have been triggered to easily, leave it out (target only has to be reached once)

                # eye_to_shifted_target = fh.to_unit(self.rig_leds[shifted_target_led, :] - T_eye_world)
                # gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                # eye_to_shifted_target_angle = np.rad2deg(np.arccos(eye_to_shifted_target @ gaze_normals_world))
                # is_fixating_target_after_landing = eye_to_shifted_target_angle <= after_landing_fixation_threshold
                #
                # if not is_fixating_target_after_landing:
                #     print('landing target was not properly fixated')
                #     break

        # sampling loop over
        if trial_successful:

            trial_data = OrderedDict([
                # arrays need to be wrapped in a list so pandas doesn't try to make them long columns
                ('o_data', [self.othread.get_shortened_data()]),
                ('p_data', [self.pthread.get_shortened_data()]),
                ('helmet', self.helmet),
                ('nonlinear_parameters', [self.nonlinear_parameters]),
                ('R_eye_head', [self.R_eye_head]),
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

        self.athread.write_uint8(calibration_point, 128, 0, 0)
        print('Press space to reset eye calibration.')
        fh.wait_for_keypress(pygame.K_SPACE)
        self.pthread.reset_3d_eye_model()
        self.athread.write_uint8(calibration_point, 0, 0, 128)

        time.sleep(self.after_reset_wait)

        while True:

            self.athread.write_uint8(calibration_point, 128, 0, 0)

            print('Calibration pending. Press space to start.')
            fh.wait_for_keypress(pygame.K_SPACE)

            self.athread.write_uint8(calibration_point, 255, 0, 0)

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

            f_interpolate = interp1d(odata[:, 30], odata[:, 3:15], axis=0, bounds_error=False, fill_value=np.nan)
            odata_interpolated = f_interpolate(pdata[:, 2]).reshape((-1, 4, 3))

            R_head_world, ref_points = self.helmet.solve(odata_interpolated)
            T_head_world = ref_points[:, 0, :]

            confidence_enough = pdata[:, 6] > 0.6
            rotations_valid = ~np.any(np.isnan(R_head_world).reshape((-1, 9)), axis=1)
            chosen_mask = confidence_enough & rotations_valid

            T_target_world = np.tile(self.rig_leds[calibration_point, :], (chosen_mask.sum(), 1))

            ini_T_eye_head = self.helmet.ref_points[5, :] - self.helmet.ref_points[0, :]

            calibration_result = fh.calibrate_pupil_nonlinear(
                T_head_world[chosen_mask, ...],
                R_head_world[chosen_mask, ...],
                gaze_normals[chosen_mask, ...],
                T_target_world,
                ini_T_eye_head=ini_T_eye_head,
                leave_T_eye_head=True)

            print('Optimization done.\n')
            print('Error: ', calibration_result.fun, '\n')
            print('Parameters: ', calibration_result.x, '\n')

            # signal quality of calibration via leds
            if calibration_result.fun <= 0.7:
                self.athread.write_uint8(127, 0, 255, 0)  # green
            elif 0.7 < calibration_result.fun <= 1:
                self.athread.write_uint8(127, 255, 255, 0)  # yellow
            else:
                self.athread.write_uint8(127, 255, 0, 0)  # red

            print('Accept calibration? Yes: Space, No: Escape')
            key = fh.wait_for_keypress(pygame.K_SPACE, pygame.K_ESCAPE)

            if key == pygame.K_SPACE:
                self.R_eye_head = fh.from_yawpitchroll(calibration_result.x[0:3])
                self.nonlinear_parameters = calibration_result.x[3:9]
                # self.helmet.ref_points[5, :] = self.helmet.ref_points[0, :] + calibration_result.x[9:12]
                break

            elif key == pygame.K_ESCAPE:
                continue
        self.athread.write_uint8(255, 0, 0, 0)  # leds off

    def create_helmet(self):
        head_measurement_points = [
            'head straight',
            'nasion',
            'inion',
            'right ear',
            'left ear',
            'right eye']

        for i, measurement_point in enumerate(head_measurement_points):
            print('Press space to measure: ' + measurement_point)
            # light up an led to signal which measurement is going on
            signal_led = 255
            signal_length = 1
            while True:
                self.athread.write_uint8(signal_led, 255, 255, 255)  # bright light to start and see something
                fh.wait_for_keypress(pygame.K_SPACE)
                current_sample = self.othread.current_sample.copy()
                helmet_leds = current_sample[3:15].reshape((4, 3))

                if np.any(np.isnan(helmet_leds)):
                    print('Helmet LEDs not all visible. Try again.')
                    self.athread.write_uint8(signal_led, 255, 0, 0)  # red light for failure
                    time.sleep(signal_length)
                    continue

                if i == 0:
                    helmet = fh.Rigidbody(helmet_leds)
                    self.athread.write_uint8(signal_led, 0, 255, 0)  # green light for success
                    time.sleep(signal_length)
                    break
                else:
                    _, probe_tip = fh.FourMarkerProbe().solve(current_sample[15:27].reshape((4, 3)))
                    if np.any(np.isnan(probe_tip)):
                        print('Probe not visible. Try again.')
                        self.athread.write_uint8(signal_led, 255, 0, 0)  # red light for failure
                        time.sleep(signal_length)
                        continue
                    helmet.add_reference_points(helmet_leds, probe_tip)

                    # replace eye measurement
                    if i == 5:
                        nasion_to_inion = fh.to_unit(helmet.ref_points[2, :] - helmet.ref_points[1, :])
                        # estimate eye at 15 mm inwards from probe in nasion inion direction
                        estimated_eye_position = helmet.ref_points[5, :] + 15 * nasion_to_inion
                        # replace measured value with estimation
                        helmet.ref_points[5, :] = estimated_eye_position

                    self.athread.write_uint8(signal_led, 0, 255, 0)  # green light for success
                    time.sleep(signal_length)
                    break

        self.helmet = helmet
        print('Helmet creation done.')
        self.athread.write_uint8(255, 0, 0, 0)  # turn off leds

    def pause_experiment(self):
        # make three leds pulse to signal that there's currently a pause
        max_brightness = 255
        duration_cycle = 0.4
        n_cycle_updates = 30
        led_update_interval = duration_cycle / n_cycle_updates
        while True:
            for i in range(3):
                led_index = 127 + int((i - i/2) * 10)
                for brightness in (np.sin(np.linspace(0, np.pi, n_cycle_updates)) * max_brightness).astype(np.int):
                    self.athread.write_uint8(led_index, brightness, 0, 0)
                    time.sleep(led_update_interval)
                    # stop pause animation if space is pressed
                    if fh.was_key_pressed(pygame.K_SPACE):
                        return
            time.sleep(duration_cycle)

    def play_finish_animation(self):

        for i in range(300):
            led = np.random.randint(0, 255)
            r = 255 if led % 3 == 0 else 0
            g = 255 if (led + 1) % 3 == 0 else 0
            b = 255 if (led + 2) % 3 == 0 else 0
            self.athread.write_uint8(led, r, g, b)
            time.sleep(0.02)
        self.athread.write_uint8(255, 0, 0, 0)

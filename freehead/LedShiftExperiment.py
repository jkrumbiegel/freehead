import freehead as fh
import numpy as np
import time
import enum
import pygame


class Phase(enum.Enum):
    BEFORE_FIXATION = 0
    DURING_FIXATION = 1
    BEFORE_SACCADE = 2
    DURING_SACCADE = 3
    AFTER_LANDING = 4


class LedShiftExperiment:

    def __init__(
            self,
            othread: fh.OptotrakThread,
            pthread: fh.PupilThread,
            athread: fh.ArduinoThread,
            helmet: fh.Rigidbody,
            R_eye_head: np.ndarray,
            rig_leds: np.ndarray,
            settings: dict
    ):
        self.trial_data = []

        self.othread = othread
        self.pthread = pthread
        self.athread = athread
        self.helmet = helmet
        self.R_eye_head = R_eye_head
        self.rig_leds = rig_leds

        self.trials_per_shift = settings['trials_per_shift']
        self.shift_magnitudes = settings['shift_magnitudes']
        self.trial_matrix = self.build_trial_matrix()
        self.remaining_trials = np.arange(self.trial_matrix.shape[0])

        self.default_fixation_led_index = settings['default_fixation_led_index']
        self.default_target_led_index = settings['default_target_led_index']
        self.max_random_led_offset = settings['max_random_led_offset']

        self.before_fixation_color = settings['before_fixation_color']
        self.during_fixation_color = settings['during_fixation_color']
        self.before_response_target_color = settings['before_response_target_color']
        self.during_response_target_color = settings['during_response_target_color']

        self.pupil_min_confidence = settings['pupil_min_confidence']
        self.fixation_threshold = settings['fixation_threshold']
        self.fixation_duration = settings['fixation_duration']
        self.fixation_head_velocity_threshold = settings['fixation_head_velocity_threshold']
        self.saccade_threshold = settings['saccade_threshold']
        self.maximum_saccade_latency = settings['maximum_saccade_latency']
        self.maximum_target_reaching_duration = settings['maximum_target_reaching_duration']
        self.after_landing_fixation_threshold = settings['after_landing_fixation_threshold']
        self.after_landing_fixation_duration = settings['after_landing_fixation_duration']

        self.with_blanking = settings['with_blanking']
        if self.with_blanking:
            self.blanking_duration = settings['blanking_duration']

    def build_trial_matrix(self):
        return np.repeat(self.shift_magnitudes, self.trials_per_shift)

    def run_trial(self):

        trial_successful = False
        while not trial_successful:

            self.othread.reset_data_buffer()
            self.pthread.reset_data_buffer()

            # set up variables for one trial
            random_index = np.random.randint(self.remaining_trials.size)
            chosen_trial = self.remaining_trials[random_index]
            current_shift = self.trial_matrix[chosen_trial]
            current_offset = np.random.randint(-self.max_random_led_offset, self.max_random_led_offset + 1)
            current_fixation = self.default_fixation_led_index + current_offset
            current_target = self.default_target_led_index + current_offset
            current_shifted_target = self.default_target_led_index + current_offset + current_shift

            if self.with_blanking:
                t_blanking_ended = None

            # show the fixation led
            self.athread.write_uint8(current_fixation, *self.before_fixation_color)
            phase = Phase.BEFORE_FIXATION

            t_trial_started = time.monotonic()
            last_i = None
            R_head_world = np.full((3, 3), np.nan)
            # this loop runs during data collection in the trial
            # if you break out of it and trial_successful is not true, another try is started with new randomly picked
            # settings
            # if trial_successful is true, the trial's parameters and timings are saved
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
                if fh.anynan(R_head_world) or fh.anynan(gaze_normals) or confidence < self.pupil_min_confidence:
                    if phase == Phase.BEFORE_FIXATION:
                        continue
                    elif phase == Phase.DURING_FIXATION:
                        phase = Phase.BEFORE_FIXATION
                        self.athread.write_uint8(current_fixation, *self.before_fixation_color)
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

                    eye_to_fixpoint = fh.to_unit(self.rig_leds[current_fixation, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                    eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                    is_fixating = eye_to_fixpoint_angle <= self.fixation_threshold
                    is_holding_still = current_head_angular_velocity <= self.fixation_head_velocity_threshold

                    if is_fixating and is_holding_still:
                        self.athread.write_uint8(current_fixation, *self.during_fixation_color)
                        t_started_fixating = time.monotonic()
                        i_started_fixating = current_i
                        phase = Phase.DURING_FIXATION

                elif phase == Phase.DURING_FIXATION:

                    eye_to_fixpoint = fh.to_unit(self.rig_leds[current_fixation, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                    eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                    is_fixating = eye_to_fixpoint_angle <= self.fixation_threshold
                    is_holding_still = current_head_angular_velocity <= self.fixation_head_velocity_threshold

                    if not (is_fixating and is_holding_still):
                        self.athread.write_uint8(current_fixation, *self.before_fixation_color)
                        phase = Phase.BEFORE_FIXATION
                        # if fixation is lost here, don't start a completely new trial, that would be wasteful because
                        # the target wasn't even shown
                    else:
                        if time.monotonic() - t_started_fixating >= self.fixation_duration:
                            i_target_appeared = current_i
                            t_target_appeared = time.monotonic()
                            self.athread.write_uint8(current_target, *self.before_response_target_color)
                            phase = Phase.BEFORE_SACCADE

                elif phase == Phase.BEFORE_SACCADE:

                    if time.monotonic() - t_target_appeared > self.maximum_saccade_latency:
                        # abort trial because saccade latency was too long
                        break

                    eye_to_fixpoint = fh.to_unit(self.rig_leds[current_fixation, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                    eye_to_fixpoint_angle = np.rad2deg(np.arccos(eye_to_fixpoint @ gaze_normals_world))
                    has_started_saccade = eye_to_fixpoint_angle > self.saccade_threshold

                    if has_started_saccade:
                        i_saccade_started = current_i
                        t_saccade_started = time.monotonic()
                        if self.with_blanking:
                            self.athread.write_uint8(255, 0, 0, 0)
                        else:
                            self.athread.write_uint8(current_shifted_target, *self.before_response_target_color)

                        phase = Phase.DURING_SACCADE

                elif phase == Phase.DURING_SACCADE:
                    if not self.with_blanking:
                        if time.monotonic() - t_saccade_started > self.maximum_target_reaching_duration:
                            # abort trial because saccade took too long
                            break

                        eye_to_shifted_target = fh.to_unit(self.rig_leds[current_shifted_target, :] - T_eye_world)
                        gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                        eye_to_shifted_target_angle = np.rad2deg(np.arccos(eye_to_shifted_target @ gaze_normals_world))
                        is_fixating_target = eye_to_shifted_target_angle <= self.fixation_threshold

                        if is_fixating_target:
                            i_saccade_landed = current_i
                            t_saccade_landed = time.monotonic()
                            phase = Phase.AFTER_LANDING
                    else:
                        if time.monotonic() - t_saccade_started >= self.blanking_duration:
                            if t_blanking_ended is None:
                                t_blanking_ended = time.monotonic()
                                i_blanking_ended = current_i
                                self.athread.write_uint8(current_shifted_target, *self.before_response_target_color)

                            if time.monotonic() - t_blanking_ended > self.maximum_target_reaching_duration:
                                # abort trial because saccade to target took too long
                                break

                            eye_to_shifted_target = fh.to_unit(self.rig_leds[current_shifted_target, :] - T_eye_world)
                            gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                            eye_to_shifted_target_angle = np.rad2deg(
                                np.arccos(eye_to_shifted_target @ gaze_normals_world))
                            is_fixating_target = eye_to_shifted_target_angle <= self.fixation_threshold

                            if is_fixating_target:
                                i_saccade_landed = current_i
                                t_saccade_landed = time.monotonic()
                                phase = Phase.AFTER_LANDING

                elif phase == Phase.AFTER_LANDING:

                    if time.monotonic() - t_saccade_landed > self.after_landing_fixation_duration:
                        self.athread.write_uint8(current_shifted_target, *self.during_response_target_color)
                        response = fh.wait_for_keypress(pygame.K_LEFT, pygame.K_RIGHT)
                        trial_successful = True
                        break

                    eye_to_shifted_target = fh.to_unit(self.rig_leds[current_shifted_target, :] - T_eye_world)
                    gaze_normals_world = R_head_world @ self.R_eye_head @ gaze_normals
                    eye_to_shifted_target_angle = np.rad2deg(np.arccos(eye_to_shifted_target @ gaze_normals_world))
                    is_fixating_target_after_landing = eye_to_shifted_target_angle <= self.after_landing_fixation_threshold

                    if not is_fixating_target_after_landing:
                        # abort trial because after saccade fixation was not long enough
                        break

            # sampling loop over
        # trial loop over

        current_trial_data = {
            'trial_index': chosen_trial,
            'o_data': self.othread.get_shortened_data(),
            'p_data': self.pthread.get_shortened_data(),
            't_trial_started': t_trial_started,
            'i_started_fixating': i_started_fixating,
            't_started_fixating': t_started_fixating,
            'i_target_appeared': i_target_appeared,
            't_target_appeared': t_target_appeared,
            't_saccade_started': t_saccade_started,
            'i_saccade_started': i_saccade_started,
            't_saccade_landed': t_saccade_landed,
            'i_saccade_landed': i_saccade_landed,
            'response': response
        }

        if self.with_blanking:
            current_trial_data['i_blanking_ended'] = i_blanking_ended
            current_trial_data['t_blanking_ended'] = t_blanking_ended

        self.trial_data.append(current_trial_data)

        # remove the completed trial from the remaining trials array
        self.remaining_trials = np.delete(self.remaining_trials, random_index)

















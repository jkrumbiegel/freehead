import numpy as np
import pandas as pd
import freehead as fh
from collections import OrderedDict


def apply_analysis_pipeline_for_valid_trials(df: pd.DataFrame):

    df.drop(
        df[df.apply(lambda r: r['eng_merg'] is None, axis=1)].index,
        inplace=True
    )

    def i_shift_done_rel(r):
        difference = r['t_led_shift_done'] - r['t_saccade_started']
        in_ms = 1000 * difference
        distances_to_available_timestamps = in_ms - r['t_sacc']
        i_closest_timestamp = np.argmin(np.abs(distances_to_available_timestamps))
        return i_closest_timestamp

    fh.array_apply(
        df,
        OrderedDict([
            # index of fastest saccade
            ('i_max_amp_sacc', lambda r: np.argmax(np.abs(r['eng_merg'][3]))),
            ('max_sacc_amp', lambda r: r['eng_merg'][3][r['i_max_amp_sacc']]),
            (('df', 'max_sacc_amp_uni'), lambda df: df['direction_sign'] * df['max_sacc_amp']),
            ('i_start_max_amp_sacc', lambda r: r['eng_merg'][0][r['i_max_amp_sacc']][0]),
            ('i_end_max_amp_sacc', lambda r: r['eng_merg'][0][r['i_max_amp_sacc']][1]),
            ('start_max_amp_sacc', lambda r: r['t_sacc'][r['i_start_max_amp_sacc']]),
            ('end_max_amp_sacc', lambda r: r['t_sacc'][r['i_end_max_amp_sacc']]),
            # real amplitude dependent on eye position
            ('real_amplitude',
                lambda r: r['eye_to_target_ang'][r['i_start_max_amp_sacc'] - 5, 0]  # 5 samples before saccade onset
                          - r['eye_to_fixation_ang'][r['i_start_max_amp_sacc'] - 5, 0]),
            (('df', 'real_amplitude_uni'), lambda df: df['real_amplitude'] * df['direction_sign']),
            # detected saccade related timestamps
            (('df', 't_det_sacc'), lambda df: df['t_sacc'] - df['start_max_amp_sacc']),

            # relative led shift
            ('i_shift_done_rel', i_shift_done_rel),

            # amplitude of saccade with highest peak velocity
            ('vpeak_max_amp_sacc', lambda r: r['eng_merg'][1][r['i_max_amp_sacc']]),
            # duration of fastest saccade
            (('df','dur_max_amp_sacc'), lambda df: df['end_max_amp_sacc'] - df['start_max_amp_sacc']),
            # landing points of main saccades
            ('land_pos_sacc_hor', lambda r: r['gaze_angle_vs_target'][r['i_end_max_amp_sacc'], 0]),
            ('land_pos_sacc_ver', lambda r: r['gaze_angle_vs_target'][r['i_end_max_amp_sacc'], 1]),
            # ratio max amplitude real amplitude
            (('df', 'ratio_max_real_amp'), lambda df: df['max_sacc_amp'] / df['real_amplitude']),
            # landing error
            (('df', 'landing_error'), lambda df: df['ratio_max_real_amp'] - 1),
            # abs landing error
            (('df', 'landing_err_abs'), lambda df: df['landing_error'].abs()),
            # head contribution
            ('head_contrib_sacc',
                lambda r: r['ypr_head_world'][r['i_end_max_amp_sacc'], 0] - r['ypr_head_world'][r['i_start_max_amp_sacc'], 0]),
            # head contribution unidirectional
            (('df', 'head_contrib_sacc_uni'), lambda df: df['head_contrib_sacc'] * df['direction_sign']),
            # relative head contribution
            (('df', 'rel_head_contrib_sacc'), lambda df: df['head_contrib_sacc'] / df['max_sacc_amp']),
            # head contribution
            ('head_contrib_shift_done',
                lambda r: r['ypr_head_world'][r['i_shift_done_rel'], 0] - r['ypr_head_world'][r['i_start_max_amp_sacc'], 0]),
            # relative head contribution
            ('rel_head_contrib_shift_done', lambda r: r['head_contrib_shift_done'] / r['max_sacc_amp']),
            # max head velocity
            ('max_head_velocity',
                lambda r: np.nanmax(np.abs(np.diff(r['ypr_head_world'][:, 0]) / np.diff(r['t_det_sacc'])))),
            # was the max amp saccade detected after the recorded trigger? then it should be dismissed
            ('saccade_after_threshold', lambda r: r['start_max_amp_sacc'] > 0),
            # did the led change happen before the saccade was over? if not, it should be dismissed
            ('led_change_before_saccade_end',
                lambda r: r['end_max_amp_sacc'] >= (
                    (r['t_target_turned_off'] if r['blanking_duration'] > 0 else r['t_led_shift_done'])
                        - r['t_saccade_started'])),
            # max amp saccade ratio criterion
            ('max_amp_saccade_length_valid', lambda r: 1.25 > r['ratio_max_real_amp'] > 0.75),
            # do all criteria apply?
            ('valid_trials', lambda r: (not r['saccade_after_threshold']) and r['led_change_before_saccade_end'] and r[
                'max_amp_saccade_length_valid']),
            # direction vectors from inion to nasion in world
            ('inion_nasion_world', lambda r: fh.to_unit(r['Ts_head_world'][:, 1, :] - r['Ts_head_world'][:, 2, :])),
            # angles for inion to nasion
            ('inion_nasion_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['inion_nasion_world']))),
            # difference of inion->nasion and eye->target vector angles
            ('d_ininas_eye_target_ang', lambda r: r['inion_nasion_ang'] - r['eye_to_target_ang']),
            # difference of inion->nasion and eye->fixation vector angles
            ('d_ininas_eye_fix_ang', lambda r: r['inion_nasion_ang'] - r['eye_to_fixation_ang']),
            # difference of inion->nasion and gaze vector angles
            ('d_gaze_ininas_ang', lambda r: r['inion_nasion_ang'] - r['gaze_in_world_ang']),
            # inion nasion angle relative to head for centering the in head angles
            ('ininas_ref_ang',
             lambda r: np.rad2deg(fh.to_azim_elev(r['helmet'].ref_points[1, :] - r['helmet'].ref_points[2, :]))),
            # centered gaze ang head
            ('gaze_ang_head_centered', lambda r: r['gaze_in_head_ang'] - r['ininas_ref_ang'])
        ]),
        add_inplace=True,
        print_log=True
    )
import pandas as pd
import freehead as fh
import numpy as np
from scipy.signal import savgol_filter
from collections import OrderedDict
import warnings


def prepend_nan(arr, axis=0, n=1):
    pad_shape = [s if i != axis else n for i, s in enumerate(arr.shape)]
    pad = np.full(pad_shape, np.nan, dtype=arr.dtype)
    return np.concatenate((pad, arr), axis=axis)


def apply_analysis_pipeline_for_all_trials(df: pd.DataFrame):

    warnings.filterwarnings('ignore', category=np.RankWarning)
    
    df.rename(columns={'shift_percent_approx': 'shift_percent'}, inplace=True)

    fh.array_apply(
        df,
        OrderedDict([
            # chosen so that to target direction is positive (right to left is positive angle in mathematics)
            ('direction_sign', lambda r: -1 if r['left_to_right'] else +1),
            ('fixation_led', lambda r: r['fixation_led'] if r['left_to_right'] else 254 - r['fixation_led']),
            (('df', 'target_led'), lambda df: df['fixation_led'] - df['direction_sign'] * df['amplitude']),
            (('df', 'starget_led'), lambda df: df['target_led'] - df['direction_sign'] * df['shift']),
            ('is_outward_response', lambda r: r['response'] == ('right' if r['left_to_right'] else 'left')),
            ('response_ward', lambda r: 'outward' if r['is_outward_response'] else 'inward'),
            ('correct_response', lambda r: None if r['shift'] == 0 else 'right' if (r['shift'] > 0) == r['left_to_right'] else 'left'),
            ('is_correct', lambda r: None if r['correct_response'] is None else r['correct_response'] == r['response']),
            ('shift_percent_uni', lambda r: r['shift_percent'] if r['left_to_right'] else -r['shift_percent']),
            # new time index for upsampling
            ('t_sacc', lambda r: np.arange(-400, 801, 5)),
            # pupil data in around saccade interval upsampled
            ('p_data_upsampled', lambda r: fh.interpolate_a_onto_b_time(r['p_data'][:, 2:5],
                                                                1000 * (r['p_data'][:, 0] - r['t_saccade_started']),
                                                                r['t_sacc'], kind='linear')),
            # optotrak data upsampled
            ('o_data_upsampled', lambda r: fh.interpolate_a_onto_b_time(r['o_data'][:, 3:15],
                                                                  1000 * (r['o_data'][:, 30] - r['t_saccade_started']),
                                                                  r['t_sacc'], kind='linear')),
            # latency of pupil signal
            ('pupil_latency', lambda r: fh.interpolate_a_onto_b_time(r['p_data'][:, 1] - r['p_data'][:, 0], 1000 * (
                        r['p_data'][:, 0] - r['t_saccade_started']), r['t_sacc'], kind='linear')),
            # rotation of head rigidbody
            ('R_head_world', lambda r: r['helmet'].solve(r['o_data_upsampled'].reshape((-1, 4, 3)))[0]),
            # yaw pitch roll head rigidbody
            ('ypr_head_world', lambda r: fh.to_yawpitchroll(r['R_head_world'])),
            # reference positions of head rigidbody
            ('Ts_head_world', lambda r: r['helmet'].solve(r['o_data_upsampled'].reshape((-1, 4, 3)))[1]),
            # position of fixation led
            ('fixation_pos', lambda r: r['rig'][r['fixation_led'], :]),
            # position of target led
            ('target_pos', lambda r: r['rig'][r['target_led'], :]),
            # position of shifted target led
            ('starget_pos', lambda r: r['rig'][r['starget_led'], :]),
            # vector from eye to target position
            ('eye_to_fixation', lambda r: fh.to_unit(r['fixation_pos'] - r['Ts_head_world'][:, 3, :])),
            # vector from eye to target position
            ('eye_to_target', lambda r: fh.to_unit(r['target_pos'] - r['Ts_head_world'][:, 3, :])),
            # vector from eye to shifted target position
            ('eye_to_starget', lambda r: fh.to_unit(r['starget_pos'] - r['Ts_head_world'][:, 3, :])),
            # gaze vector in head without distortion correction
            ('gaze_in_head_distorted', lambda r: (r['R_eye_head'] @ r['p_data_upsampled'].T).T),
            # gaze vector in head with distortion correction
            ('gaze_in_head',
             lambda r: fh.normals_nonlinear_angular_transform(r['gaze_in_head_distorted'], r['nonlinear_parameters'])),
            # gaze angles in head
            ('gaze_in_head_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['gaze_in_head']))),
            # gaze vector in world
            ('gaze_in_world', lambda r: np.einsum('tij,tj->ti', r['R_head_world'], r['gaze_in_head'])),
            # gaze angles in world
            ('gaze_in_world_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['gaze_in_world']))),
            # angles from eye to target in world
            ('eye_to_target_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['eye_to_target']))),
            # difference of eye to target angles and gaze in world
            ('gaze_angle_vs_target', lambda r: r['gaze_in_world_ang'] - r['eye_to_target_ang']),
            # angles from eye to shifted target in world
            ('eye_to_starget_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['eye_to_starget']))),
            # difference of eye to shifted target angles and gaze in world
            ('gaze_angle_vs_starget', lambda r: r['gaze_in_world_ang'] - r['eye_to_starget_ang']),
            # angles from eye to fixation in world
            ('eye_to_fixation_ang', lambda r: np.rad2deg(fh.to_azim_elev(r['eye_to_fixation']))),
            # difference of eye to fixation angles and gaze in world
            ('gaze_angle_vs_fixation', lambda r: r['gaze_in_world_ang'] - r['eye_to_fixation_ang']),
            # time steps
            ('dt', lambda r: fh.padded_diff(r['t_sacc'])),
            # velocity of difference of eye to target angles and gaze in world
            ('gaze_angvel_vs_target', lambda r: fh.padded_diff(r['gaze_angle_vs_target']) / r['dt'][:, None]),
            ('gaze_angvel_vs_target_savgol',
                lambda r: prepend_nan(
                    savgol_filter(r['gaze_angvel_vs_target'][1:, ...], 3, 1, axis=0),
                    axis=0)),
            # saccade detection engbert & mergenthaler
            ('eng_merg', lambda r: fh.sacc_dec_engb_merg_horizontal(r['gaze_angle_vs_target'][:, 0],
                                                                    r['gaze_angvel_vs_target_savgol'][:, 0], 6, 5)),
        ]),
        add_inplace=True,
        print_log=True
    )

    df.drop(
        columns=[
            'p_data',
            'o_data',
            # 'helmet',
            'o_data_upsampled',
            'p_data_upsampled',
            'gaze_in_head_distorted',
        ],
        inplace=True
    )

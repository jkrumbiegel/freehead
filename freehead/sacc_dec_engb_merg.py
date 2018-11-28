import numpy as np
from scipy.signal import savgol_filter
from numba import jit

def sacc_dec_engb_merg(xy, vel, VFAC, MINDUR):

    #vel = vecvel(xy, SAMPLING, smoothing=smoothing)

    msd = np.sqrt(
        np.nanmedian(vel ** 2, axis=0) -
        np.nanmedian(vel, axis=0) ** 2)

    # print(msd)

    if np.any(msd < 1e-16):
        raise Exception('Threshold is effectively zero.')

    radius = msd * VFAC

    # x entries
    criterion_fulfilled = np.where(np.sum((vel / radius) ** 2, axis=1) > 1)[0]

    # print(criterion_fulfilled)

    # x-1 entries
    index_diffs = np.diff(criterion_fulfilled)
    # print(index_diffs)
    # x-2 entries
    boundaries = np.where(index_diffs > 1)[0]

    # print(boundaries)

    starts = np.concatenate(([0], boundaries + 1))
    ends_inclusive = np.concatenate((boundaries, [criterion_fulfilled.size - 1]))

    # print(starts)
    # print(ends_inclusive)

    start_end_pairs = np.vstack(
        (
            criterion_fulfilled[starts],
            criterion_fulfilled[ends_inclusive]
        )).T

    # print(start_end_pairs)

    durations = start_end_pairs[:, 1] - start_end_pairs[:, 0]

    long_enough_pairs = start_end_pairs[durations >= MINDUR, :]
    n_saccades = long_enough_pairs.shape[0]

    vpeaks = np.empty((n_saccades,), dtype=np.float64)
    start_end_pos = np.empty((n_saccades, 2), dtype=np.float64)
    amplitudes = np.empty((n_saccades, 2), dtype=np.float64)

    for i in range(n_saccades):
        a = long_enough_pairs[i, 0]
        b = long_enough_pairs[i, 1]

        vpeak = np.max(
            np.sqrt(
                np.sum(
                    vel[a: b, :] ** 2,
                    axis=1)))
        vpeaks[i] = vpeak

        start_end_pos[i, :] = xy[b, :] - xy[a, :]

        minxy = np.min(xy[a:b, :], axis=0)
        argminxy = np.argmin(xy[a:b, :], axis=0)
        maxxy = np.max(xy[a:b, :], axis=0)
        argmaxxy = np.argmax(xy[a:b, :], axis=0)

        amplitudes[i, :] = np.sign(argmaxxy - argminxy) * (maxxy - minxy)

    return long_enough_pairs, vpeaks, start_end_pos, amplitudes


def vecvel(xy, SAMPLING, smoothing=None):

    if smoothing is None:
        return SAMPLING * np.concatenate((np.zeros((1, 2)), np.diff(xy, axis=0)))
    elif isinstance(smoothing, dict):
        if smoothing['type'] == 'savgol':
            raw = SAMPLING * np.concatenate((np.zeros((1, 2)), np.diff(xy, axis=0)))
            return savgol_filter(
                raw,
                smoothing['window'],
                smoothing['poly_order'],
                mode='mirror',
                axis=0)

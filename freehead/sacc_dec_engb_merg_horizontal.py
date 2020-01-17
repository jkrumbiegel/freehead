import numpy as np
from scipy.signal import savgol_filter
from numba import jit

def sacc_dec_engb_merg_horizontal(x, vel, VFAC, MINDUR):

    try:
        msd = np.sqrt(
            np.nanmedian(vel ** 2) -
            np.nanmedian(vel) ** 2)

        if np.any(msd < 1e-16):
            return None

        threshold = msd * VFAC

        # x entries
        criterion_fulfilled = np.where(np.abs((vel / threshold)) > 1)[0]

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

        vpeaks = np.full((n_saccades,), np.nan)
        start_end_pos = np.full((n_saccades, ), np.nan)
        amplitudes = np.full((n_saccades, ), np.nan)

        for i in range(n_saccades):
            a = long_enough_pairs[i, 0]
            b = long_enough_pairs[i, 1]

            arg_vpeak = np.nanargmax(np.abs(vel[a: b]))
            vpeak = vel[a: b][arg_vpeak]
            vpeaks[i] = vpeak

            start_end_pos[i] = x[b] - x[a]

            minx = np.nanmin(x[a:b])
            argminx = np.nanargmin(x[a:b])
            maxx = np.nanmax(x[a:b])
            argmaxx = np.nanargmax(x[a:b])

            amplitudes[i] = np.sign(argmaxx - argminx) * (maxx - minx)

        if len(vpeaks) == 0:
            return None
        else:
            return long_enough_pairs, vpeaks, start_end_pos, amplitudes
    except:
        print('Engbert Mergenthaler failed')
        return None



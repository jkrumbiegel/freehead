import numpy as np
from scipy.linalg import orthogonal_procrustes
import freehead as fh
import pickle
import os


class Rigidbody:

    def __init__(self, markers: np.ndarray, ref_points=None):
        assert(markers.ndim == 2 and markers.shape[1] == 3)
        assert(not np.any(np.isnan(markers)))

        self.reference_markers = markers
        if ref_points is None:
            ref_points = np.mean(markers, axis=0)[None, :]  # use centroid as reference point if none is provided
        elif not isinstance(ref_points, np.ndarray):
            raise Exception('Reference points need to be a numpy array.')
        elif ref_points.shape == (3,):
            ref_points = ref_points[None, :]
        self.ref_points = ref_points

    def solve(self, markers):
        if markers.ndim == 2:
            rotation, ref_points = self._solve_multiple(markers[None, :, :])
            return rotation.reshape((3, 3)), ref_points.reshape((-1, 3))
        elif markers.ndim == 3:
            return self._solve_multiple(markers)

    def _solve_multiple(self, markers):
        # check which markers have nan values
        # markers: N x M x 3
        which_not_nan = ~np.any(np.isnan(markers), axis=2)
        # find unique marker combination rows
        unique, indices = np.unique(which_not_nan, axis=0, return_inverse=True)

        # prepare result array
        n = markers.shape[0]
        if self.ref_points.ndim != 2:
            raise Exception('Wrong dimensionality of reference points saved in rigidbody. Should have two dimensions, has ' + self.ref_points.ndim)
        p = self.ref_points.shape[0]
        result_rotations = np.full((n, 3, 3), np.nan)
        result_ref_points = np.full((n, p, 3), np.nan)

        # loop through unique marker combinations
        for unique_index in range(unique.shape[0]):
            # do vectorized version for each combination
            unique_combination = unique[unique_index, :]
            if unique_combination.sum() < 3:
                # ignore combinations of fewer than 3 valid markers
                continue

            unique_rows = indices == unique_index
            valid_marker_rows = unique[unique_index]
            # index markers from rows with current unique marker combination

            # make a multidimensional index array for getting the wanted rows
            index_mask = np.repeat(
                np.logical_and(unique_rows[:, None, None], valid_marker_rows[None, :, None]),
                3,
                axis=2)
            valid_markers = markers[index_mask].reshape((unique_rows.sum(), -1, 3))  # Nu x Mu x 3
            reference_subset = self.reference_markers[valid_marker_rows, :] # Mu x 3

            valid_markers_centroid = valid_markers.mean(axis=1)  # Nu x 3
            valid_markers_centered = valid_markers - valid_markers_centroid[:, None, :]  # Nu x Mu x 3

            reference_subset_centroid = reference_subset.mean(axis=0)  # 3
            reference_subset_centered = reference_subset - reference_subset_centroid  # Mu x 3

            ref_center_to_ref_points = self.ref_points - reference_subset_centroid  # P x 3 - 3 = P x 3

            rotations = fh.multidim_ortho_procrustes(valid_markers_centered, reference_subset_centered[None, :, :])  # Nu x 3 x 3
            ref_points_translated = np.einsum('nij,tj->nti', rotations, ref_center_to_ref_points) + valid_markers_centroid[:, None, :]  # Nu x P x 3

            result_rotations[unique_rows, :, :] = rotations
            result_ref_points[unique_rows, :, :] = ref_points_translated

        return result_rotations, result_ref_points


class FourMarkerProbe(Rigidbody):
    def __init__(self):
        calib_file_path = os.path.join(os.path.dirname(__file__), '../datafiles/four_marker_probe_calibrated.pickle')
        calibration = pickle.load(open(calib_file_path, 'rb' ))
        super(FourMarkerProbe, self).__init__(calibration['markers'], ref_points=calibration['ref_point'])
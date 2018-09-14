import numpy as np
from scipy.linalg import orthogonal_procrustes


class Rigidbody():

    def __init__(self, markers: np.ndarray, ref_points=None):
        assert(markers.ndim == 2 and markers.shape[1] == 3)
        assert(not np.any(np.isnan(markers)))

        self.reference_markers = markers
        if ref_points is None:
            ref_points = np.mean(markers, axis=0)[None, :]  # use centroid as reference point if none is provided
        self.ref_points = ref_points

    def solve(self, markers):
        if markers.ndim == 2:
            return self._solve_single(markers)
        elif markers.ndim == 3:
            return self._solve_multiple(markers)

    def _solve_single(self, markers):
        # if fewer than three markers are available, procrustes can't work
        valid_marker_rows = ~np.any(np.isnan(markers), axis=1)
        if np.sum(valid_marker_rows) < 3:
            return None, None

        valid_markers = markers[valid_marker_rows, :]
        reference_subset = self.reference_markers[valid_marker_rows, :]

        valid_markers_centroid = valid_markers.mean(axis=0)
        print('valid_markers_centroid\n', valid_markers_centroid)
        valid_markers_centered = valid_markers - valid_markers_centroid
        print('valid_markers_centered\n', valid_markers_centered)

        reference_subset_centroid = reference_subset.mean(axis=0)
        print('reference_subset_centroid\n', reference_subset_centroid)

        reference_subset_centered = reference_subset - reference_subset_centroid
        print('reference_subset_centered\n', reference_subset_centered)

        centroid_distance = valid_markers_centroid - reference_subset_centroid
        print('centroid distance\n', centroid_distance)
        ref_centroid_to_ref_points = self.ref_points - reference_subset_centroid  # N x 3
        print('ref_centroid_to_ref_points\n', ref_centroid_to_ref_points)

        rotation = orthogonal_procrustes(reference_subset_centered, valid_markers_centered)[0]
        ref_points_translated = np.einsum('ij,tj->ti', rotation, ref_centroid_to_ref_points) + valid_markers_centroid
        print('ref_points_translated\n', ref_points_translated)

        return rotation, ref_points_translated


    def _solve_multiple(self, markers):
        pass

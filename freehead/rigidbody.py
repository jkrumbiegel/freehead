import numpy as np
from scipy.linalg import orthogonal_procrustes


class Rigidbody():

    def __init__(self, markers: np.ndarray, origin=None):
        assert(markers.ndim == 2 and markers.shape[1] == 3)
        assert(not np.any(np.isnan(markers)))

        self.reference_markers = markers
        if origin is None:
            origin = np.mean(markers, axis=0)  # use centroid as origin if none is provided
        self.origin = origin
        self.origin_distances = origin - markers

    def solve(self, markers):
        if markers.ndim == 2:
            return self._solve_single(markers)
        elif markers.ndim == 3:
            return self._solve_multiple(markers)

    def _solve_single(self, markers):
        # if fewer than three markers are available, procrustes can't work
        valid_marker_rows = ~np.any(np.isnan(markers), axis=1)
        if np.sum(valid_marker_rows) < 3:
            return None

        valid_markers = markers[valid_marker_rows, :]
        reference_subset = self.reference_markers[valid_marker_rows, :]

        valid_markers_centered = valid_markers - valid_markers.mean()
        reference_subset_centered = reference_subset - reference_subset.mean()

        rotation = orthogonal_procrustes(valid_markers_centered, reference_subset_centered)
        return rotation


    def _solve_multiple(self, markers):
        pass

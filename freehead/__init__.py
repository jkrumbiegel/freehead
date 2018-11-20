import logging
import sys
import os
from .PupilThread import PupilThread
from .OptotrakThread import OptotrakThread
from .ArduinoThread import ArduinoThread
from .wait_for_keypress import wait_for_keypress
from .u_theta import u_theta
from .from_yawpitchroll import from_yawpitchroll
from .to_yawpitchroll import to_yawpitchroll
from .get_rig_transform import get_rig_transform
from .constants import LED_POSITIONS
from .to_unit import to_unit
from .markers_to_ortho import markers_to_ortho
from .calibrate_pupil import *
from .rigidbody import Rigidbody, FourMarkerProbe, LedRig
from .is_rotation_matrix import is_rotation_matrix
from .tup3d import tup3d
from .multidim_ortho_procrustes import multidim_ortho_procrustes
from .anynan import anynan
from .LedShiftExperiment import LedShiftExperiment
from .qplot3d import qplot3d
from .create_trial_frame import create_trial_frame
from .normals_nonlinear_angular_transform import normals_nonlinear_angular_transform, normals_nonlinear_transform
from .was_key_pressed import was_key_pressed
from .expand_df_arrays import expand_df_arrays
from .array_apply import array_apply
from .padded_diff import padded_diff
from .sacc_dec_engb_merg import sacc_dec_engb_merg
from .to_azim_elev import to_azim_elev
from .interpolate_a_onto_b_time import interpolate_a_onto_b_time
from .save_experiment_files import save_experiment_files

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

root = logging.getLogger('freehead')
if not root.handlers:
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)



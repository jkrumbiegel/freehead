import logging
import sys
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
from .save_experiment_files import save_experiment_files
from .focus_pygame_window import focus_pygame_window

root = logging.getLogger('freehead')
if not root.handlers:
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)



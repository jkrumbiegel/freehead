import logging
import sys
from .PupilThread import PupilThread
from .OptotrakThread import OptotrakThread
from .wait_for_keypress import wait_for_keypress
from .u_theta import u_theta
from .yawpitchroll import yawpitchroll
from .get_rig_transform import get_rig_transform
from .constants import LED_POSITIONS, FOUR_MARKER_PROBE
from .to_unit import to_unit
from .markers_to_ortho import markers_to_ortho

root = logging.getLogger('freehead')
if not root.handlers:
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)



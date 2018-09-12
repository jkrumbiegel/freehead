import logging
import sys
from .PupilThread import PupilThread
from .OptotrakThread import OptotrakThread
from .wait_for_keypress import wait_for_keypress


root = logging.getLogger('freehead')
if not root.handlers:
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)



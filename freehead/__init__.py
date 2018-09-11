import logging
import sys
from .PupilThread import PupilThread
from .OptotrakThread import OptotrakThread


root = logging.getLogger('freehead')
if not root.handlers:
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)



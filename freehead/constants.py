import os
import numpy as np


four_marker_probe_path = os.path.join(os.path.dirname(__file__), '../datafiles/four_marker_probe.npy')
FOUR_MARKER_PROBE = np.load(four_marker_probe_path)

led_positions_path = os.path.join(os.path.dirname(__file__), '../datafiles/led_positions.npy')
LED_POSITIONS = np.load(led_positions_path)
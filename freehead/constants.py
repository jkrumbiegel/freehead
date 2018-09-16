import os
import numpy as np


led_positions_path = os.path.join(os.path.dirname(__file__), '../datafiles/led_positions.npy')
LED_POSITIONS = np.load(led_positions_path)
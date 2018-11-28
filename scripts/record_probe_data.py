import freehead as fh
import numpy as np
import pygame
import os
import pickle
from datetime import datetime
#%%
pygame.init()
pygame.display.set_mode((300, 300))

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

#%%
probe = fh.FourMarkerProbe()
fh.focus_pygame_window()


fh.wait_for_keypress(pygame.K_SPACE)
othread.reset_data_buffer()
fh.wait_for_keypress(pygame.K_SPACE)
data = othread.get_shortened_data()
othread.should_stop.set()
othread.join()

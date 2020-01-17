import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import freehead as fh
import pygame
import seaborn as sns

pygame.init()
pygame.display.set_mode((300, 300))

default_switch_interval = sys.getswitchinterval()
print(default_switch_interval)
sys.setswitchinterval(0.0001)

class SwitchThread(threading.Thread):

    def __init__(self):
        super(SwitchThread, self).__init__()
        self.should_stop = threading.Event()
        self.timestamps = []

    def run(self):
        while not self.should_stop.is_set():
            self.timestamps.append(time.monotonic())
            time.sleep(0.0005)


threads = [SwitchThread() for _ in range(3)]

fh.wait_for_keypress(pygame.K_SPACE)
for t in threads:
    t.start()

# time.sleep(1)
fh.wait_for_keypress(pygame.K_SPACE)
for t in threads:
    t.should_stop.set()

for t in threads:
    t.join()

pygame.quit()
sys.setswitchinterval(default_switch_interval)

timestamps = [np.array(t.timestamps) for t in threads]
dts = [fh.padded_diff(t) for t in timestamps]

colors = sns.color_palette('husl', len(threads))
for t, dt, c in zip(timestamps, dts, colors):
    plt.plot(t, dt, 'o', linewidth=0.75, markersize=3, markerfacecolor=(0, 0, 0, 0), markeredgecolor=c, markeredgewidth=0.25)

plt.show()
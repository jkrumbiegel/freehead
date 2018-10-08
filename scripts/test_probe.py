import freehead as fh
import numpy as np
import time

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

probe = fh.FourMarkerProbe()

start_time = time.monotonic()
duration = 60

last_i = None
while time.monotonic() - start_time < duration:
    current_i = othread.i_current_sample
    if current_i == last_i:
        time.sleep(0.3)
    last_i = current_i
    
    probe_data = othread.current_sample[15:27].reshape(4, 3)
    rotation, tip = probe.solve(probe_data)
    
    print(f'x: {tip[0,0]:4.2f} | y: {tip[0,1]:4.2f} | z: {tip[0,2]:4.2f}')
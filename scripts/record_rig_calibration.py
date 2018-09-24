# record rigidbody leds as calibration
import freehead as fh
import numpy as np
import pygame

pygame.init()
pygame.display.set_mode((300, 300))

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

athread = fh.ArduinoThread()
athread.start()
athread.started_running.wait()

probe = fh.FourMarkerProbe()

led_positions = np.full((255, 3), np.nan)

for i in range(255):
    athread.write_uint8(i, 100, 0, 0)
    while True:
        fh.wait_for_keypress(pygame.K_SPACE)
        probe_rotation, probe_tip = probe.solve(othread.current_sample.copy()[15:27].reshape(4,3))
        if fh.anynan(probe_rotation):
            continue
        led_positions[i, :] = probe_tip[:]
        break
        
athread.write_uint8(255, 0, 0, 0)

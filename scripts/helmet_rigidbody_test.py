import freehead as fh
import time
import numpy as np
import matplotlib.pyplot as plt
import pygame
import serial
import struct


arduino = serial.Serial('/dev/ttyUSB0', baudrate=115200)

pygame.init()
pygame.display.set_mode((300, 300))

othread = fh.OptotrakThread()
othread.start()
othread.started_running.wait()

probe = fh.FourMarkerProbe()

def extract_probe(ot_data):
    return ot_data[..., 15:27].reshape((-1, 4, 3)).squeeze()

def extract_helmet(ot_data):
    return ot_data[..., 3:15].reshape((-1, 4, 3)).squeeze()


if not 'rig_led_positions' in locals():
    led_rig_indices = [0, 70, 141, 213, 254] # these are the array indices
    probe_tips = np.full((len(led_rig_indices), 3), np.nan)
    for i, led_rig_index in enumerate(led_rig_indices):
        arduino.write(struct.pack('>B', led_rig_index))
        print('Press space to record led with index', led_rig_index)
        while True:
            fh.wait_for_keypress(pygame.K_SPACE)
            current_sample = othread.current_sample.copy()
            rotation, tip = probe.solve(extract_probe(current_sample))
            if np.any(np.isnan(tip)):
                print('Probe not visible, try again.')
                continue
            probe_tips[i, :] = tip
            break
    arduino.write(struct.pack('>B', 255))
    
    led_rig_transform_result = fh.get_rig_transform(probe_tips, led_rig_indices)
    
    R_rig = fh.from_yawpitchroll(led_rig_transform_result.x[0:3])
    T_rig = led_rig_transform_result.x[3:6]
    
    rig_led_positions = np.einsum('ij,tj->ti', R_rig, fh.LED_POSITIONS) + T_rig
else:
    print('rig_led_positions calibration already present. Delete if you want to recalibrate.')


head_measurement_points = [
        'head straight',
        'nasion',
        'inion',
        'right ear',
        'left ear']

for i, measurement_point in enumerate(head_measurement_points):
    print('Press space to measure: ' + measurement_point)
    while True:
        fh.wait_for_keypress(pygame.K_SPACE)
        current_sample = othread.current_sample.copy()
        helmet_leds = extract_helmet(current_sample)
        
        if np.any(np.isnan(helmet_leds)):
            print('Helmet LEDs not all visible. Try again.')
            continue

        if i == 0:
            helmet = fh.Rigidbody(helmet_leds)
            break
        else:
            _, probe_tip = probe.solve(extract_probe(current_sample))
            if np.any(np.isnan(probe_tip)):
                print('Probe not visible. Try again.')
                continue
            helmet.add_reference_points(helmet_leds, probe_tip)
            break
        
start_time = time.monotonic()
duration = 60
last_i = None
while time.monotonic() - start_time < duration:
    current_i = othread.i_current_sample
    if current_i == last_i:
        time.sleep(0)
        continue
    last_i = current_i
    current_sample = othread.current_sample.copy()
    rotation, ref_points = helmet.solve(extract_helmet(current_sample))
    if np.any(np.isnan(rotation)):
        continue
    inion_to_nasion = fh.to_unit(ref_points[1, :] - ref_points[2, :]).squeeze()
    nasion_to_leds = fh.to_unit(rig_led_positions - ref_points[1, :])
    angles = np.rad2deg(np.arccos(np.einsum('li,i->l', nasion_to_leds, inion_to_nasion)))
    closest_led = int(np.argmin(angles))
    if angles.min() > 5:
        closest_led = 255
    arduino.write(struct.pack('>B', closest_led))
        
othread.should_stop.set()
othread.join()

pygame.quit()

arduino.write(struct.pack('>B', 255))
arduino.close()

print('done')
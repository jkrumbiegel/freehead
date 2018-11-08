import freehead as fh
import serial
import struct
import time
import itertools
import re
import matplotlib.pyplot as plt

lightmeter_port = '/dev/ttyACM0'
lightmeter = serial.Serial(lightmeter_port, timeout=0.5)

athread = fh.ArduinoThread()
athread.start()
athread.started_running.wait()

for i in range(4000):
    athread.write_uint8(255, 255, 255, 255)
    athread.write_uint8(255, 0, 0, 0)

time.sleep(5)

def to_bytes(*args):
    # ints or chars allowed
    args = [arg if isinstance(arg, str) else [arg] for arg in args]
    ints = [arg if isinstance(arg, int) else ord(arg) for arg in itertools.chain.from_iterable(args)]
    return struct.pack(f'>{len(ints)}B', *ints)


# show available commands
lightmeter.write(to_bytes('?\r'))
time.sleep(1)
response = lightmeter.read_all()
print(response.decode('utf-8'))

try:
    lightmeter.write(to_bytes('T1\r'))
    time.sleep(0.5)
    print('Set trigger mode\n', lightmeter.read_all().decode('utf-8'))
     
    # set sweep length
    lightmeter.write(to_bytes('L1000\r'))
    time.sleep(0.5)
    print('Set sweep length\n', lightmeter.read_all().decode('utf-8'))
    
    
    # set sample period in microseconds, min: 5
    lightmeter.write(to_bytes('P100\r'))
    time.sleep(0.5)
    print('Set sample period\n', lightmeter.read_all().decode('utf-8'))
    time.sleep(1)
    
    # start recording
    lightmeter.write(to_bytes('A\r'))
    print('Start recording\n',lightmeter.read_until(size=6).decode('utf-8'))
    
    #athread.write_uint8(255, 255, 0, 0)
    
    # loop until data is ready
    def get_status():
        lightmeter.write(to_bytes('S\r'))
        response = lightmeter.read_until(size=9).decode('utf-8')
        return response[-2]
    
    while get_status() != '3':
        time.sleep(0)
    print('Recording finished')
    
    
    lightmeter.write(to_bytes('D\r'))
    result = b''
    while True:
        # because of low input buffer size (1020 bytes) need to read as fast as
        # possible and time out when no more data is being sent
        # timeout length is set at serial port creation
        new_byte = lightmeter.read()
        if new_byte == b'':
            break
        result += new_byte
    
    result_numbers = [int(i) for i in re.sub('[^0-9,]', '', result.decode('utf-8')).split(',')]
    
    diode_1, diode_2 = result_numbers[::2], result_numbers[1::2]
    lightmeter.close()
    athread.should_stop.set()
    
    plt.plot(diode_1, label='diode 1')
    plt.plot(diode_2, label='diode 2')
    plt.legend()

except:
    lightmeter.flushInput()
    lightmeter.close()
    athread.should_stop.set()
    raise

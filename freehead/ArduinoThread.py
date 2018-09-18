import serial
import struct
import threading
from collections import deque
import time


class ArduinoThread(threading.Thread):

    def __init__(self, path='dev/ttyUSB0', baudrate=9600):
        super(ArduinoThread, self).__init__()
        self.should_stop = threading.Event()
        self.arduino = serial.Serial(path, baudrate=baudrate)
        self.queue = deque([])

    def run(self):
        while not self.should_stop.is_set():
            # first a ready signal needs to be read
            self.arduino.read()
            # then a message can be sent
            while True:
                if len(self.queue):
                    self.arduino.write(self.queue.popleft())
                    break
                time.sleep(0)

        self.arduino.close()

    def write_uint8(self, *ints):
        if not all([isinstance(part, int) and (0 >= part <= 255) for part in ints]):
            raise ValueError('Message needs to be a tuple with ints between 0 and 255')
        byte_message = struct.pack(f'>{len(ints)}B', *ints)
        self.queue.append(byte_message)

import serial
import struct
import threading
from collections import deque
import time
import logging
import numbers

logger = logging.getLogger(__name__)


class ArduinoThread(threading.Thread):

    def __init__(self, path='/dev/ttyUSB0', baudrate=115200):
        super(ArduinoThread, self).__init__()
        self.daemon = True
        self.should_stop = threading.Event()
        logger.info(f'Establishing connection to arduino at "{path}" with baudrate {baudrate}.')
        self.arduino = serial.Serial(path, baudrate=baudrate, timeout=1)
        self.started_running = threading.Event()
        self.queue = deque([])

    def run(self):
        logger.info('Pausing shortly for arduino start-up.')
        time.sleep(2)
        logger.info('Arduino ready.')
        self.started_running.set()
        while not self.should_stop.is_set():

            # a message can be sent
            while not self.should_stop.is_set():
                if len(self.queue):
                    self.arduino.write(self.queue.popleft())
                    break
                time.sleep(0)
            # but before the next one a ready signal needs to be read
            while not self.should_stop.is_set() and self.arduino.read() == b'':
                pass

        logger.info('Closing arduino.')
        self.arduino.close()

    def write_uint8(self, *ints):
        if not all([isinstance(part, numbers.Integral) and (0 <= part <= 255) for part in ints]):
            raise ValueError('Message needs to be a tuple with ints between 0 and 255')
        byte_message = struct.pack(f'>{len(ints)}B', *ints)
        self.queue.append(byte_message)
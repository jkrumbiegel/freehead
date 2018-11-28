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

        self.command_index = -1
        self.command_timestamps = []

    def run(self):
        logger.info('Pausing shortly for arduino start-up.')
        time.sleep(2)
        logger.info('Arduino ready.')

        self.reset_command_timestamps()
        self.started_running.set()

        while not self.should_stop.is_set():

            if len(self.queue):
                self.arduino.write(self.queue.popleft())
                logger.info('Command sent')

                # before the next one a ready signal needs to be read
                while not self.should_stop.is_set() and self.arduino.read() == b'':
                    time.sleep(0)

                # when the return message has been received, a timestamp is appended to be retrieved later
                self.command_timestamps.append(time.monotonic())

                # and the next message can be retrieved
                continue

            # if there is no command waiting, control is given away if possible
            time.sleep(0)

        logger.info('Closing arduino.')
        self.arduino.close()

    def write_uint8(self, *ints) -> int:
        if not all([isinstance(part, numbers.Integral) and (0 <= part <= 255) for part in ints]):
            raise ValueError('Message needs to be a tuple with ints between 0 and 255')
        byte_message = struct.pack(f'>{len(ints)}B', *ints)
        self.queue.append(byte_message)

        # with the command index, it will later be possible to retrieve the end timestamp of the led change
        self.command_index += 1
        return self.command_index

    def reset_command_timestamps(self):

        logger.info('Waiting for remaining commands to finish...')
        while len(self.command_timestamps) < (self.command_index + 1):  # index 0 needs 1 timestamp, 1 needs 2 etc.
            time.sleep(0)

        self.command_timestamps = []
        self.command_index = -1
        logger.info('Command timestamps reset.')



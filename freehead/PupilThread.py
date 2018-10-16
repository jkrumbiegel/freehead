import threading
import numpy as np
import pandas as pd
import zmq
import time
import msgpack
import logging

logger = logging.getLogger(__name__)

T_PUPIL = 't_pupil'
T_SYS_REL = 't_sys_rel'
T_SYS_ABS = 't_sys_abs'
NORM_X = 'norm_x'
NORM_Y = 'norm_y'
NORM_Z = 'norm_z'
CONFIDENCE = 'confidence'


class PupilThread(threading.Thread):

    address = '127.0.0.1'
    request_port = '50020'

    last_sync_time = None

    buffer_length = 120 * 60 * 10  # standard number of time steps in buffer
    sample_size = 7  # pupil time, system time corrected, system time, gaze normal x, y, z, confidence
    sample_components = [T_PUPIL, T_SYS_REL, T_SYS_ABS, NORM_X, NORM_Y, NORM_Z, CONFIDENCE]

    data = None
    i_current_sample = 0
    current_sample = None
    buffer_limit_reached = False

    def __init__(self):
        super(PupilThread, self).__init__()
        self.daemon = True
        self.context = zmq.Context()

        # open a request socket for general communication
        logger.info('Opening a request socket to pupil service. If this hangs, check that pupil service is running.')
        self.request_socket = self.context.socket(zmq.REQ)
        self.request_socket.connect("tcp://{}:{}".format(self.address, self.request_port))
        logger.info('Request socket connected.')

        # ask for the sub port
        self.request_socket.send_string('SUB_PORT')
        self.sub_port = self.request_socket.recv_string()

        self.sub_socket = self.context.socket(zmq.SUB)

        # define thread synchronization events
        self.should_stop = threading.Event()
        self.started_running = threading.Event()
        self.recording_allowed = threading.Event()
        self.requesting_data_reset = threading.Event()
        self.reset_request_received = threading.Event()

        # set the pupil timer to 0
        self.synchronize_time()

    def run(self):

        self.reset_data_buffer()

        # listen to pupil data on the sub port
        logger.info('Connecting subscription socket and subscribing to pupil data...')
        self.sub_socket.connect("tcp://{}:{}".format(self.address, self.sub_port))
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, 'pupil.')
        logger.info('Success.')

        self.started_running.set()

        while not self.should_stop.is_set():

            if self.requesting_data_reset.is_set():
                self.reset_request_received.set()
                logger.debug('Waiting for recording allowance')
                self.recording_allowed.wait()
                logger.debug('Recording allowance received. Continuing.')

            if self.i_current_sample == self.buffer_length - 1:
                self.buffer_limit_reached = True
                logger.warning('Buffer limit reached, all samples until reset will be lost.')

            # receive samples even if buffer limit reached, so the socket doesn't fill up
            try:
                topic = self.sub_socket.recv_string()
                message = self.sub_socket.recv()
                system_time_received = time.monotonic()
                msg = msgpack.loads(message, encoding='utf-8')
                self.current_sample = np.array([
                    msg['timestamp'],
                    system_time_received - self.last_sync_time,
                    system_time_received,
                    msg['circle_3d']['normal'][0],
                    msg['circle_3d']['normal'][1],
                    msg['circle_3d']['normal'][2],
                    msg['confidence']
                ])
            except KeyboardInterrupt:
                logger.warning('Keyboard Interrupt detected, closing...')
                break

            if not self.buffer_limit_reached:
                self.data[self.i_current_sample, :] = self.current_sample

            self.i_current_sample += 1

        # after loop, close sockets and context
        self.cleanup()

    def synchronize_time(self, t="0.0"):
        logger.info('Sending pupil sync signal for T= ' + t + '.')
        self.request_socket.send_string('T ' + t)
        self.last_sync_time = time.perf_counter()
        self.request_socket.recv_string()
        logger.info('Sync successful.')

    def reset_data_buffer(self):
        if self.started_running.is_set():
            self.recording_allowed.clear()
            self.requesting_data_reset.set()
            logger.debug('Waiting for reset request received signal.')
            self.reset_request_received.wait()
            # now the data gathering loop should wait for allowance, the data array can be reset

        data_array = np.full((self.buffer_length, self.sample_size), np.nan, dtype=np.float)
        self.data = data_array
        self.i_current_sample = 0
        self.buffer_limit_reached = False
        logger.debug('Successfully reset data array.')

        self.reset_request_received.clear()
        self.requesting_data_reset.clear()
        self.recording_allowed.set()

    def cleanup(self):
        self.request_socket.close()
        self.sub_socket.close()
        self.context.term()
        logger.info('ZMQ context terminated gracefully.')

    def get_shortened_data(self):
        if self.data is None:
            return None
        # check if data has been written to the last sample of the row
        # otherwise we're currently filling this time step and return the last complete
        if self.data[self.i_current_sample, -1] == np.nan:
            return self.data[0:self.i_current_sample - 1, :]
        else:
            return self.data[0:self.i_current_sample, :]

    def get_last_angular_velocity(self):
        n_samples = self.i_current_sample
        if n_samples > 1:
            angular_velocity = np.rad2deg(np.arccos(np.dot(
                self.data.values[n_samples - 1, 3:6],
                self.data.values[n_samples - 2, 3:6]))) * 120
            return angular_velocity
        else:
            return 0

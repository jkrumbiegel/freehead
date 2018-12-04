import threading
import numpy as np
import pandas as pd
import zmq
import time
import msgpack
import logging

logger = logging.getLogger(__name__)


class PupilThread(threading.Thread):

    address = '127.0.0.1'
    request_port = '50020'

    buffer_length = 200 * 60 * 10 # standard number of time steps in buffer (10 minutes)
    sample_size = 9  # pupil time, system time receipt, gaze normal x, y, z, confidence, eyecenter x, y, z

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

        # no need for synchronization now that pupil sends only time.monotonic()
        # self.synchronize_time()

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
                    system_time_received,
                    msg['circle_3d']['normal'][0],
                    msg['circle_3d']['normal'][1],
                    msg['circle_3d']['normal'][2],
                    msg['confidence'],
                    msg['sphere']['center'][0],
                    msg['sphere']['center'][1],
                    msg['sphere']['center'][2],
                ])
            except KeyboardInterrupt:
                logger.warning('Keyboard Interrupt detected, closing...')
                break

            if not self.buffer_limit_reached:
                self.data[self.i_current_sample, :] = self.current_sample

            self.i_current_sample += 1
            time.sleep(0.001)

        # after loop, close sockets and context
        self.cleanup()

    def synchronize_time(self):
        t = time.monotonic()
        self.request_socket.send_string(f'T {t}')
        logger.info(f'Sent pupil sync signal for T= {t}.')
        self.request_socket.recv_string()
        logger.info('Sync successful.')

    def reset_3d_eye_model(self):
        self.request_socket.send_string('notify.detector3d.reset_model', flags=zmq.SNDMORE)
        self.request_socket.send(msgpack.dumps({
            'subject': 'detector3d.reset_model'
        }))
        return self.request_socket.recv_string()

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


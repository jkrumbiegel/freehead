import threading
import numpy as np
import pandas as pd
import time
import logging
import pylsl
import yaml
from copy import deepcopy as dcopy


logger = logging.getLogger(__name__)

config_outlet_info = {
    'name': 'Client',
    'type': 'String',
    'channel_count': 1,
    'nominal_srate': 1000,
    'channel_format': pylsl.cf_string,
    'source_id': 'ID1'
}

default_client_config = {
    'lsl': {
        'start_code': 0,  # write this to stream in order to request a sample
        'stop_code': 1,  # write this to stop the Collection on the server
        'pause_code': 2,
        'outlet': {
            'name': 'ExperimentStream',  # outgoing data from Experiment host
            'type': 'Experiment',  #
            'channel_count': 1,  # we just need one channel with a string
            'nominal_srate': 150,
            'channel_format': pylsl.cf_float32,
            'source_id': 'myuniquesourceid2'
        },
        'inlet':{
            'name': 'OptotrakStream',  # incoming data from Optotrak host
            'connection_timeout': 100  # seconds
        }
    },
     # datapixx
    'dp': {
        'start_collection_wave': {
            'trigger_pin' : 6,  # serial pin 7 on the optotrak scu goes to parallel pin 4, which is digital out 6 on the datapixx
            'buffer_address' : 8e6,  # this is the address of the digital out buffer of datapixx
            'schedule_rate' : 0.001,
            'schedule_rate_mode' : 3,
            'n_wave_samples' : 2,
            'duration' : 1,
            'duration_mode' : 1,
            'register_write_mode' : 'RegWrRd'
        },
        'optotrak_clock_wave': {
            'trigger_pin' : 2,  # serial pin 3 on the optotrak scu goes to parallel pin 2, which is digital out 2 on the datapixx
            'buffer_address' : 8e6,  # this is the address of the digital out buffer of datapixx
            'schedule_rate': 15000,
            'schedule_rate_mode' : 1,
            'n_wave_samples' : 100,
            'duration' : 0,
            'duration_mode' : 0,
            'register_write_mode' : 'RegWrRd'
        }
    }
}


default_server_config = {
    'lsl': {
        'start_code': 0,  # write this to stream in order to request a sample
        'stop_code': 1,  # write this to stop the Collection on the server
        'pause_code': 2,
        'outlet': {
            'name': 'OptotrakStream',  # outgoing data from Experiment host
            'type': 'Optotrak',  #
            'n_channels': 30,
            'sampling_rate': 150,
            'format': 'cf_float32',
            'source_id': 'myuniquesourceid1'
        },
        'inlet': {
            'name': 'ExperimentStream',  # incoming data from Optotrak host
            'connection_timeout': 100  # seconds
        }
    },
    'optotrak': {
        'file_identifier': 'optotrak_pupil_1',
        'save_buffer_file': False,
        'cam_filename': 'Aligned20180906_1',
        'matlab_to_optotrak_mex_path': r'C:\Users\locallab\Documents\Visual Studio 2015\Projects\Matlab_to_Optotrak\Debug',
        'collection_num_markers_1': 4,  # 4 helmet
        'collection_num_markers_2': 4,  # 4 marker probe
        'collection_frequency': 120,
        'collection_duration': 600,
        'external_clock_yes': 0,
        'start_marker_3d': 1,
        'end_marker_3d': 9,
        'blocking': False
    }
}


class OptotrakThread(threading.Thread):

    current_sample = None
    current_timestamp = None

    data = None
    i_current_sample = 0
    buffer_length = 120 * 60 * 10
    buffer_limit_reached = False
    sample_size = None

    def __init__(self, server_config=None, client_config=None):
        super(OptotrakThread, self).__init__()
        self.config_outlet_streaminfo = pylsl.StreamInfo(**config_outlet_info)
        self.config_outlet = None

        # prepare the config dictionaries
        self.server_config = dcopy(default_server_config) if server_config is None else dcopy(default_server_config).update(server_config)
        self.client_config = dcopy(default_client_config) if client_config is None else dcopy(default_client_config).update(client_config)

        self.sample_size = self.server_config['lsl']['outlet']['n_channels'] + 1  # plus 1 for lsl timestamp

        self.control_outlet = None
        self.data_inlet = None

        self.should_stop = threading.Event()
        self.started_running = threading.Event()
        self.recording_allowed = threading.Event()
        self.requesting_data_reset = threading.Event()
        self.reset_request_received = threading.Event()

    def run(self):

        self.reset_data_buffer()

        self.send_server_config()
        self.create_control_outlet()
        self.create_data_inlet()

        logger.info('Sending start code via lsl.')
        self.control_outlet.push_sample([self.client_config['lsl']['start_code']])
        self.started_running.set()

        while not self.should_stop.is_set():

            if self.requesting_data_reset.is_set():
                self.reset_request_received.set()
                logger.info('Waiting for recording allowance')
                self.recording_allowed.wait()
                logger.info('Recording allowance received. Continuing.')

            if self.i_current_sample == self.buffer_length - 1:
                self.buffer_limit_reached = True
                logger.warning('Buffer limit reached, all samples until reset will be lost.')

            # receive samples even if buffer limit reached, so the socket doesn't fill up
            try:
                current_data, current_timestamp = self.data_inlet.pull_sample()
                current_data = np.array(current_data)
                current_data[current_data < -3.6e+28] = np.nan
                current_timestamp += self.data_inlet.time_correction();
                self.current_sample = np.concatenate((current_data, [current_timestamp]))
            except KeyboardInterrupt:
                logger.warning('Keyboard Interrupt detected, closing...')
                break

            if not self.buffer_limit_reached:
                self.data[self.i_current_sample, :] = self.current_sample

            self.i_current_sample += 1

        self.cleanup()

    def reset_data_buffer(self):

        if self.started_running.is_set():
            self.recording_allowed.clear()
            self.requesting_data_reset.set()
            logger.info('Waiting for reset request received signal.')
            self.reset_request_received.wait()
            # now the data gathering loop should wait for allowance, the data array can be reset

        data_array = np.full((self.buffer_length, self.sample_size), np.nan, dtype=np.float)
        self.data = data_array
        self.i_current_sample = 0
        self.buffer_limit_reached = False
        logger.info('Successfully reset data array.')

        self.reset_request_received.clear()
        self.requesting_data_reset.clear()
        self.recording_allowed.set()


    def create_data_inlet(self):
        logger.info('Trying to resolve data inlet stream.')
        streams = pylsl.resolve_stream('name', self.client_config['lsl']['inlet']['name'])
        self.data_inlet = pylsl.StreamInlet(streams[0])
        logger.info('Data inlet created successfully. Running first time correction so subsequent ones are instantaneous.')
        self.data_inlet.time_correction()
        logger.info('First time correction done.')

    def create_control_outlet(self):
        logger.info('Creating server control outlet')
        control_outlet_streaminfo = pylsl.StreamInfo(**(self.client_config['lsl']['outlet']))
        self.control_outlet = pylsl.StreamOutlet(control_outlet_streaminfo)

    def send_server_config(self):
        logger.info('Creating server configuration outlet.')
        self.config_outlet = pylsl.StreamOutlet(self.config_outlet_streaminfo)
        # sleep shortly to allow server inlet to resolve the stream and be ready
        time.sleep(2)
        config_string = padded_lsl_string(yaml.dump(self.server_config))
        logger.info('Pushing server configuration yaml string.')
        self.config_outlet.push_chunk(config_string)

    def get_shortened_data(self):
        if self.data is None:
            return None
        # check if data has been written to the last sample of the row
        # otherwise we're currently filling this time step and return the last complete
        if self.data[self.i_current_sample, -1] == np.nan:
            return self.data[0:self.i_current_sample, :]
        else:
            return self.data[0:self.i_current_sample + 1, :]

    def cleanup(self):
        logger.info('Pushing stop code to Optotrak server.')
        self.control_outlet.push_sample([self.client_config['lsl']['stop_code']])
        logger.info('Deleting config outlet.')
        del self.config_outlet
        logger.info('Deleting control outlet.')
        del self.control_outlet
        logger.info('Deleting data inlet.')
        del self.data_inlet

def padded_lsl_string(string):
    return '<<STRT>>' + string + '<<STOP>>'

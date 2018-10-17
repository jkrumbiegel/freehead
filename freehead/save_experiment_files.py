import pandas as pd
import numpy as np
from datetime import datetime
import os


def save_experiment_files(
        experiment_df: pd.DataFrame,
        trial_df: pd.DataFrame,
        rig_leds: np.ndarray,
        subject_prefix,
        recordings_folder='../recordings/'):

    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    folder_path = recordings_folder + timestamp + '/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    experiment_df.to_pickle(folder_path + subject_prefix + '_' + 'experiment_' + timestamp + '.pickle')
    trial_df.to_pickle(folder_path + subject_prefix + '_' + 'trials_' + timestamp + '.pickle')
    np.save(folder_path + 'rig_' + subject_prefix + '_' + timestamp + '.npy', rig_leds)

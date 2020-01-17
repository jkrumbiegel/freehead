import os
import re


def get_available_recordings(data_folder):

    subfolders = [f for f in os.listdir(data_folder) if re.fullmatch(r'\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d', f)]

    participants = {}

    for sf in subfolders:
        sf_path = os.path.join(data_folder, sf)
        files = [f for f in os.listdir(sf_path) if re.match(r'.*' + sf, f)]

        if len(files) != 3:
            raise Exception(f'Not all data files present in folder {sf}.')

        participant = files[0][0]
        session = files[0][1]

        rig = (f for f in files if re.match(r'.*rig', f)).__next__()
        experiment = (f for f in files if re.match(r'.*experiment', f)).__next__()
        trials = (f for f in files if re.match(r'.*trials', f)).__next__()

        if participant not in participants:
            participants[participant] = {}

        participants[participant][session] = dict(
            rig=os.path.join(sf_path, rig),
            experiment=os.path.join(sf_path, experiment),
            trials=os.path.join(sf_path, trials)
        )

    return participants

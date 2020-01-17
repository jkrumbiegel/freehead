from . import get_available_recordings, load_participant_df
import pandas as pd


def load_complete_dataframe(data_folder) -> pd.DataFrame:

    available_recordings = get_available_recordings(data_folder)

    dfs = [
        load_participant_df(participant, participant_dict)
        for participant, participant_dict in available_recordings.items()
    ]

    print('Concatenating all data frames...', end='')
    df = pd.concat(dfs, ignore_index=True)
    print(f' Done. Length: {len(df)} trials.')

    return df

import pandas as pd
import numpy as np


def load_participant_df(participant, participant_dict: dict):

    sessions = list(sorted(participant_dict.keys()))

    files = [
        [
            participant_dict[session]['experiment'],
            participant_dict[session]['trials'],
            participant_dict[session]['rig']]
        for session in sessions
    ]

    print(f'Loading files for participant {participant}...', end='')
    exp_dfs, trial_dfs, led_rigs = zip(
        *[(pd.read_pickle(e), pd.read_pickle(t), np.load(l)) for e, t, l in files])
    print(' Done.')

    for session, df, tdf, rig in (zip(sessions, exp_dfs, trial_dfs, led_rigs)):
        df['participant'] = participant
        df['session'] = session
        df['trial_in_session'] = df.index
        df['rig'] = [rig for _ in range(len(df))]
        tdf['session'] = session
        tdf['trial_number'] = tdf.index

    exp_df = pd.concat(exp_dfs, ignore_index=True)
    trial_df = pd.concat(trial_dfs, ignore_index=True)

    df = pd.merge(exp_df, trial_df, how='outer', on=['session', 'trial_number'])

    return df

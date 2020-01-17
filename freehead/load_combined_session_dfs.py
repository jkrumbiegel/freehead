import os
import re
import pandas as pd
import numpy as np


def load_combined_session_dfs(paths):

    all_files = [
        [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for path in paths]

    files = [
        [
            list(filter(re.compile(s).match, f))[0]
            for s in ['.*_experiment_.*', '.*_trials_.*', '.*_rig_.*']
        ]
        for f in all_files]

    exp_df_list, trial_df_list, led_rigs = zip(
        *[(pd.read_pickle(e), pd.read_pickle(t), np.load(l)) for e, t, l in files])

    for i, (df, tdf, rig) in enumerate(zip(exp_df_list, trial_df_list, led_rigs)):
        df['session'] = i + 1
        df['trial_in_session'] = df.index
        df['rig'] = [rig for _ in range(len(df))]
        tdf['session'] = i + 1
        tdf['trial_number'] = tdf.index

    exp_df = pd.concat(exp_df_list, ignore_index=True)
    trial_df = pd.concat(trial_df_list, ignore_index=True)

    df = pd.merge(exp_df, trial_df, how='outer', on=['session', 'trial_number'])

    return df

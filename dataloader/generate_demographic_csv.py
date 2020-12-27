import re
from glob import glob

import pandas as pd

for split in ['test', 'train']:
    dfs = []
    for filepath in glob(f'mimic3-benchmarks/data/root/{split}/**/episode*.csv'):
        match = re.match(r'.*/(\d+)/(episode\d+).csv', filepath)
        if match is None:
            continue
        patient, episode = match[1], match[2]
        df = pd.read_csv(filepath)
        df['filename'] = f'{patient}_{episode}_timeseries.csv'
        df = df[['filename'] + list(df.columns[:-1])]
        dfs.append(df)

    df = pd.concat(dfs, 0)

    df.to_csv(f'mimic3-benchmarks/data/multitask/{split}/demogfile.csv',
              index=False)

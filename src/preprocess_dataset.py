import re

import pandas as pd

from dataset_store import load_dataset

SOFT_VARIANTS = ['SUPERSOFT', 'ULTRASOFT', 'HYPERSOFT']
JUNK_COMPOUNDS = ['UNKNOWN', 'TEST_UNKNOWN', 'TEST', 'None', 'nan']

def is_classified(s):
    if pd.isna(s):
        return False
    if s == 'Finished' or s == 'Lapped':
        return True
    return bool(re.match(r'^\+\d+ Laps?$', s))

def load_clean_dataset():
    df = load_dataset(db_path='../data/f1_dataset.db')

    junk_cols = [c for c in df.columns if any(x in c for x in JUNK_COMPOUNDS)]
    df = df.drop(columns=junk_cols)

    for prefix in ['LongRunPace', 'DegSlope']:
        variant_cols = [f'{prefix}_{v}' for v in SOFT_VARIANTS if f'{prefix}_{v}' in df.columns]
        if not variant_cols:
            continue
        soft_col = f'{prefix}_SOFT'
        all_soft_cols = ([soft_col] if soft_col in df.columns else []) + variant_cols
        df[soft_col] = df[all_soft_cols].mean(axis=1)
        df = df.drop(columns=variant_cols)

    df = df.drop(columns=['GrandPrix'], errors='ignore')

    status = df[['Year', 'RoundNumber', 'Driver', 'Status']].copy()
    status['Classified'] = status['Status'].apply(is_classified)
    df = df.merge(status[['Year', 'RoundNumber', 'Driver', 'Classified']],
              on=['Year', 'RoundNumber', 'Driver'], how='left')

    df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
    df['_DNF'] = (~df['Classified']).astype(int)

    circuit_race_dnf = (
        df.groupby(['Circuit', 'Year', 'RoundNumber'])['_DNF']
        .mean()
        .reset_index()
        .sort_values(['Circuit', 'Year', 'RoundNumber'])
    )

    circuit_race_dnf['CircuitDNFRate'] = (
        circuit_race_dnf.groupby('Circuit', group_keys=False)['_DNF']
        .apply(lambda s: s.shift().expanding().mean())
    )
    df = df.merge(
        circuit_race_dnf[['Circuit', 'Year', 'RoundNumber', 'CircuitDNFRate']],
        on=['Circuit', 'Year', 'RoundNumber'], how='left'
    )

    team_race_dnf = (
        df.groupby(['TeamName', 'Year', 'RoundNumber'])['_DNF']
        .mean()
        .reset_index()
        .sort_values(['TeamName', 'Year', 'RoundNumber'])
    )
    team_race_dnf['TeamDNFRate'] = (
        team_race_dnf.groupby('TeamName', group_keys=False)['_DNF']
        .apply(lambda s: s.shift().rolling(10, min_periods=1).mean())
    )
    df = df.merge(
        team_race_dnf[['TeamName', 'Year', 'RoundNumber', 'TeamDNFRate']],
        on=['TeamName', 'Year', 'RoundNumber'], how='left'
    )

    overall_dnf_rate = df['_DNF'].mean()
    df['CircuitDNFRate'] = df['CircuitDNFRate'].fillna(overall_dnf_rate)
    df['TeamDNFRate'] = df['TeamDNFRate'].fillna(overall_dnf_rate)
 
    df = df.drop(columns=['_DNF'])

    
    df = df[df['Classified']].drop(columns=['Classified'])

    df = df.drop(columns=['Status'])

    new_order = ['Year', 'RoundNumber', 'Circuit', 'Driver', 'TeamName', 'QualiPosition',
       'GridPosition', 'GridPenalty', 'RaceFinishPosition', 'FPDeltaToFastest',
       'FP3Sector1', 'FP3Sector2', 'FP3Sector3',
       'LongRunPace_SOFT', 'LongRunPace_MEDIUM', 'LongRunPace_HARD', 'LongRunPace_INTERMEDIATE', 'LongRunPace_WET',
       'DegSlope_SOFT', 'DegSlope_MEDIUM', 'DegSlope_HARD', 'DegSlope_INTERMEDIATE', 'DegSlope_WET',
       'AvgQualiPos', 'RecentFormAvgFinish', 'DriverPointsBeforeRace',
       'ConstructorPointsBeforeRace', 'AvgCircuitQualiPos', 'TeamDNFRate', 'CircuitDNFRate', 'OvertakingDifficulty', 
       'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall',
    ]
    df = df[new_order]

    DROP_COLS = [
    'LongRunPace_WET',          # 96% missing
    'DegSlope_WET',             # 99% missing
    'LongRunPace_INTERMEDIATE', # 91% missing
    'DegSlope_INTERMEDIATE',    # 97% missing
    'LongRunPace_HARD',         # 95% missing
    'DegSlope_HARD',            # 82% missing
    'LongRunPace_MEDIUM',       # 76% missing
    'ConstructorPointsBeforeRace',  # name-mapping bug, 28% missing, biased
    ]
    df = df.drop(columns=DROP_COLS)

    df = df.dropna(subset=['RaceFinishPosition'])

    field_size = df.groupby(['Year', 'RoundNumber'])['GridPosition'].transform('max')
    pit_lane_start = df['GridPosition'] == 0
    has_grid = df['QualiPosition'].isna() & (df['GridPosition'] > 0)
    df.loc[has_grid, 'QualiPosition'] = df.loc[has_grid, 'GridPosition']
    no_grid = df['QualiPosition'].isna() & pit_lane_start
    df.loc[no_grid, 'QualiPosition'] = field_size[no_grid] + 1
    df.loc[no_grid, 'GridPosition'] = field_size[no_grid] + 1 
    df['GridPenalty'] = df['QualiPosition'] - df['GridPosition']

    group_cols = ['Year', 'RoundNumber']

    for col in ['LongRunPace_SOFT', 'DegSlope_SOFT', 'DegSlope_MEDIUM', 'FPDeltaToFastest']:
        df[f'{col}_missing'] = df[col].isna().astype(int)

    df['FP3Sectors_missing'] = df['FP3Sector1'].isna().astype(int)

    df['RookieDriver'] = df['AvgCircuitQualiPos'].isna().astype(int)

    for col in ['FPDeltaToFastest', 'FP3Sector1', 'FP3Sector2', 'FP3Sector3',
            'LongRunPace_SOFT', 'DegSlope_SOFT', 'DegSlope_MEDIUM']:
        df[col] = df[col].fillna(df.groupby(group_cols)[col].transform('median'))
        df[col] = df[col].fillna(df[col].median()) 


    for col in ['AvgQualiPos', 'RecentFormAvgFinish', 'AvgCircuitQualiPos']:
        df[col] = df[col].fillna(df.groupby(group_cols)[col].transform(lambda s: s.quantile(0.75)))
        df[col] = df[col].fillna(df[col].quantile(0.75))

    df['DriverPointsBeforeRace'] = df['DriverPointsBeforeRace'].fillna(
        df.groupby(group_cols)['DriverPointsBeforeRace'].transform(lambda s: s.quantile(0.25))
    )

    df['OvertakingDifficulty'] = df['OvertakingDifficulty'].fillna(
        df.groupby('Circuit')['OvertakingDifficulty'].transform('median')
    )

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

if __name__ == '__main__':
    df = load_clean_dataset()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing data (%):")
    print((df.isna().mean() * 100).sort_values(ascending=False).head(20).round(1))
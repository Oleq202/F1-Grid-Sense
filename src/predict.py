import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor


def make_model():
    return RandomForestRegressor(
        random_state=42, n_jobs=-1, max_depth=14, max_features=0.21844994003313262,
        min_samples_leaf=10, min_samples_split=17, n_estimators=289
    )


def get_selectable_races(df):
    races = (
        df[['Year', 'RoundNumber', 'Circuit']]
        .drop_duplicates()
        .sort_values(['Year', 'RoundNumber'])
        .reset_index(drop=True)
    )
    return races[races['Year'] >= 2019].reset_index(drop=True)


def predict_race(df, feature_cols, year, round_number, dnf_df=None):
    all_races = (
        df[['Year', 'RoundNumber']]
        .drop_duplicates()
        .sort_values(['Year', 'RoundNumber'])
        .reset_index(drop=True)
    )

    target = all_races[(all_races['Year'] == year) & (all_races['RoundNumber'] == round_number)]
    if target.empty:
        return None
    target_idx = target.index[0]

    if year < 2019:
        return None

    train_races = all_races.iloc[:target_idx]
    train_df = df.merge(train_races, on=['Year', 'RoundNumber'], how='inner')
    test_df = df[(df['Year'] == year) & (df['RoundNumber'] == round_number)].copy()

    model = make_model()
    model.fit(train_df[feature_cols], train_df['RaceFinishPosition'])
    preds = model.predict(test_df[feature_cols])

    test_df['PredictedScore'] = preds
    test_df['PredictedPosition'] = rankdata(preds, method='ordinal').astype(int)
    test_df['PositionDelta'] = test_df['RaceFinishPosition'] - test_df['PredictedPosition']

    result = test_df[['Driver', 'TeamName', 'GridPosition', 'RaceFinishPosition',
                       'PredictedPosition', 'PositionDelta']].sort_values('RaceFinishPosition')

    metrics = compute_race_metrics(test_df)

    records = result.to_dict(orient='records')

    if dnf_df is not None:
        race_dnfs = dnf_df[(dnf_df['Year'] == year) & (dnf_df['RoundNumber'] == round_number)]
        for _, row in race_dnfs.iterrows():
            grid = row['GridPosition']
            records.append({
                'Driver': row['Driver'],
                'TeamName': row['TeamName'],
                'GridPosition': grid if pd.notna(grid) else '-',
                'RaceFinishPosition': '-',
                'PredictedPosition': '-',
                'PositionDelta': '-',
            })

    return {
        'results': records,
        'metrics': metrics,
    }


def compute_race_metrics(test_df: pd.DataFrame) -> dict:
    if test_df['RaceFinishPosition'].nunique() < 2:
        spearman = float('nan')
    else:
        corr, _ = spearmanr(test_df['RaceFinishPosition'], test_df['PredictedPosition'])
        spearman = float(corr) if pd.notna(corr) else float('nan')

    mae = float(test_df['PositionDelta'].abs().mean())

    naive_mae = float((test_df['RaceFinishPosition'] - test_df['GridPosition']).abs().mean())
    if test_df['RaceFinishPosition'].nunique() < 2:
        naive_spearman = float('nan')
    else:
        naive_corr, _ = spearmanr(test_df['RaceFinishPosition'], test_df['GridPosition'])
        naive_spearman = float(naive_corr) if pd.notna(naive_corr) else float('nan')

    return {
        'spearman': round(spearman, 3) if spearman == spearman else None,
        'mae': round(mae, 3),
        'naive_grid_spearman': round(naive_spearman, 3) if naive_spearman == naive_spearman else None,
        'naive_grid_mae': round(naive_mae, 3),
    }
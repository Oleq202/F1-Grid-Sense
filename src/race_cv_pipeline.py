import numpy as np
import pandas as pd
from scipy.stats import randint, spearmanr, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def walk_forward_splits(df, min_train_size=20, test_size=10):
    races = df[['Year', 'RoundNumber']].drop_duplicates().sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

    n_races = len(races)

    splits = []
    start = min_train_size
    while start + test_size <= n_races:
        train_races = races.iloc[:start]
        test_races = races.iloc[start: start + test_size]
        splits.append((train_races, test_races))
        start += test_size

    return splits

def races_to_rows(df, race_ids):
    return df.merge(race_ids, on=['Year', 'RoundNumber'], how='inner')

def mean_race_spearman(df_test, y_pred, group_cols=('Year', 'RoundNumber')):
    tmp = df_test.copy()
    tmp['_pred'] = y_pred

    correlations = []
    for _, race in tmp.groupby(list(group_cols)):
        if (race['RaceFinishPosition']).nunique() < 2:
            continue
        corr, _ = spearmanr(race['RaceFinishPosition'], race['_pred'])
        if not np.isnan(corr):
            correlations.append(corr)

    return float(np.mean(correlations)) if correlations else float('nan')

def mean_absolute_error_pooled(df_test, y_pred):
    return float(np.mean(np.abs(df_test['RaceFinishPosition'].values - y_pred)))

def naive_grid_baseline(df_test):
    return df_test['GridPosition'].values

def run_rf_baseline(train_df, test_df, feature_cols, target_col='RaceFinishPosition', random_state=42, rf_params=None):
    default_params = {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5}
    if rf_params:
        default_params.update(rf_params)
    model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **default_params)
    model.fit(train_df[feature_cols], train_df[target_col])
    preds = model.predict(test_df[feature_cols])
    return preds, model

def build_walk_forward_cv_indices(df, min_train_races=20, test_size=10):
    splits = walk_forward_splits(df, min_train_races, test_size)
    df_with_pos = df.reset_index().rename(columns={'index': '_pos'})
    cv_indices = []
    for train_races, test_races in splits:
        train_idx = df_with_pos.merge(train_races, on=['Year', 'RoundNumber'], how='inner')['_pos'].to_numpy()
        test_idx = df_with_pos.merge(test_races, on=['Year', 'RoundNumber'], how='inner')['_pos'].to_numpy()
        cv_indices.append((train_idx, test_idx))
    return cv_indices


def make_race_spearman_scorer(df):
    def scorer(estimator, X, y):
        preds = estimator.predict(X)
        aux = df.loc[X.index, ['Year', 'RoundNumber', 'RaceFinishPosition']]
        return mean_race_spearman(aux, preds)
    return scorer


def tune_rf_hyperparams(df, feature_cols, target_col='RaceFinishPosition',
                         min_train_races=20, test_size=10,
                         n_iter=50, random_state=42, verbose=2):
    df = df.reset_index(drop=True)
    X = df[feature_cols]
    y = df[target_col]

    cv_indices = build_walk_forward_cv_indices(df, min_train_races, test_size)
    scorer = make_race_spearman_scorer(df)

    param_distributions = {
        'n_estimators': randint(100, 800),
        'max_depth': randint(3, 20),
        'min_samples_leaf': randint(1, 20),
        'min_samples_split': randint(2, 20),
        'max_features': uniform(0.1, 0.8),
    }

    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv_indices,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    search.fit(X, y)

    if verbose:
        print(f"\nBest mean race-Spearman across folds: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

    return search


def evaluate_walk_forward(df, feature_cols, min_train_races=20, test_size=10, verbose=True, rf_params=None):
    splits = walk_forward_splits(df, min_train_races, test_size)
    results = []
    for i, (train_races, test_races) in enumerate(splits):
        train_df = races_to_rows(df, train_races)
        test_df = races_to_rows(df, test_races)

        naive_preds = naive_grid_baseline(test_df)
        naive_spearman = mean_race_spearman(test_df, naive_preds)
        naive_mae = mean_absolute_error_pooled(test_df, naive_preds)

        rf_preds, rf_model = run_rf_baseline(train_df, test_df, feature_cols, rf_params=rf_params)
        rf_spearman = mean_race_spearman(test_df, rf_preds)
        rf_mae = mean_absolute_error_pooled(test_df, rf_preds)

        fold_result = {
            'fold': i,
            'train_races': len(train_races),
            'test_races': len(test_races),
            'naive_spearman': naive_spearman,
            'naive_mae': naive_mae,
            'rf_spearman': rf_spearman,
            'rf_mae': rf_mae,
        }

        results.append(fold_result)

        if verbose:
            print(f"Fold {i:>2} | train={len(train_races):>3} races, test={len(test_races):>3} races | "
                  f"naive: spearman={naive_spearman:.3f} mae={naive_mae:.2f} | "
                  f"RF: spearman={rf_spearman:.3f} mae={rf_mae:.2f}")
    results_df = pd.DataFrame(results)

    if verbose:
        print("\n--- Averages across all folds ---")
        print(results_df[['naive_spearman', 'naive_mae', 'rf_spearman', 'rf_mae']].mean().round(3))
 
    return results_df, rf_model

if __name__ == '__main__':
    from preprocess_dataset import load_clean_dataset
 
    df = load_clean_dataset()
 
    non_feature_cols = ['Year', 'RoundNumber', 'Circuit', 'Driver', 'TeamName',
                         'RaceFinishPosition']
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
 
    print(f"Using {len(feature_cols)} features: {feature_cols}\n")

    print("=== Baseline RF ===")
    baseline_results, baseline_rf_model = evaluate_walk_forward(df, feature_cols, min_train_races=20, test_size=10, rf_params={'max_depth': 14, 'max_features': np.float64(0.21844994003313262), 'min_samples_leaf': 10, 'min_samples_split': 17, 'n_estimators': 289})

    print("\n=== Hyperparameter search ===")
    search = tune_rf_hyperparams(df, feature_cols, min_train_races=20, test_size=10, n_iter=30)

    print("\n=== Tuned RF ===")
    tuned_results, tuned_rf_model = evaluate_walk_forward(df, feature_cols, min_train_races=20, test_size=10,
                                           rf_params=search.best_params_)

    importances = pd.Series(tuned_rf_model.feature_importances_, index=feature_cols)
    print("\n=== Feature importances (last fold's model) ===")
    print(importances.sort_values(ascending=False).round(4))

    importance_threshold = 0.01
    useless_features = importances[importances < importance_threshold].index.tolist()
    reduced_feature_cols = [c for c in feature_cols if c not in useless_features]

    print(f"\nDropping {len(useless_features)} features with importance < {importance_threshold}: {useless_features}")
    print(f"Remaining {len(reduced_feature_cols)} features: {reduced_feature_cols}\n")

    print("=== Tuned RF (reduced feature set) ===")
    reduced_results, reduced_rf_model = evaluate_walk_forward(
        df, reduced_feature_cols, min_train_races=20, test_size=10,
        rf_params={'max_depth': 14, 'max_features': np.float64(0.21844994003313262), 'min_samples_leaf': 10, 'min_samples_split': 17, 'n_estimators': 289})

    print("\n=== Full vs reduced feature set (mean across folds) ===")
    comparison = pd.DataFrame({
        'full': tuned_results[['rf_spearman', 'rf_mae']].mean(),
        'reduced': reduced_results[['rf_spearman', 'rf_mae']].mean(),
    })
    comparison['diff'] = comparison['reduced'] - comparison['full']
    print(comparison.round(4))
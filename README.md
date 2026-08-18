# F1 Race Predictor

[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/Model-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

Predicts Formula 1 race finishing order from pre-race data (grid position, driver/team season form, DNF rates, etc.), then compares the prediction against what actually happened. Built on ~145 non-sprint races (2018–present) pulled via [FastF1](https://github.com/theOehrly/Fast-F1) into a local SQLite dataset.

Pick a season and race, and the app trains a model using only races that happened _before_ it, predicts that race's finishing order, and shows how close it got.

## How it works

- **Data**: race-level features (grid position, rolling driver/team form, DNF rates, etc.) built from FastF1 telemetry/results and stored in `data/f1_dataset.db`.
- **Model**: a tuned `RandomForestRegressor`, validated via walk-forward cross-validation (train on past races, test on the next block) rather than random splits — a random split would leak future form into training and make results look better than they'd actually be.
- **Per-race prediction**: for any race the user selects (2019 onward), the model is retrained from scratch using only races strictly before it in the calendar, then predicts that race. This mirrors the walk-forward validation setup and avoids the model ever having "seen" the race it's predicting.
- **Evaluation metric**: mean race-level Spearman rank correlation (predicted order vs actual order) and mean absolute position error, each compared against a naive "everyone finishes where they qualified" baseline.

2018 is used purely as training history and isn't selectable — there's not enough prior data to fairly evaluate a prediction that early.

## Tech stack

- **Data pipeline**: Python, pandas, FastF1, SQLite
- **Model**: scikit-learn (RandomForestRegressor)
- **Backend**: FastAPI
- **Frontend**: plain HTML/CSS/JS (no framework), served directly by FastAPI

## Project structure

```
F1-Race-Predictor/
├── data/
│   └── f1_dataset.db          # SQLite dataset of race results + features
├── frontend/
│   └── index.html             # UI: race picker, results table, accuracy panel
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app: /races, /predict, serves frontend
│   ├── predict.py              # per-race walk-forward prediction + metrics
│   ├── preprocess_dataset.py   # loads and cleans the dataset for modeling
│   ├── dataset_store.py        # SQLite read/write layer
│   ├── build_dataset.py        # dataset assembly
│   ├── build_gp_dataframe.py   # per-Grand-Prix dataframe construction
│   ├── fastf1_data_fetcher.py  # FastF1 API wrapper with retry/timeout handling
│   ├── race_cv_pipeline.py     # walk-forward CV, model comparison, tuning
│   └── refetch_missing.py      # re-pulls rounds with missing/bad data
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
```

Then open `http://127.0.0.1:8000/` in a browser.

## API

| Endpoint                                | Description                                                              |
| --------------------------------------- | ------------------------------------------------------------------------ |
| `GET /races`                            | List of selectable races (2019+) with year, round, and circuit           |
| `GET /predict?year=YYYY&round_number=N` | Predicted vs actual finishing order for that race, plus accuracy metrics |

## Notes / limitations

- Per-race Spearman correlation is computed over ~13–20 rows (one race), so it's noisy by nature — a single chaotic wet race can swing it dramatically, including into negative territory. The walk-forward mean across all races (computed during model development, see `race_cv_pipeline.py`) is the more reliable measure of overall model quality; the per-race number in the UI is meant for exploring individual races, not judging the model in isolation.
- Predictions retrain the model per request (cached after the first call for a given race), which is fine given the small dataset size but means the first request for an uncached race takes a moment.

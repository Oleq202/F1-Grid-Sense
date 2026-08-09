from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .predict import get_selectable_races, predict_race
from .preprocess_dataset import load_clean_dataset, load_full_dataset

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

df = load_clean_dataset()
full_df = load_full_dataset()
non_feature_cols = ['Year', 'RoundNumber', 'Circuit', 'Driver', 'TeamName', 'RaceFinishPosition']
feature_cols = [c for c in df.columns if c not in non_feature_cols]

_prediction_cache = {}

@app.get("/races")
def list_races():
    return get_selectable_races(df).to_dict(orient='records')

@app.get("/predict")
def predict(year: int, round_number: int):
    if year < 2019:
        raise HTTPException(status_code=400, detail="2018 races are training-only, not selectable")
    key = (year, round_number)
    if key in _prediction_cache:
        return _prediction_cache[key]
    payload = predict_race(df, feature_cols, year, round_number, full_df=full_df)
    if payload is None:
        raise HTTPException(status_code=404, detail="Race not found")
    _prediction_cache[key] = payload
    return payload

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
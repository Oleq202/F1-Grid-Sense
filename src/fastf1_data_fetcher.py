import time

import fastf1
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast
from fastf1.exceptions import RateLimitExceededError

fastf1.Cache.enable_cache('data/f1_cache')

def fetch_session_data(year, grand_prix, session, laps=True, telemetry=False, weather=True, messages=False):
    
    session_obj = fastf1.get_session(year, grand_prix, session)
    wait = 30
    while True:
        try:
            session_obj.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)
            return session_obj
        except RateLimitExceededError:
            print(f"  Rate limited - waiting {wait}s before retrying...")
            time.sleep(wait)
            wait = min(wait * 2, 300)


def fetch_event_schedule(year):
    return fastf1.get_event_schedule(year)

def is_sprint_weekend(year, round_number):
    schedule = fetch_event_schedule(year)
    event_rows = schedule[schedule['RoundNumber'] == round_number]
    if event_rows.empty:
        return False
    event_format = event_rows.iloc[0].get('EventFormat', '')
    return 'sprint' in str(event_format).lower()

def fetch_weather_data(session):
    weather = session.weather_data
    weather_info = pd.DataFrame([{
        'AirTemp': round(weather['AirTemp'].mean(), 1),
        'TrackTemp': round(weather['TrackTemp'].mean(), 1),
        'Humidity': round(weather['Humidity'].mean(), 1),
        'Pressure': round(weather['Pressure'].mean(), 1),
        'WindSpeed': round(weather['WindSpeed'].mean(), 1),
        'Rainfall': weather['Rainfall'].any(),
    }])
    return weather_info


def get_session_info(session, weather_info):
    info = session.session_info

    session_data = {
        'SessionKey': info.get('Key'),
        'SessionType': info.get('Type'),
        'Circuit': info.get('Meeting', {}).get('Circuit', {}).get('ShortName'),
    }
    session_data.update(weather_info.iloc[0].to_dict())

    return pd.DataFrame([session_data])



def compute_fp_delta(fp_laps_by_session):
    all_best = []
    for laps in fp_laps_by_session.values():
        quick = laps.pick_quicklaps()
        if quick.empty:
            continue
        all_best.append(quick.groupby('Driver')['LapTime'].min().reset_index())

    if not all_best:
        return pd.DataFrame(columns=['Driver', 'DeltaToFastest'])

    combined = pd.concat(all_best)
    best_overall = combined.groupby('Driver')['LapTime'].min().reset_index()
    fastest = best_overall['LapTime'].min()
    best_overall['DeltaToFastest'] = (best_overall['LapTime'] - fastest).dt.total_seconds()
    return best_overall[['Driver', 'DeltaToFastest']]


def compute_sector_pace(laps):
    quick = laps.pick_quicklaps()
    if quick.empty:
        return pd.DataFrame(columns=['Driver', 'Sector1Time', 'Sector2Time', 'Sector3Time'])
    return quick.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time']].min().reset_index()


def compute_long_run_pace(laps):
    accurate = laps.pick_accurate()
    if accurate.empty:
        return pd.DataFrame(columns=['Driver', 'Compound', 'LapTime'])
    return accurate.groupby(['Driver', 'Compound'])['LapTime'].mean().reset_index()


def compute_tyre_degradation(laps):
    accurate = laps.pick_accurate()
    degradation = []
    for (driver, compound), group in accurate.groupby(['Driver', 'Compound']):
        if len(group) < 2:
            continue
        group = group.sort_values('TyreLife')
        slope = np.polyfit(group['TyreLife'], group['LapTime'].dt.total_seconds(), 1)[0]
        degradation.append({'Driver': driver, 'Compound': compound, 'DegSlope': slope})
    return pd.DataFrame(degradation, columns=['Driver', 'Compound', 'DegSlope'])



def get_fp_delta(year, grand_prix):
    fp_laps = {
        fp: fetch_session_data(year, grand_prix, fp, laps=True, telemetry=False).laps
        for fp in ['FP1', 'FP2', 'FP3']
    }
    return compute_fp_delta(fp_laps)


def get_fp3_sector_pace(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'FP3', laps=True, telemetry=False)
    return compute_sector_pace(session.laps)


def get_long_run_pace(year, grand_prix, session_name='FP3'):
    session = fetch_session_data(year, grand_prix, session_name, laps=True, telemetry=False)
    return compute_long_run_pace(session.laps)


def get_tyre_degredation(year, grand_prix, session_name='FP2'):
    session = fetch_session_data(year, grand_prix, session_name, laps=True, telemetry=False)
    return compute_tyre_degradation(session.laps)


def get_grid_penalties(year, grand_prix):
    quali = fetch_session_data(year, grand_prix, 'Q', laps=False, telemetry=False).results[['Abbreviation', 'Position']]
    quali = quali.rename(columns={'Abbreviation': 'Driver'})
    race = fetch_session_data(year, grand_prix, 'R', laps=False, telemetry=False).results[['Abbreviation', 'GridPosition']]
    race = race.rename(columns={'Abbreviation': 'Driver'})
    merged = quali.merge(race, on='Driver')
    merged['GridPenalty'] = merged['Position'] - merged['GridPosition']
    return merged


def get_qualifying_positions(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'Q', laps=False, telemetry=False)
    grid_positions = session.results[['Abbreviation', 'Position']].copy()
    return grid_positions.rename(columns={'Abbreviation': 'Driver'})


def get_grid_positions(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'R', laps=False, telemetry=False)
    grid_positions = session.results[['Abbreviation', 'GridPosition']].copy()
    return grid_positions.rename(columns={'Abbreviation': 'Driver'})


def get_constructor_qualifying_form(year, upcoming_round, n_races=5):
    schedule = fetch_event_schedule(year)
    past_events = schedule[schedule['RoundNumber'] < upcoming_round].tail(n_races)

    results = []
    for _, event in past_events.iterrows():
        session = fetch_session_data(year, event['EventName'], 'Q', laps=False, telemetry=False)
        quali_results = session.results[['TeamName', 'Position']].copy()
        quali_results['Round'] = event['RoundNumber']
        results.append(quali_results)

    combined = pd.concat(results)
    return combined.groupby('TeamName')['Position'].mean().reset_index(name='AvgQualiPos')


def get_recent_driver_form(year, upcoming_round, n_races=5):
    events = []
    yr = year
    round_cutoff = upcoming_round
    seasons_back = 0
    while len(events) < n_races and seasons_back <= 3:
        schedule = fetch_event_schedule(yr)
        past_events = schedule[(schedule['RoundNumber'] > 0) & (schedule['RoundNumber'] < round_cutoff)]
        past_events = past_events.sort_values('RoundNumber', ascending=False)
        for _, event in past_events.iterrows():
            events.append((yr, int(event['RoundNumber'])))
            if len(events) >= n_races:
                break
        yr -= 1
        round_cutoff = 10_000
        seasons_back += 1

    results = []
    for yr_e, rnd_e in events:
        try:
            race = fetch_session_data(yr_e, rnd_e, 'R', laps=False, telemetry=False)
        except Exception as e:
            print(f"Skipping {yr_e} R{rnd_e} for recent form: {e}")
            continue
        race_results = race.results[['Abbreviation', 'Position']].dropna(subset=['Position'])
        results.append(race_results)

    if not results:
        return pd.DataFrame(columns=['Driver', 'RecentFormAvgFinish'])

    combined = pd.concat(results).rename(columns={'Abbreviation': 'Driver'})
    return combined.groupby('Driver')['Position'].mean().reset_index(name='RecentFormAvgFinish')


def get_championship_standings_before_race(year, round_number):
    
    ergast = Ergast()
    empty = (
        pd.DataFrame(columns=['Driver', 'DriverPointsBeforeRace']),
        pd.DataFrame(columns=['TeamName', 'ConstructorPointsBeforeRace']),
    )

    try:
        if round_number <= 1:
            prior_year = year - 1
            driver_resp = ergast.get_driver_standings(season=prior_year)
            constructor_resp = ergast.get_constructor_standings(season=prior_year)
        else:
            driver_resp = ergast.get_driver_standings(season=year, round=round_number - 1)
            constructor_resp = ergast.get_constructor_standings(season=year, round=round_number - 1)
    except Exception as e:
        print(f"No standings available before {year} R{round_number}: {e}")
        return empty

    if not driver_resp.content or not constructor_resp.content:
        return empty

    driver_standings = driver_resp.content[0]
    constructor_standings = constructor_resp.content[0]

    driver_df = driver_standings[['driverCode', 'points']].rename(
        columns={'driverCode': 'Driver', 'points': 'DriverPointsBeforeRace'})
    constructor_df = constructor_standings[['constructorName', 'points']].rename(
        columns={'constructorName': 'TeamName', 'points': 'ConstructorPointsBeforeRace'})
    return driver_df, constructor_df


def get_driver_circuit_qualifying_history(driver, grand_prix, year, years_back=5):
    positions = []

    for y in range(year - 1, year - years_back - 1, -1):
        try:
            session = fetch_session_data(y, grand_prix, 'Q', laps=False, telemetry=False)
            row = session.results[session.results['Abbreviation'] == driver]

            if row.empty:
                continue

            pos = row.iloc[0]['Position']
            if pd.isna(pos):
                continue

            positions.append(pos)

        except Exception as e:
            print(f"Skipping {y} {grand_prix}: {e}")
            continue

    return sum(positions) / len(positions) if positions else None


def get_estimate_difficulty_of_overtaking(year, grand_prix, years_back=10):
    deltas = []
    for yr in range(year - 1, year - years_back - 1, -1):
        try:
            race = fetch_session_data(yr, grand_prix, 'R', laps=False, telemetry=False)
        except Exception as e:
            print(f"Skipping {yr} {grand_prix}: {e}")
            continue
        df = race.results[['GridPosition', 'Position']].dropna()
        deltas.extend((df['Position'] - df['GridPosition']).abs().tolist())
    return sum(deltas) / len(deltas) if deltas else None
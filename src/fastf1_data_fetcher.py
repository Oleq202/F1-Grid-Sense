import fastf1
import numpy as np
import pandas as pd

fastf1.cache.enable_cache('data/f1_cache')

def fetch_session_data(year, grand_prix, session):
    session = fastf1.get_session(year, grand_prix, session)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session

def fetch_event_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule

def fetch_weather_data(session):
    weather = session.weather_data
    weather_info = pd.DataFrame([{
        'AirTemp': round(weather['AirTemp'].mean(), 1),
        'TrackTemp': round(weather['TrackTemp'].mean(), 1),
        'Humidity': round(weather['Humidity'].mean(), 1),
        'Pressure': round(weather['Pressure'].mean(), 1),
        'WindSpeed': round(weather['WindSpeed'].mean(), 1),
        'Rainfall': weather['Rainfall'].any()
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

def get_fp_delta(year, grand_prix):
    all_best_laps = []

    for fp in ['FP1', 'FP2', 'FP3']:
        session = fetch_session_data(year, grand_prix, fp)
        laps = session.laps.pick_quicklaps()
        best_per_driver = laps.groupby('Driver')['LapTime'].min().reset_index()
        best_per_driver['Session'] = fp
        all_best_laps.append(best_per_driver)

    combined = pd.concat(all_best_laps)
    best_overall = combined.groupby('Driver')['LapTime'].min().reset_index()
    fastest_time = best_overall['LapTime'].min()

    best_overall['DeltaToFastest'] = (best_overall['LapTime'] - fastest_time).dt.total_seconds()
    return best_overall

def get_fp3_sector_pace(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'FP3')
    laps = session.laps.pick_quicklaps()
    sector_pace = laps.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time']].min().reset_index()
    return sector_pace

def get_constructor_qualifying_form(year, upcoming_round, n_races=5):
    schedule = fetch_event_schedule(year)
    past_events = schedule[schedule['RoundNumber'] < upcoming_round].tail(n_races)

    results = []
    for _, event in past_events.iterrows():
        session = fetch_session_data(year, event['EventName'], 'Q')
        quali_results = session.results[['TeamName', 'Position']].copy()
        quali_results['Round'] = event['RoundNumber']
        results.append(quali_results)

    combined = pd.concat(results)
    return combined.groupby('TeamName')['Position'].mean().reset_index(name='AvgQualiPos')

def get_driver_circuit_qualifying_history(driver, grand_prix, year, years_back = 5):
    positions = []

    for y in range(year - 1, year - years_back - 1, -1):
        try:
            session = fetch_session_data(y, grand_prix, 'Q')
            row = session.results[session.results['Driver'] == driver]

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

def get_qualifying_positions(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'Q')
    grid_positions = session.results[['Driver', 'Position']].copy()
    return grid_positions

def get_grid_positions(year, grand_prix):
    session = fetch_session_data(year, grand_prix, 'R')
    grid_positions = session.results[['Driver', 'GridPosition']].copy()
    return grid_positions

def get_long_run_pace(year, grand_prix, session_name = 'FP3'):
    session = fetch_session_data(year, grand_prix, session_name)
    laps = session.laps.pick_accurate()
    long_run = laps.groupby(['Driver', 'Compound'])['LapTime'].mean().reset_index()
    return long_run

def get_tyre_degredation(year, grand_prix, session_name = 'FP2'):
    session = fetch_session_data(year, grand_prix, session_name)
    laps = session.laps.pick_accurate()

    degradation = []
    for (driver, compound), group in laps.groupby(['Driver', 'Compound']):
        if len(group) < 2:
            continue
        group = group.sort_values('TyreLife')
        slope = np.polyfit(group['TyreLife'], group['LapTime'].dt.total_seconds(), 1)[0]
        degradation.append({'Driver': driver, 'Compound': compound, 'DegSlope': slope})

    return pd.DataFrame(degradation)

def get_grid_penalties(year, grand_prix):
    quali = fetch_session_data(year, grand_prix, 'Q').results[['Driver', 'Position']]
    race = fetch_session_data(year, grand_prix, 'R').results[['Driver', 'GridPosition']]
    merged = quali.merge(race, on='Driver')
    merged['GridPenalty'] = merged['Position'] - merged['GridPosition']
    return merged

def get_estimate_difficulty_of_overtaking(year, grand_prix, years_back = 20):
    deltas = []
    for yr in range(year - 1, year - years_back - 1, -1):
        race = fetch_session_data(yr, grand_prix, 'R')
        df = race.results[['GridPosition', 'Position']].dropna()
        deltas.extend((df['Position'] - df['GridPosition']).abs().tolist())
    return sum(deltas) / len(deltas) if deltas else None
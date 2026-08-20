import fastf1
import pandas as pd

try:
    from .fastf1_data_fetcher import (
        compute_fp_delta,
        compute_long_run_pace,
        compute_sector_pace,
        compute_tyre_degradation,
        fetch_session_data,
        fetch_weather_data,
        find_round_for_location,
        get_championship_standings_before_race,
        get_circuit_location,
        get_constructor_qualifying_form,
        get_estimate_difficulty_of_overtaking,
        get_recent_driver_form,
        get_session_info,
        normalize_team_name,
    )
except (ImportError, ValueError):
    from fastf1_data_fetcher import (
        compute_fp_delta,
        compute_long_run_pace,
        compute_sector_pace,
        compute_tyre_degradation,
        fetch_session_data,
        fetch_weather_data,
        find_round_for_location,
        get_championship_standings_before_race,
        get_circuit_location,
        get_constructor_qualifying_form,
        get_estimate_difficulty_of_overtaking,
        get_recent_driver_form,
        get_session_info,
        normalize_team_name,
    )


def _resolve_round_number(year, grand_prix):
    if isinstance(grand_prix, int):
        return grand_prix
    event = fastf1.get_event(year, grand_prix)
    return int(event['RoundNumber'])


def _pivot_by_compound(df, value_col, prefix):
    if df is None or df.empty:
        return pd.DataFrame(columns=['Driver'])
    pivoted = df.pivot_table(index='Driver', columns='Compound', values=value_col, aggfunc='first')
    pivoted.columns = [f'{prefix}_{c}' for c in pivoted.columns]
    return pivoted.reset_index()


def build_gp_dataframe(year, grand_prix, history_years_back=5, overtaking_years_back=4, constructor_n_races=5):

    round_number = _resolve_round_number(year, grand_prix)

    q_session = fetch_session_data(year, grand_prix, 'Q', laps=False, telemetry=False)
    r_session = fetch_session_data(year, grand_prix, 'R', laps=False, telemetry=False)
    fp_sessions = {
        fp: fetch_session_data(year, grand_prix, fp, laps=True, telemetry=False)
        for fp in ['FP1', 'FP2', 'FP3']
    }

    weather_info = fetch_weather_data(r_session)
    session_info = get_session_info(q_session, weather_info)

    # --- Base: qualifying results give us Driver / Team / Position ---
    df = q_session.results[['Abbreviation', 'TeamName', 'Position']].copy()
    df = df.rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'})

    # --- Grid position, grid penalty, race finish position (target), and status ---
    race_results = r_session.results[['Abbreviation', 'GridPosition', 'Position', 'Status']].copy()
    race_results = race_results.rename(columns={
        'Abbreviation': 'Driver',
        'Position': 'RaceFinishPosition',
    })
    df = df.merge(race_results, on='Driver', how='left')
    df['GridPenalty'] = df['QualiPosition'] - df['GridPosition']

    # --- Free practice pace: best-lap delta across FP1-3 combined ---
    fp_delta = compute_fp_delta({fp: sess.laps for fp, sess in fp_sessions.items()})
    fp_delta = fp_delta.rename(columns={'DeltaToFastest': 'FPDeltaToFastest'})
    df = df.merge(fp_delta, on='Driver', how='left')

    # --- FP3 sector pace ---
    sector_pace = compute_sector_pace(fp_sessions['FP3'].laps)
    if not sector_pace.empty:
        sector_pace = sector_pace.rename(columns={
            'Sector1Time': 'FP3Sector1',
            'Sector2Time': 'FP3Sector2',
            'Sector3Time': 'FP3Sector3',
        })
        df = df.merge(sector_pace, on='Driver', how='left')

    # --- Long-run pace by compound (from FP3, accurate laps) ---
    long_run = compute_long_run_pace(fp_sessions['FP3'].laps)
    df = df.merge(_pivot_by_compound(long_run, 'LapTime', 'LongRunPace'), on='Driver', how='left')

    # --- Tyre degradation slope by compound (from FP2, accurate laps) ---
    degradation_df = compute_tyre_degradation(fp_sessions['FP2'].laps)
    df = df.merge(_pivot_by_compound(degradation_df, 'DegSlope', 'DegSlope'), on='Driver', how='left')

    # --- Constructor qualifying form over recent rounds ---
    df['TeamNameCanonical'] = df['TeamName'].map(normalize_team_name)
    constructor_form = get_constructor_qualifying_form(year, round_number, n_races=constructor_n_races)
    df = df.merge(constructor_form, on='TeamNameCanonical', how='left')
    df = df.drop(columns=['TeamNameCanonical'])

    # --- Driver's recent race form (last 5 starts, any circuit) ---
    recent_form = get_recent_driver_form(year, round_number, n_races=5)
    df = df.merge(recent_form, on='Driver', how='left')

    # --- Championship points entering this race ---
    driver_points, constructor_points = get_championship_standings_before_race(year, round_number)
    df = df.merge(driver_points, on='Driver', how='left')
    df = df.merge(constructor_points, on='TeamName', how='left')

    # --- Driver's historical qualifying pace at this circuit ---
    circuit_location = get_circuit_location(year, grand_prix)
    history_by_driver = {drv: [] for drv in df['Driver']}
    for yr in range(year - 1, year - history_years_back - 1, -1):
        past_round = find_round_for_location(yr, circuit_location)
        if past_round is None:
            continue
        try:
            past_session = fetch_session_data(yr, past_round, 'Q', laps=False, telemetry=False)
        except Exception as e:
            print(f"Skipping {yr} R{past_round} ({circuit_location}): {e}")
            continue
        past_results = past_session.results[['Abbreviation', 'Position']].dropna(subset=['Position'])
        pos_by_driver = dict(zip(past_results['Abbreviation'], past_results['Position']))
        for drv in history_by_driver:
            if drv in pos_by_driver:
                history_by_driver[drv].append(pos_by_driver[drv])

    df['AvgCircuitQualiPos'] = df['Driver'].map(
        lambda drv: (sum(history_by_driver[drv]) / len(history_by_driver[drv]))
        if history_by_driver[drv] else None
    )

    # --- Circuit-level overtaking difficulty (scalar, broadcast) ---
    df['OvertakingDifficulty'] = get_estimate_difficulty_of_overtaking(
        year, grand_prix, years_back=overtaking_years_back
    )

    # --- Session / weather info (scalar, broadcast) ---
    info = session_info.iloc[0]
    for col in ['Circuit', 'AirTemp', 'TrackTemp', 'Humidity', 'Pressure', 'WindSpeed', 'Rainfall']:
        df[col] = info[col]

    df['Year'] = year
    df['RoundNumber'] = round_number
    df['GrandPrix'] = grand_prix

    return df


if __name__ == '__main__':
    everything = build_gp_dataframe(2024, 'Monza')
    print(everything)
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).parent.parent / 'data' / 'f1_dataset.db'
TABLE_NAME = 'gp_driver_rows'


def init_db(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            Year INTEGER,
            RoundNumber INTEGER,
            Driver TEXT,
            PRIMARY KEY (Year, RoundNumber, Driver)
        )
    """)
    conn.commit()
    return conn


def _ensure_columns(conn, columns):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")}
    for col in columns:
        if col not in existing:
            conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col}"')


def save_gp_dataframe(df, db_path=DEFAULT_DB_PATH):
    if df.empty:
        return

    conn = init_db(db_path)
    _ensure_columns(conn, df.columns)

    df_to_store = df.copy()
    for col in df_to_store.columns:
        if pd.api.types.is_timedelta64_dtype(df_to_store[col]):
            df_to_store[col] = df_to_store[col].dt.total_seconds()

    df_to_store.to_sql('_tmp_upsert', conn, if_exists='replace', index=False)
    cols = ", ".join(f'"{c}"' for c in df_to_store.columns)
    conn.execute(f'INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) SELECT {cols} FROM _tmp_upsert')
    conn.execute('DROP TABLE _tmp_upsert')
    conn.commit()
    conn.close()


def load_dataset(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM {TABLE_NAME}', conn)
    conn.close()
    return df


def get_stored_keys(db_path=DEFAULT_DB_PATH):
    conn = init_db(db_path)
    rows = conn.execute(f'SELECT DISTINCT Year, RoundNumber FROM {TABLE_NAME}').fetchall()
    conn.close()
    return set(rows)


def row_count(db_path=DEFAULT_DB_PATH):
    conn = init_db(db_path)
    count = conn.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}').fetchone()[0]
    conn.close()
    return count


def save_pre_race_dataframe(df, db_path=DEFAULT_DB_PATH):
    if df.empty:
        return
    
    conn = init_db(db_path)
    _ensure_columns(conn, df.columns)

    df_to_store = df.copy()
    for col in df_to_store.columns:
        if pd.api.types.is_timedelta64_dtype(df_to_store[col]):
            df_to_store[col] = df_to_store[col].dt.total_seconds()

    df_to_store.to_sql('_tmp_upsert', conn, if_exists='replace', index=False)
    cols = ", ".join(f'"{c}"' for c in df_to_store.columns)
    conn.execute(f'INSERT OR REPLACE INTO {TABLE_NAME} ({cols}) SELECT {cols} FROM _tmp_upsert')
    conn.execute('DROP TABLE _tmp_upsert')
    conn.commit()
    conn.close()


def update_race_results(year, round_number, db_path=DEFAULT_DB_PATH):
    try:
        from src.fastf1_data_fetcher import fetch_session_data
    except (ImportError, ValueError):
        from fastf1_data_fetcher import fetch_session_data
    
    conn = init_db(db_path)
    
    try:
        r_session = fetch_session_data(year, round_number, 'R', laps=False, telemetry=False)
        race_results = r_session.results[['Abbreviation', 'GridPosition', 'Position', 'Status']].copy()
        race_results = race_results.rename(columns={
            'Abbreviation': 'Driver',
            'Position': 'RaceFinishPosition',
        })
        
        quali_results = conn.execute(
            f'SELECT Driver, QualiPosition FROM {TABLE_NAME} WHERE Year = ? AND RoundNumber = ?',
            (year, round_number)
        ).fetchall()
        
        quali_dict = {row[0]: row[1] for row in quali_results}
        race_results['QualiPosition'] = race_results['Driver'].map(quali_dict)
        race_results['GridPenalty'] = race_results['QualiPosition'] - race_results['GridPosition']
        
        for _, row in race_results.iterrows():
            driver = row['Driver']
            grid_penalty = row['GridPenalty'] if pd.notna(row['GridPenalty']) else 0
            finish_pos = row['RaceFinishPosition']
            status = row['Status']
            
            conn.execute(f'''
                UPDATE {TABLE_NAME} 
                SET GridPosition = ?, GridPenalty = ?, RaceFinishPosition = ?, Status = ?, IsComplete = 1
                WHERE Year = ? AND RoundNumber = ? AND Driver = ?
            ''', (row['GridPosition'], grid_penalty, finish_pos, status, year, round_number, driver))
        
        # Ensure any remaining rows for this race (e.g. DNS / withdrawn drivers) are also marked complete
        conn.execute(f'''
            UPDATE {TABLE_NAME}
            SET IsComplete = 1
            WHERE Year = ? AND RoundNumber = ? AND IsComplete = 0
        ''', (year, round_number))

        conn.commit()
        print(f"Updated race results for {year} R{round_number}")
        return True
        
    except Exception as e:
        print(f"Failed to update race results for {year} R{round_number}: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_incomplete_races(db_path=DEFAULT_DB_PATH):
    """Get races that have pre-race data but missing race results."""
    conn = init_db(db_path)
    
    # Check if IsComplete column exists
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")}
    
    if 'IsComplete' not in columns:
        # Add the column and set all existing records to complete
        conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN IsComplete INTEGER DEFAULT 1')
        conn.commit()
        conn.close()
        return set()
    
    rows = conn.execute(
        f'SELECT DISTINCT Year, RoundNumber FROM {TABLE_NAME} WHERE IsComplete = 0'
    ).fetchall()
    conn.close()
    return set(rows)


if __name__ == '__main__':
    print(f"Rows stored: {row_count()}")
    print(f"GPs stored: {len(get_stored_keys())}")
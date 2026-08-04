import sqlite3

import pandas as pd

DEFAULT_DB_PATH = 'data/f1_dataset.db'
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


if __name__ == '__main__':
    print(f"Rows stored: {row_count()}")
    print(f"GPs stored: {len(get_stored_keys())}")
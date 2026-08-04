import sys
from datetime import datetime

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from build_gp_dataframe import build_gp_dataframe
from dataset_store import get_stored_keys, load_dataset, row_count, save_gp_dataframe
from fastf1_data_fetcher import fetch_event_schedule, is_sprint_weekend


def build_historical_dataset(start_year, end_year, skip_existing=True, **build_kwargs):
    stored = get_stored_keys() if skip_existing else set()
    today = pd.Timestamp(datetime.now().date())

    succeeded, skipped, sprint_skipped, failed = 0, 0, 0, 0

    for year in range(start_year, end_year + 1):
        schedule = fetch_event_schedule(year)
        real_events = schedule[(schedule['RoundNumber'] > 0) & (schedule['EventDate'] < today)]

        for _, event in real_events.iterrows():
            round_number = int(event['RoundNumber'])
            event_name = event['EventName']

            if (year, round_number) in stored:
                skipped += 1
                continue

            if is_sprint_weekend(year, round_number):
                print(f"Skipping {year} R{round_number} {event_name} (sprint weekend, not yet supported)")
                sprint_skipped += 1
                continue

            print(f"Building {year} R{round_number} {event_name}...")
            try:
                df = build_gp_dataframe(year, round_number, **build_kwargs)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                continue

            save_gp_dataframe(df)
            print(f"  Saved {len(df)} driver rows.")
            succeeded += 1

    print(f"\nDone. Succeeded: {succeeded}, skipped (already stored): {skipped}, "
          f"skipped (sprint): {sprint_skipped}, failed: {failed}")
    print(f"Total rows in store: {row_count()}")


if __name__ == '__main__':
    current_year = datetime.now().year
    build_historical_dataset(2018, current_year, overtaking_years_back=4, history_years_back=5)

    full_dataset = load_dataset()
    print(f"\nFinal dataset shape: {full_dataset.shape}")
    print(full_dataset.head())
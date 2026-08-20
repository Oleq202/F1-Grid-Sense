from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.fastf1_data_fetcher import fetch_event_schedule, is_sprint_weekend
    from src.dataset_store import update_race_results, get_incomplete_races
except (ImportError, ValueError):
    from fastf1_data_fetcher import fetch_event_schedule, is_sprint_weekend
    from dataset_store import update_race_results, get_incomplete_races


def update_completed_races():
    today = pd.Timestamp(datetime.now().date())
    
    incomplete = get_incomplete_races()
    
    if not incomplete:
        print("No incomplete races to update")
        return
    
    schedules = {}
    
    for year, round_num in incomplete:
        if year not in schedules:
            try:
                schedules[year] = fetch_event_schedule(year)
            except Exception as e:
                print(f"Failed to fetch schedule for {year}: {e}")
                continue

        schedule = schedules[year]
        if is_sprint_weekend(year, round_num):
            print(f"Skipping {year} R{round_num} (sprint weekend)")
            continue
        
        event_row = schedule[(schedule['RoundNumber'] == round_num)]
        if event_row.empty:
            continue
        
        event_date = event_row.iloc[0]['EventDate']
        
        if today >= event_date:
            print(f"Updating race results for {year} R{round_num}")
            success = update_race_results(year, round_num)
            if not success:
                print(f"Failed to update {year} R{round_num}")


if __name__ == '__main__':
    update_completed_races()

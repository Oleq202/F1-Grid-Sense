from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fastf1_data_fetcher import fetch_event_schedule, is_sprint_weekend
from src.build_pre_race_dataframe import build_pre_race_dataframe
from src.dataset_store import save_pre_race_dataframe, get_stored_keys


def scrape_next_race():
    today = pd.Timestamp(datetime.now().date())
    current_year = today.year
    
    schedule = fetch_event_schedule(current_year)
    real_events = schedule[(schedule['RoundNumber'] > 0)]
    
    upcoming = real_events[real_events['EventDate'] > today]
    
    if upcoming.empty:
        print("No upcoming races found")
        return
    
    for _, event in upcoming.iterrows():
        year = current_year
        round_num = int(event['RoundNumber'])
        event_name = event['EventName']
        event_date = event['EventDate']
        
        if is_sprint_weekend(year, round_num):
            print(f"Skipping {year} R{round_num} {event_name} (sprint weekend)")
            continue
        
        if (year, round_num) in get_stored_keys():
            print(f"Already scraped {year} R{round_num} {event_name}")
            continue
        
        estimated_quali_date = event_date - timedelta(days=1)
        
        if today >= estimated_quali_date:
            print(f"Scraping pre-race data for {year} R{round_num} {event_name}")
            try:
                df = build_pre_race_dataframe(year, round_num)
                save_pre_race_dataframe(df)
                print(f"Successfully saved pre-race data for {len(df)} drivers")
                return
            except Exception as e:
                print(f"Failed to scrape {year} R{round_num}: {e}")
                continue
        else:
            print(f"Qualifying not yet completed for {year} R{round_num} {event_name}")
            return
    
    print("No eligible races to scrape")


if __name__ == '__main__':
    scrape_next_race()

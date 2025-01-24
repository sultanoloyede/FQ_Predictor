import os
import pandas as pd
from nba_api.stats.static import teams
import logging

# Configure Logging
logging.basicConfig(filename='../logs/fetch_teams.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_and_save_teams(csv_path):
    try:
        # Fetch all NBA teams
        nba_teams = teams.get_teams()
        
        # Create DataFrame
        teams_df = pd.DataFrame(nba_teams)
        
        # Select and rename relevant columns
        teams_df = teams_df[['id', 'full_name']]
        teams_df.rename(columns={'id': 'team_id', 'full_name': 'team_name'}, inplace=True)
        
        # Save to CSV
        teams_df.to_csv(csv_path, index=False)
        logging.info(f"Successfully saved teams data to {csv_path}.")
    except Exception as e:
        logging.error(f"Error fetching or saving teams data: {e}")

def main():
    # Define the path to save teams.csv
    csv_path = os.path.join('..', 'data', 'teams.csv')
    
    # Fetch and save teams data
    fetch_and_save_teams(csv_path)

if __name__ == "__main__":
    main()

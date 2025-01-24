import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, playergamelog, leaguegamefinder
from nba_api.stats.static import teams
from datetime import datetime
import time
import logging

# Configure Logging
logging.basicConfig(
    filename='../logs/initialize_csvs.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Define Paths
PLAYER_FOLDER = '../data/players/'
TEAMS_CSV_PATH = '../data/teams.csv'

# Ensure Player Folder Exists
os.makedirs(PLAYER_FOLDER, exist_ok=True)

# Function to Determine Current, Last Completed, and Previous Seasons
def get_seasons():
    today = datetime.today()
    year = today.year
    month = today.month

    season = ""
    seasons = []
    while(season != '2021-22'):
        if len(seasons) < 1:
            if month >= 10:
                current_season_start = year
                current_season_end = year + 1
                
                season = f"{current_season_start}-{str(current_season_end)[-2:]}"

                seasons.append(season)

                current_season_start -= 1
                current_season_end -= 1
            else:
                current_season_start = year - 1
                current_season_end = year
                
                season = f"{current_season_start}-{str(current_season_end)[-2:]}"

                seasons.append(season)

                current_season_start -= 1
                current_season_end -= 1

        else:
            season = f"{current_season_start}-{str(current_season_end)[-2:]}"

            seasons.append(season)

            current_season_start -= 1
            current_season_end -= 1
    return seasons

# Function to Fetch team stats
def fetch_team_stats(seasons):
    game_data = pd.DataFrame()
    for season in seasons:
        print(season)
        # Fetch games for the season
        game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        games = game_finder.get_data_frames()[0]  # Convert response to DataFrame
        game_data = pd.concat([game_data, games], ignore_index=True)

    # Convert column 'B' to numeric
    game_data['GAME_ID'] = pd.to_numeric(game_data['GAME_ID'])
    game_data = game_data[game_data['GAME_ID'] <= 99999999]
    game_data['GAME_ID'] = game_data['GAME_ID'].astype(str).str.zfill(10)

    print(game_data.columns)

    game_data.to_csv('../data/features/all_games.csv')

seasons = get_seasons()
fetch_team_stats(seasons)
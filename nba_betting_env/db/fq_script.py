# Import necessary libraries
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder, boxscoresummaryv2, teamgamelog
import pandas as pd
import numpy as np
import time

# Retrieve the list of NBA teams
nba_teams = teams.get_teams()

# Create mappings between team names, IDs, and abbreviations
team_dict = {team['full_name']: team['id'] for team in nba_teams}
team_abbreviation_to_id = {team['abbreviation']: team['id'] for team in nba_teams}
team_id_to_abbreviation = {team['id']: team['abbreviation'] for team in nba_teams}

# Define the seasons of interest
seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']

# Initialize an empty DataFrame to hold all games
all_games = pd.DataFrame()

from nba_api.stats.endpoints import leaguegamelog

# Initialize DataFrame to hold all games
all_games = pd.DataFrame()

for season in seasons:
    print(f'\nProcessing season {season}...')

    time.sleep(1)  # Respect rate limits

    try:
        gamelog = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star='Regular Season'
        )
        games = gamelog.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching games for season {season}: {e}")
        continue  # Skip to the next season

    # Convert game dates to datetime
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

    # Extract start and end years from the season string
    start_year_str, end_year_str = season.split('-')
    start_year = int(start_year_str)
    end_year = int('20' + end_year_str) if int(end_year_str) < 50 else int('19' + end_year_str)

    # Define the All-Star break date
    all_star_break = pd.to_datetime(f'02/15/{end_year}')

    # Filter games before the All-Star break
    games_before_asb = games[games['GAME_DATE'] < all_star_break]

    # Drop duplicates based on 'GAME_ID'
    games_before_asb_unique = games_before_asb.drop_duplicates(subset=['GAME_ID'])

    # Append to all_games
    all_games = pd.concat([all_games, games_before_asb_unique], ignore_index=True)

    # Test: Print the number of unique games collected for the season
    print(f"Number of unique games before All-Star break in {season}: {len(games_before_asb_unique)}")

# Filter for home games and extract opponent information
home_games = all_games[all_games['MATCHUP'].str.contains(r'vs\.')].copy()

# Extract opponent abbreviation
home_games['OPPONENT_ABBREVIATION'] = home_games['MATCHUP'].str.extract(r'vs\. (\w+)', expand=False)

# Map team abbreviations to IDs
home_games['AWAY_TEAM_ID'] = home_games['OPPONENT_ABBREVIATION'].map(team_abbreviation_to_id)

# Rename TEAM_ID to HOME_TEAM_ID
home_games.rename(columns={'TEAM_ID': 'HOME_TEAM_ID'}, inplace=True)

# Select required columns
home_games = home_games[['GAME_ID', 'GAME_DATE', 'SEASON_ID', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']]

# Test: Check for missing AWAY_TEAM_IDs
missing_away_ids = home_games[home_games['AWAY_TEAM_ID'].isnull()]
if not missing_away_ids.empty:
    print("\nGames with missing AWAY_TEAM_IDs:")
    print(missing_away_ids[['GAME_ID', 'OPPONENT_ABBREVIATION']])

# Initialize list to hold game data
final_data = []

# Cache for team game logs to avoid redundant API calls
team_game_logs_cache = {}

# Loop through each game to collect first-quarter points and other stats
for index, game in home_games.iterrows():
    game_id = game['GAME_ID']
    home_team_id = game['HOME_TEAM_ID']
    away_team_id = game['AWAY_TEAM_ID']
    game_date = game['GAME_DATE']
    season = game['SEASON_ID']
    
    # Introduce a delay to respect rate limits
    time.sleep(0.6)
    
    # Get box score summary to extract first-quarter points
    try:
        boxscore = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        line_score = boxscore.line_score.get_data_frame()
    except Exception as e:
        print(f"Error fetching box score for game {game_id}: {e}")
        continue  # Skip to the next game
    
    # Extract first-quarter points for both teams
    home_line = line_score.loc[line_score['TEAM_ID'] == home_team_id]
    away_line = line_score.loc[line_score['TEAM_ID'] == away_team_id]
    
    if home_line.empty or away_line.empty:
        print(f"Skipping game {game_id}: Missing line score data.")
        continue  # Skip if data is missing
    
    home_first_q_points = pd.to_numeric(home_line['PTS_QTR1'].values[0], errors='coerce')
    away_first_q_points = pd.to_numeric(away_line['PTS_QTR1'].values[0], errors='coerce')
    
    if pd.isnull(home_first_q_points) or pd.isnull(away_first_q_points):
        print(f"Skipping game {game_id}: Missing first-quarter points.")
        continue  # Skip if data is missing
    
    total_first_q_points = home_first_q_points + away_first_q_points
    
    # Create a record for the game
    game_record = {
        'Game_ID': game_id,
        'Date': game_date,
        'Season': season,
        'Home_Team_ID': home_team_id,
        'Away_Team_ID': away_team_id,
        'Home_FirstQ_Points': home_first_q_points,
        'Away_FirstQ_Points': away_first_q_points,
        'Total_FirstQ_Points': total_first_q_points
    }
    
    # Append the data
    final_data.append(game_record)
    
    # Test: Print progress every 100 games
    if len(final_data) % 100 == 0:
        print(f"Collected data for {len(final_data)} games.")
        print(game_record)









import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog
from nba_api.stats.static import teams
from datetime import datetime
import time
import logging
from requests.exceptions import RequestException

# Configure Logging
logging.basicConfig(
    filename='../logs/fetch_players.log',
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

    if month >= 10:
        current_season_start = year
        current_season_end = year + 1
        last_completed_season_start = year - 1
        last_completed_season_end = year
    else:
        current_season_start = year - 1
        current_season_end = year
        last_completed_season_start = year - 2
        last_completed_season_end = year - 1

    current_season = f"{current_season_start}-{str(current_season_end)[-2:]}"  # e.g., '2024-25'
    last_completed_season = f"{last_completed_season_start}-{str(last_completed_season_end)[-2:]}"  # e.g., '2023-24'

    # Correctly compute the previous_season
    previous_season_start_year = last_completed_season_start - 1
    previous_season_end_year = last_completed_season_end - 1
    previous_season_suffix = f"{previous_season_end_year % 100:02d}"  # Ensures two-digit format
    previous_season = f"{previous_season_start_year}-{previous_season_suffix}"  # e.g., '2022-23'

    return current_season, last_completed_season, previous_season

# Function to Fetch Top N PPG Players per Team
def fetch_top_n_ppg_players(current_season, last_completed_season, teams_df, top_n=4):
    try:
        # Fetch player stats for the last completed season
        stats_last = leaguedashplayerstats.LeagueDashPlayerStats(
            season=last_completed_season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_last_season = stats_last.get_data_frames()[0]

        # Fetch player stats for the current ongoing season
        stats_current = leaguedashplayerstats.LeagueDashPlayerStats(
            season=current_season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_current_season = stats_current.get_data_frames()[0]

        # Combine both seasons' data
        combined_df = pd.concat([df_last_season, df_current_season], ignore_index=True)

        # Remove duplicate player-season entries by aggregating PTS (sum or mean)
        # Here, we'll take the mean PTS across seasons
        combined_df = combined_df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID'], as_index=False).agg({'PTS': 'mean'}).rename(columns={'PTS': 'AVG_PTS'})

        # Merge with teams_df to get team names if not already present
        if 'team_name' not in combined_df.columns:
            combined_df = combined_df.merge(teams_df, left_on='TEAM_ID', right_on='team_id', how='left')

        # Handle cases where team information is missing after the merge
        missing_teams = combined_df[combined_df['team_name'].isnull()]['TEAM_ID'].unique()
        if len(missing_teams) > 0:
            logging.warning(f"Missing team names for TEAM_IDs: {missing_teams}")

        # Sort by TEAM_ID and AVG_PTS to get top N PPG players per team
        top_players = combined_df.sort_values(['TEAM_ID', 'AVG_PTS'], ascending=[True, False]).groupby('TEAM_ID').head(top_n)

        # Log the number of players fetched per team
        players_per_team = top_players.groupby('TEAM_ID').size()
        for team_id, count in players_per_team.items():
            team_name = teams_df[teams_df['team_id'] == team_id]['team_name'].values
            if len(team_name) > 0:
                team_name = team_name[0]
            else:
                team_name = "Unknown Team"
            logging.info(f"Team: {team_name} (ID: {team_id}) - Top {count} PPG Players Fetched.")

        # Log total players fetched
        total_players_fetched = len(top_players)
        expected_total = top_n * len(teams_df)
        logging.info(f"Total players fetched: {total_players_fetched} out of expected {expected_total}.")

        # Identify teams missing the expected number of players
        teams_fetched = top_players['TEAM_ID'].unique()
        teams_missing = set(teams_df['team_id']) - set(teams_fetched)
        if teams_missing:
            missing_team_names = teams_df[teams_df['team_id'].isin(teams_missing)]['team_name'].tolist()
            logging.warning(f"Missing top players for teams: {missing_team_names}")
        else:
            logging.info("All teams have their top PPG players fetched.")

        return top_players
    except Exception as e:
        logging.error(f"Error fetching top {top_n} PPG players: {e}")
        return pd.DataFrame()

# Function to Get Existing Player IDs
def get_existing_player_ids():
    existing_files = os.listdir(PLAYER_FOLDER)
    player_ids = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files if filename.startswith('player_') and filename.endswith('.csv')]
    return player_ids

# Function to Initialize a Player CSV (Same as in initialize_csvs.py)
def initialize_player_csv(player, seasons_to_fetch):
    player_id = player['PLAYER_ID']  # Updated to match API column
    player_name = player.get('PLAYER_NAME', 'Unknown')  # Use .get to handle missing names
    team_id = player['TEAM_ID']

    # Define Player CSV Path
    csv_path = os.path.join(PLAYER_FOLDER, f"player_{player_id}.csv")

    if os.path.exists(csv_path):
        logging.info(f"CSV for player {player_name} (ID: {player_id}) already exists. Skipping initialization.")
        return

    # Fetch Game Logs with Season-Specific Limits
    game_logs = fetch_player_game_logs(player_id, seasons_to_fetch)

    if game_logs.empty:
        logging.warning(f"No game logs found for player {player_name} (ID: {player_id}). Skipping.")
        return

    # Standardize column names to uppercase
    game_logs.rename(columns=lambda x: x.upper(), inplace=True)

    # Log the columns present in game_logs
    logging.info(f"Columns in game_logs for player {player_name} (ID: {player_id}): {game_logs.columns.tolist()}")

    # Select Relevant Columns (Added 'MIN' and 'PF')
    relevant_columns = ['GAME_ID', 'GAME_DATE', 'PTS', 'FGA', 'MIN', 'PF']

    # Check if all relevant columns are present
    missing_columns = [col for col in relevant_columns if col not in game_logs.columns]
    if missing_columns:
        logging.error(f"Missing columns {missing_columns} for player {player_name} (ID: {player_id}). Skipping.")
        return

    game_logs = game_logs[relevant_columns]

    # Rename Columns for Clarity
    game_logs.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'PTS': 'points',
        'FGA': 'fga',
        'MIN': 'minutes',  # New Column
        'PF': 'fouls'       # New Column
    }, inplace=True)

    # Add Player and Team Information
    game_logs['player_id'] = player_id
    game_logs['player_name'] = player_name
    game_logs['team_id'] = team_id

    # Save to CSV
    try:
        game_logs.to_csv(csv_path, index=False)
        logging.info(f"Initialized CSV for player {player_name} (ID: {player_id}).")
    except Exception as e:
        logging.error(f"Error saving CSV for player {player_name} (ID: {player_id}): {e}")

# Function to Fetch Game Logs for a Player with Season-Specific Limits (Same as in initialize_csvs.py)
def fetch_player_game_logs(player_id, seasons, limit_previous_season=20):
    try:
        all_game_logs = pd.DataFrame()

        for season_info in seasons:
            season = season_info['season']
            limit = season_info.get('limit', None)  # None means no limit

            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            games_df = gamelog.get_data_frames()[0]

            # Convert GAME_DATE to datetime
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'], format='%b %d, %Y')

            # Sort by GAME_DATE descending to get the most recent games first
            games_df = games_df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

            if limit:
                games_df = games_df.head(limit)

            all_game_logs = pd.concat([all_game_logs, games_df], ignore_index=True)
            time.sleep(0.5)  # Respect API rate limits

        # After fetching, sort all games by GAME_DATE ascending (oldest first)
        all_game_logs = all_game_logs.sort_values('GAME_DATE').reset_index(drop=True)

        return all_game_logs
    except Exception as e:
        logging.error(f"Error fetching game logs for player {player_id}: {e}")
        return pd.DataFrame()

# Function to Update an Existing Player CSV with New Games
def update_player_csv(player, seasons_to_fetch):
    player_id = player['PLAYER_ID']
    player_name = player.get('PLAYER_NAME', 'Unknown')
    team_id = player['TEAM_ID']

    # Define Player CSV Path
    csv_path = os.path.join(PLAYER_FOLDER, f"player_{player_id}.csv")

    if not os.path.exists(csv_path):
        logging.warning(f"CSV for player {player_name} (ID: {player_id}) does not exist. Skipping update.")
        return

    try:
        # Load existing game logs
        existing_game_logs = pd.read_csv(csv_path, parse_dates=['game_date'])

        if existing_game_logs.empty:
            last_game_date = None
        else:
            last_game_date = existing_game_logs['game_date'].max()

        # Fetch all relevant game logs
        new_game_logs = fetch_player_game_logs(player_id, seasons_to_fetch)

        if new_game_logs.empty:
            logging.info(f"No new game logs found for player {player_name} (ID: {player_id}).")
            return

        # Filter games that are newer than the last_game_date
        if last_game_date:
            # Ensure that the 'game_date' column is in datetime format
            new_game_logs['GAME_DATE'] = pd.to_datetime(new_game_logs['GAME_DATE'], format='%b %d, %Y')
            new_games = new_game_logs[new_game_logs['GAME_DATE'] > last_game_date]
        else:
            new_games = new_game_logs

        if new_games.empty:
            logging.info(f"No new games to update for player {player_name} (ID: {player_id}).")
            return

        # Standardize column names to uppercase
        new_games.rename(columns=lambda x: x.upper(), inplace=True)

        # Select Relevant Columns (Added 'MIN' and 'PF')
        relevant_columns = ['GAME_ID', 'GAME_DATE', 'PTS', 'FGA', 'MIN', 'PF']

        # Check if all relevant columns are present
        missing_columns = [col for col in relevant_columns if col not in new_games.columns]
        if missing_columns:
            logging.error(f"Missing columns {missing_columns} for player {player_name} (ID: {player_id}). Skipping update.")
            return

        new_games = new_games[relevant_columns]

        # Rename Columns for Clarity
        new_games.rename(columns={
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'PTS': 'points',
            'FGA': 'fga',
            'MIN': 'minutes',  # New Column
            'PF': 'fouls'       # New Column
        }, inplace=True)

        # Add Player and Team Information
        new_games['player_id'] = player_id
        new_games['player_name'] = player_name
        new_games['team_id'] = team_id

        # Append new games to existing_game_logs
        updated_game_logs = pd.concat([existing_game_logs, new_games], ignore_index=True)

        # Sort by game_date ascending
        updated_game_logs = updated_game_logs.sort_values('game_date').reset_index(drop=True)

        # Save back to CSV
        updated_game_logs.to_csv(csv_path, index=False)
        logging.info(f"Updated CSV for player {player_name} (ID: {player_id}) with {len(new_games)} new games.")
    except Exception as e:
        logging.error(f"Error updating CSV for player {player_name} (ID: {player_id}): {e}")

# Main Fetch Players Function
def main():
    current_season, last_completed_season, previous_season = get_seasons()
    logging.info(f"Starting fetch_players.py for current season {current_season}, last completed season {last_completed_season}, and previous season {previous_season}.")

    # Load teams.csv to get team_id and team_name
    try:
        teams_df = pd.read_csv(TEAMS_CSV_PATH)
    except Exception as e:
        logging.error(f"Error reading teams.csv: {e}")
        return

    # Fetch Current Top 4 PPG Players
    top_players_df = fetch_top_n_ppg_players(current_season, last_completed_season, teams_df, top_n=4)

    if top_players_df.empty:
        logging.error("No top players fetched. Exiting fetch_players.py.")
        return

    # Get Existing Player IDs
    existing_player_ids = get_existing_player_ids()

    # Prepare Seasons to Fetch for Initialization
    seasons_to_fetch_init = [
        {'season': current_season, 'limit': None},          # All games from current season
        {'season': last_completed_season, 'limit': None},   # All games from last completed season
        {'season': previous_season, 'limit': 20}            # Last 20 games from previous season
    ]

    # Prepare Seasons to Fetch for Updating
    seasons_to_fetch_update = [
        {'season': current_season, 'limit': None},          # All games from current season
        {'season': last_completed_season, 'limit': None},   # All games from last completed season
        {'season': previous_season, 'limit': 20}            # Last 20 games from previous season
    ]

    # Iterate through each top player
    for _, player in top_players_df.iterrows():
        player_id = player['PLAYER_ID']
        player_name = player.get('PLAYER_NAME', 'Unknown')

        if player_id in existing_player_ids:
            # Update existing player CSV
            logging.info(f"Updating existing CSV for player {player_name} (ID: {player_id}).")
            update_player_csv(player, seasons_to_fetch_update)
        else:
            # Initialize new player CSV
            logging.info(f"Initializing new CSV for player {player_name} (ID: {player_id}).")
            initialize_player_csv(player, seasons_to_fetch_init)

        # Respect API rate limits
        time.sleep(0.5)

    logging.info("Completed fetch_players.py execution.")

if __name__ == "__main__":
    main()

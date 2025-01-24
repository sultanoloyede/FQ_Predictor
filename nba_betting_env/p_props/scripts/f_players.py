import os
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    playergamelog,
    leaguegamelog
)
from nba_api.stats.static import teams
from datetime import datetime, timedelta
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

    current_season = f"{current_season_start}-{str(current_season_end)[-2:]}"
    last_completed_season = f"{last_completed_season_start}-{str(last_completed_season_end)[-2:]}"
    # Compute previous season
    previous_season_start_year = last_completed_season_start - 1
    previous_season_end_year = last_completed_season_end - 1
    previous_season = f"{previous_season_start_year}-{previous_season_end_year % 100:02d}"

    return current_season, last_completed_season, previous_season

def fetch_active_player_ids(current_season, days_back=8):
    """
    Fetch a set of players who have played at least one game
    in the last `days_back` days during the current season using
    LeagueDashPlayerStats with date filtering.
    """
    try:
        # Define cutoff date 8 days ago in 'MM/DD/YYYY' format
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%m/%d/%Y')

        # Fetch player stats since cutoff_date; only players active since that date will appear
        recent_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=current_season,
            date_from_nullable=cutoff_date,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_recent = recent_stats.get_data_frames()[0]

        # Normalize column names to uppercase for consistency
        df_recent.columns = [col.upper() for col in df_recent.columns]

        # Check if DataFrame is empty or missing PLAYER_ID
        if df_recent.empty:
            logging.warning("No recent stats found in the last 8 days.")
            return set()
        if 'PLAYER_ID' not in df_recent.columns:
            logging.error(f"'PLAYER_ID' column not found. Columns available: {df_recent.columns.tolist()}")
            return set()

        # Extract unique active player IDs
        active_player_ids = set(df_recent['PLAYER_ID'].unique())
        logging.info(f"Found {len(active_player_ids)} active players in the last {days_back} days.")
        return active_player_ids

    except Exception as e:
        logging.error(f"Error fetching active players for last {days_back} days: {e}")
        return set()


def fetch_top_n_ppg_players(current_season, teams_df, active_player_ids, top_n=4):
    """
    Fetch top N players for the current season, but only if they have played
    within the last 'days_back' days (as defined by `active_player_ids`).
    """
    try:
        # Fetch current-season player stats
        stats_current = leaguedashplayerstats.LeagueDashPlayerStats(
            season=current_season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_current_season = stats_current.get_data_frames()[0]

        # Filter down to only "active" players
        df_current_season = df_current_season[
            df_current_season['PLAYER_ID'].isin(active_player_ids)
        ]
        
        if df_current_season.empty:
            logging.warning(
                "No players found for current season who meet the recent-activity criterion."
            )
            return pd.DataFrame()

        # Rename 'PTS' to 'AVG_PTS' for clarity
        df_current_season['AVG_PTS'] = df_current_season['PTS']

        # Merge with teams_df to get team names
        df_current_season = df_current_season.merge(
            teams_df,
            left_on='TEAM_ID',
            right_on='team_id',
            how='left'
        )

        # Identify any missing team names
        missing_teams = df_current_season[df_current_season['team_name'].isnull()]['TEAM_ID'].unique()
        if len(missing_teams) > 0:
            logging.warning(f"Missing team names for TEAM_IDs: {missing_teams}")

        # Sort by team + AVG_PTS, pick top_n per team
        top_players = (
            df_current_season
            .sort_values(['TEAM_ID', 'AVG_PTS'], ascending=[True, False])
            .groupby('TEAM_ID')
            .head(top_n)
        )

        # Log how many players per team
        players_per_team = top_players.groupby('TEAM_ID').size()
        for team_id, count in players_per_team.items():
            team_name_series = teams_df.loc[teams_df['team_id'] == team_id, 'team_name']
            team_name = team_name_series.values[0] if not team_name_series.empty else "Unknown Team"
            logging.info(f"Team: {team_name} (ID: {team_id}) - Top {count} Active PPG Players.")

        total_players_fetched = len(top_players)
        expected_total = top_n * len(teams_df)
        logging.info(
            f"Total active players fetched: {total_players_fetched} out of expected {expected_total}."
        )

        # Identify teams that got fewer than top_n
        teams_fetched = top_players['TEAM_ID'].unique()
        teams_missing = set(teams_df['team_id']) - set(teams_fetched)
        if teams_missing:
            missing_team_names = teams_df[teams_df['team_id'].isin(teams_missing)]['team_name'].tolist()
            logging.warning(f"Some teams have no active players: {missing_team_names}")
        else:
            logging.info("All teams have at least one active player in the last 8 days.")

        return top_players

    except Exception as e:
        logging.error(f"Error fetching top {top_n} active PPG players for {current_season}: {e}")
        return pd.DataFrame()

def get_existing_player_ids():
    existing_files = os.listdir(PLAYER_FOLDER)
    player_ids = [
        int(filename.split('_')[1].split('.')[0]) 
        for filename in existing_files 
        if filename.startswith('player_') and filename.endswith('.csv')
    ]
    return player_ids

def fetch_player_game_logs(player_id, seasons, limit_previous_season=20):
    try:
        all_game_logs = pd.DataFrame()
        for season_info in seasons:
            season = season_info['season']
            limit = season_info.get('limit', None)

            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            games_df = gamelog.get_data_frames()[0]

            # Convert GAME_DATE to datetime
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'], format='%b %d, %Y')
            games_df = games_df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

            if limit:
                games_df = games_df.head(limit)

            all_game_logs = pd.concat([all_game_logs, games_df], ignore_index=True)
            time.sleep(0.5)  # Rate-limit respect

        # After fetching, sort by date ascending
        all_game_logs = all_game_logs.sort_values('GAME_DATE').reset_index(drop=True)
        return all_game_logs

    except Exception as e:
        logging.error(f"Error fetching game logs for player {player_id}: {e}")
        return pd.DataFrame()

def initialize_player_csv(player, seasons_to_fetch):
    player_id = player['PLAYER_ID']
    player_name = player.get('PLAYER_NAME', 'Unknown')
    team_id = player['TEAM_ID']

    csv_path = os.path.join(PLAYER_FOLDER, f"player_{player_id}.csv")

    if os.path.exists(csv_path):
        logging.info(f"CSV for player {player_name} (ID: {player_id}) already exists. Skipping initialization.")
        return

    # Fetch game logs for specified seasons
    game_logs = fetch_player_game_logs(player_id, seasons_to_fetch)

    if game_logs.empty:
        logging.warning(f"No game logs found for player {player_name} (ID: {player_id}). Skipping.")
        return

    # Standardize columns
    game_logs.rename(columns=lambda x: x.upper(), inplace=True)

    relevant_columns = ['GAME_ID', 'GAME_DATE', 'PTS', 'FGA', 'MIN', 'PF']
    missing_columns = [col for col in relevant_columns if col not in game_logs.columns]
    if missing_columns:
        logging.error(f"Missing columns {missing_columns} for player {player_name} (ID: {player_id}). Skipping.")
        return

    game_logs = game_logs[relevant_columns]

    # Rename for clarity
    game_logs.rename(columns={
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'PTS': 'points',
        'FGA': 'fga',
        'MIN': 'minutes',
        'PF': 'fouls'
    }, inplace=True)

    # Add player/team info
    game_logs['player_id'] = player_id
    game_logs['player_name'] = player_name
    game_logs['team_id'] = team_id

    # Save
    try:
        game_logs.to_csv(csv_path, index=False)
        logging.info(f"Initialized CSV for player {player_name} (ID: {player_id}).")
    except Exception as e:
        logging.error(f"Error saving CSV for player {player_name} (ID: {player_id}): {e}")

def update_player_csv(player, seasons_to_fetch):
    player_id = player['PLAYER_ID']
    player_name = player.get('PLAYER_NAME', 'Unknown')
    team_id = player['TEAM_ID']

    csv_path = os.path.join(PLAYER_FOLDER, f"player_{player_id}.csv")

    if not os.path.exists(csv_path):
        logging.warning(f"CSV for player {player_name} (ID: {player_id}) does not exist. Skipping update.")
        return

    try:
        existing_game_logs = pd.read_csv(csv_path, parse_dates=['game_date'])
        last_game_date = existing_game_logs['game_date'].max() if not existing_game_logs.empty else None

        new_game_logs = fetch_player_game_logs(player_id, seasons_to_fetch)
        if new_game_logs.empty:
            logging.info(f"No new game logs found for player {player_name} (ID: {player_id}).")
            return

        # Filter only games newer than last_game_date
        if last_game_date:
            new_game_logs['GAME_DATE'] = pd.to_datetime(new_game_logs['GAME_DATE'], format='%b %d, %Y')
            new_games = new_game_logs[new_game_logs['GAME_DATE'] > last_game_date]
        else:
            new_games = new_game_logs

        if new_games.empty:
            logging.info(f"No new games to update for player {player_name} (ID: {player_id}).")
            return

        # Standardize columns
        new_games.rename(columns=lambda x: x.upper(), inplace=True)

        relevant_columns = ['GAME_ID', 'GAME_DATE', 'PTS', 'FGA', 'MIN', 'PF']
        missing_columns = [col for col in relevant_columns if col not in new_games.columns]
        if missing_columns:
            logging.error(f"Missing columns {missing_columns} for player {player_name} (ID: {player_id}). Skipping update.")
            return

        new_games = new_games[relevant_columns]

        new_games.rename(columns={
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'PTS': 'points',
            'FGA': 'fga',
            'MIN': 'minutes',
            'PF': 'fouls'
        }, inplace=True)

        # Add player/team info
        new_games['player_id'] = player_id
        new_games['player_name'] = player_name
        new_games['team_id'] = team_id

        updated_game_logs = pd.concat([existing_game_logs, new_games], ignore_index=True)
        updated_game_logs = updated_game_logs.sort_values('game_date').reset_index(drop=True)

        updated_game_logs.to_csv(csv_path, index=False)
        logging.info(f"Updated CSV for player {player_name} (ID: {player_id}) with {len(new_games)} new games.")
    except Exception as e:
        logging.error(f"Error updating CSV for player {player_name} (ID: {player_id}): {e}")

def main():
    current_season, last_completed_season, previous_season = get_seasons()
    logging.info(
        f"Starting fetch_players.py for current season {current_season}, "
        f"last completed season {last_completed_season}, and previous season {previous_season}."
    )

    # Read teams.csv
    try:
        teams_df = pd.read_csv(TEAMS_CSV_PATH)
    except Exception as e:
        logging.error(f"Error reading teams.csv: {e}")
        return

    # 1) Get the set of active players who played in the last 8 days
    active_player_ids = fetch_active_player_ids(current_season, days_back=8)
    if not active_player_ids:
        logging.error("No active players found. Exiting fetch_players.py.")
        return

    # 2) Fetch top 4 scorers among these active players
    top_players_df = fetch_top_n_ppg_players(current_season, teams_df, active_player_ids, top_n=4)

    if top_players_df.empty:
        logging.error("No top players fetched (all filtered out or no data). Exiting fetch_players.py.")
        return

    existing_player_ids = get_existing_player_ids()

    # Seasons logic remains the same for CSV initialization and updates
    seasons_to_fetch_init = [
        {'season': current_season, 'limit': None},
        {'season': last_completed_season, 'limit': None},
        {'season': previous_season, 'limit': 20}
    ]

    seasons_to_fetch_update = [
        {'season': current_season, 'limit': None},
        {'season': last_completed_season, 'limit': None},
        {'season': previous_season, 'limit': 20}
    ]

    # Initialize or update each selected top player
    for _, player in top_players_df.iterrows():
        player_id = player['PLAYER_ID']
        player_name = player.get('PLAYER_NAME', 'Unknown')

        if player_id in existing_player_ids:
            logging.info(f"Updating existing CSV for player {player_name} (ID: {player_id}).")
            update_player_csv(player, seasons_to_fetch_update)
        else:
            logging.info(f"Initializing new CSV for player {player_name} (ID: {player_id}).")
            initialize_player_csv(player, seasons_to_fetch_init)

        time.sleep(0.5)

    logging.info("Completed fetch_players.py execution.")

if __name__ == "__main__":
    main()

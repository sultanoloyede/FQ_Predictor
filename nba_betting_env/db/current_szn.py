import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoresummaryv2,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
)
from nba_api.stats.endpoints import leaguedashteamstats
import pickle
import os
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2
from tqdm import tqdm
import logging

# === Configure Logging ===
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# === Step 0: Configuration Changes ===

# Define the most recent season
most_recent_season = ['2024-25']  # Adjust this if the current season changes

# Define filenames
historical_data_filename = 'final_training_dataset.csv'  # Original historical data
new_season_data_filename = 'new_season_data.pkl'
checkpoint_filename = 'new_season_dataset_with_trends.csv'

# Load historical data
if os.path.exists(historical_data_filename):
    historical_dataset_df = pd.read_csv(historical_data_filename)
    # Ensure 'GAME_DATE' is in datetime format
    historical_dataset_df['GAME_DATE'] = pd.to_datetime(historical_dataset_df['GAME_DATE'], errors='coerce')
    print(f"Historical data loaded from '{historical_data_filename}'.")
else:
    print(f"Historical data file '{historical_data_filename}' not found. Exiting.")
    exit()

# === Function Definitions ===

def fetch_game_ids(seasons):
    """Fetches all regular-season NBA game IDs for the specified seasons."""
    all_games = []
    for season in seasons:
        print(f"Fetching games for season {season}")
        try:
            # Use the correct parameter and season format
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season, season_type_nullable='Regular Season'
            )
            games = gamefinder.get_data_frames()[0]

            print(f"Number of games fetched for season {season} before filtering: {len(games)}")

            if games.empty:
                print(f"No games found for season {season}")
                continue  # Skip to the next season

            # Convert GAME_ID to numeric, handling errors
            games['GAME_ID_INT'] = pd.to_numeric(games['GAME_ID'], errors='coerce')
            games = games.dropna(subset=['GAME_ID_INT'])
            games['GAME_ID_INT'] = games['GAME_ID_INT'].astype(int)

            # Filter out G League games
            games = games[games['GAME_ID_INT'] < 100000000]

            print(f"Number of games after filtering for season {season}: {len(games)}")

            if games.empty:
                print(f"No NBA regular-season games found for season {season} after filtering")
                continue  # Skip to the next season

            all_games.append(games)
            print(f"Added {len(games)} games for season {season}")
            time.sleep(1)  # Respect rate limits
        except Exception as e:
            logging.error(f"Error fetching games for season {season}: {e}")
            print(f"Error fetching games for season {season}: {e}")
    if all_games:
        return pd.concat(all_games, ignore_index=True)
    else:
        print("No games fetched for any season.")
        return pd.DataFrame()

def fetch_game_data(game_id):
    """Fetches and structures game data for a given GAME_ID."""
    game_data = {}
    try:
        # Fetch Box Score Summary
        boxscore_summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        line_score_df = boxscore_summary.line_score.get_data_frame()

        if line_score_df.empty:
            logging.warning(f"No data available for game {game_id}")
            print(f"No data available for game {game_id}")
            return None

        # Extract relevant columns
        required_columns = ['TEAM_ID', 'TEAM_ABBREVIATION', 'PTS_QTR1']
        missing_columns = [col for col in required_columns if col not in line_score_df.columns]
        if missing_columns:
            logging.warning(f"Missing columns {missing_columns} in line score for game {game_id}. Skipping this game.")
            print(f"Missing columns {missing_columns} in line score for game {game_id}. Skipping this game.")
            return None

        line_score_df = line_score_df[required_columns]

        # Ensure two teams are present
        if len(line_score_df) != 2:
            logging.warning(f"Invalid number of teams in game {game_id}. Expected 2, got {len(line_score_df)}.")
            print(f"Invalid number of teams in game {game_id}.")
            return None

        # Assign team and opponent data
        team1 = line_score_df.iloc[0]
        team2 = line_score_df.iloc[1]

        # Create entries for both teams
        for team, opponent in [(team1, team2), (team2, team1)]:
            data = {
                'GAME_ID': game_id,
                'TEAM_ID': team['TEAM_ID'],
                'TEAM_ABBREVIATION': team['TEAM_ABBREVIATION'],
                'OPPONENT_TEAM_ID': opponent['TEAM_ID'],
                'OPPONENT_TEAM_ABBREVIATION': opponent['TEAM_ABBREVIATION'],
                'PTS_QTR1': team['PTS_QTR1'],
                'OPP_PTS_QTR1': opponent['PTS_QTR1'],
            }
            key = f"{game_id}_{team['TEAM_ID']}"
            game_data[key] = data
        return game_data
    except Exception as e:
        logging.error(f"Error fetching data for game {game_id}: {e}")
        print(f"Error fetching data for game {game_id}: {e}")
        return None

def save_data(data, filename):
    """Saves data to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data saved to {filename}")
        print(f"Data saved to '{filename}'")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")
        print(f"Error saving data to '{filename}': {e}")

def load_data(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Data loaded from '{filename}'.")
        print(f"Data loaded from '{filename}'.")
        return data
    except FileNotFoundError:
        logging.info(f"No existing data file found at '{filename}'.")
        print(f"No existing data file found at '{filename}'.")
        return []
    except Exception as e:
        logging.error(f"Error loading data from '{filename}': {e}")
        print(f"Error loading data from '{filename}': {e}")
        return []

def get_home_away(row, historical_df):
    """Determines if the team was playing at home or away."""
    # Find the corresponding game entry in historical_df
    game = historical_df[
        (historical_df['GAME_ID'] == row['GAME_ID']) &
        (historical_df['TEAM_ID'] == row['TEAM_ID'])
    ]
    if not game.empty:
        matchup = game.iloc[0].get('MATCHUP', '')
        if '@' in matchup:
            # The team is playing away
            return 'Away'
        else:
            # The team is playing at home
            return 'Home'
    else:
        logging.warning(f"No matching game found in historical data for GAME_ID {row['GAME_ID']} and TEAM_ID {row['TEAM_ID']}.")
        return 'Unknown'

def get_recent_trend(team_id, game_date, historical_df, num_games=5):
    """Fetches the recent trend of first-quarter points for a team."""
    # Check if 'PTS_QTR1' exists in historical_df
    if 'PTS_QTR1' not in historical_df.columns:
        logging.warning(f"'PTS_QTR1' column missing in historical data. Skipping 'Recent_Trend_Team' for TEAM_ID {team_id}.")
        return []
    
    # Filter games played by the team before the current game date
    team_games = historical_df[
        (historical_df['TEAM_ID'] == team_id) &
        (historical_df['GAME_DATE'] < game_date)
    ].sort_values(by='GAME_DATE', ascending=False)
    
    # Get the last 'num_games' games
    recent_games = team_games.head(num_games)
    
    # Check if 'PTS_QTR1' exists for the recent games
    if 'PTS_QTR1' not in recent_games.columns:
        logging.warning(f"'PTS_QTR1' column missing in recent games for TEAM_ID {team_id}. Skipping this feature.")
        return []
    
    # Return the list of first-quarter points
    return recent_games['PTS_QTR1'].tolist()

def get_head_to_head_q1(team_id, opponent_id, game_date, historical_df):
    """Calculates the average total first-quarter points in head-to-head games."""
    # Check if 'Total_First_Quarter_Points' exists in historical_df
    if 'Total_First_Quarter_Points' not in historical_df.columns:
        logging.warning(f"'Total_First_Quarter_Points' column missing in historical data. Skipping 'Head_to_Head_Q1' for TEAM_ID {team_id} vs OPPONENT_TEAM_ID {opponent_id}.")
        return None
    
    # Filter past games between the two teams before the current game date
    games_between = historical_df[
        (
            (historical_df['TEAM_ID'] == team_id) &
            (historical_df['OPPONENT_TEAM_ID'] == opponent_id)
        ) &
        (historical_df['GAME_DATE'] < game_date)
    ].sort_values(by='GAME_DATE', ascending=False)
    
    # Get the last 3 games
    recent_games = games_between.head(3)
    
    # Check if 'Total_First_Quarter_Points' exists for the recent games
    if 'Total_First_Quarter_Points' not in recent_games.columns:
        logging.warning(f"'Total_First_Quarter_Points' column missing in recent head-to-head games between TEAM_ID {team_id} and OPPONENT_TEAM_ID {opponent_id}.")
        return None
    
    if not recent_games.empty:
        # Calculate the average total first-quarter points
        avg_total_q1_points = recent_games['Total_First_Quarter_Points'].mean()
        return avg_total_q1_points
    else:
        return None  # Or you can decide to return a default value

def determine_season(row):
    """Determines the NBA season based on GAME_DATE."""
    date = row['GAME_DATE']
    year = date.year
    month = date.month
    if month >= 9:
        season_start = year
    else:
        season_start = year - 1
    season_end = season_start + 1
    season = f"{season_start}-{str(season_end)[-2:]}"
    return season

def fetch_team_pace(season):
    """Fetches team pace data for a given season."""
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        pace_data = team_stats[['TEAM_ID', 'PACE']]
        return pace_data
    except Exception as e:
        logging.error(f"Error fetching pace data for season {season}: {e}")
        print(f"Error fetching pace data for season {season}: {e}")
        return pd.DataFrame()

def fetch_and_save_pace_data(seasons, filename='pace_data.pkl'):
    """Fetches and saves pace data for multiple seasons."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                pace_data = pickle.load(f)
            logging.info(f"Pace data loaded from '{filename}'.")
            print(f"Pace data loaded from '{filename}'.")
        except Exception as e:
            logging.error(f"Error loading pace data from '{filename}': {e}")
            print(f"Error loading pace data from '{filename}': {e}")
            pace_data = pd.DataFrame()
    else:
        pace_dfs = []
        for season in seasons:
            print(f"Fetching pace data for season {season}")
            pace_df = fetch_team_pace(season)
            if not pace_df.empty:
                pace_df['SEASON'] = season
                pace_dfs.append(pace_df)
            time.sleep(1)  # Respect API rate limits
        if pace_dfs:
            pace_data = pd.concat(pace_dfs, ignore_index=True)
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(pace_data, f)
                logging.info(f"Pace data saved to '{filename}'.")
                print(f"Pace data saved to '{filename}'.")
            except Exception as e:
                logging.error(f"Error saving pace data to '{filename}': {e}")
                print(f"Error saving pace data to '{filename}': {e}")
        else:
            pace_data = pd.DataFrame()
    return pace_data

def fetch_pts(game_id, team_id, cache):
    """
    Fetches the PTS (Points) for a given game and team. Utilizes a cache to prevent redundant API calls.
    """
    # Ensure game_id is formatted consistently as a 10-digit string
    game_id_str = str(game_id).zfill(10)
    key = (game_id_str, team_id)

    if key in cache:
        return cache[key]

    # Fetch from API
    try:
        time.sleep(0.6)  # Respect API rate limits
        boxscore = BoxScoreTraditionalV2(game_id=game_id_str)
        team_stats = boxscore.team_stats.get_data_frame()
        team_row = team_stats[team_stats['TEAM_ID'] == team_id]

        if not team_row.empty:
            pts = team_row['PTS'].values[0]
            cache[key] = pts  # Store in cache
            logging.info(f"PTS for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {pts}")
            print(f"PTS for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {pts}")
            return pts
        else:
            logging.warning(f"No PTS data for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan
    except Exception as e:
        logging.error(f"Error fetching PTS for GAME_ID {game_id_str}, TEAM_ID {team_id}: {e}")
        cache[key] = np.nan
        return np.nan

def fetch_off_rating(game_id, team_id, cache):
    """
    Fetches the Offensive Rating for a given game and team. Utilizes a cache to prevent redundant API calls.
    """
    # Ensure game_id is formatted consistently as a 10-digit string
    game_id_str = str(game_id).zfill(10)
    key = (game_id_str, team_id)

    if key in cache:
        return cache[key]

    # Fetch from API
    try:
        time.sleep(0.6)  # Respect API rate limits
        boxscore_adv = BoxScoreAdvancedV2(game_id=game_id_str)
        team_stats = boxscore_adv.team_stats.get_data_frame()
        team_row = team_stats[team_stats['TEAM_ID'] == team_id]

        if not team_row.empty:
            off_rating = team_row['OFF_RATING'].values[0]
            cache[key] = off_rating  # Store in cache
            logging.info(f"Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {off_rating}")
            print(f"Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {off_rating}")
            return off_rating
        else:
            logging.warning(f"No Off_Rating data for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan
    except Exception as e:
        logging.error(f"Error fetching Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id}: {e}")
        cache[key] = np.nan
        return np.nan

def fetch_fg_pct(game_id, team_id, cache):
    """
    Fetches the FG_PCT for a given game and team. Utilizes a cache to prevent redundant API calls.
    """
    # Ensure game_id is formatted consistently as a 10-digit string
    game_id_str = str(game_id).zfill(10)
    key = (game_id_str, team_id)

    if key in cache:
        return cache[key]

    # Fetch from API
    try:
        time.sleep(0.6)  # Respect API rate limits
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id_str)
        team_stats = boxscore.team_stats.get_data_frame()
        team_row = team_stats[team_stats['TEAM_ID'] == team_id]

        if not team_row.empty:
            fg_pct = team_row['FG_PCT'].values[0]
            cache[key] = fg_pct  # Store in cache
            logging.info(f"FG_PCT for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {fg_pct}")
            print(f"FG_PCT for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {fg_pct}")
            return fg_pct
        else:
            logging.warning(f"No FG_PCT data for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan
    except Exception as e:
        logging.error(f"Error fetching FG_PCT for GAME_ID {game_id_str}, TEAM_ID {team_id}: {e}")
        cache[key] = np.nan
        return np.nan

def calculate_ppg_and_off_rating(dataset_df):
    """
    Calculates Points Per Game (PPG), Opponent PPG, and Offensive Rating (Off_Rating_Avg) for each team up to the previous game.
    """
    # Ensure 'GAME_DATE' is in datetime format
    dataset_df['GAME_DATE'] = pd.to_datetime(dataset_df['GAME_DATE'], errors='coerce')

    # Sort the DataFrame
    dataset_df.sort_values(by=['TEAM_ID', 'SEASON', 'GAME_DATE'], inplace=True)

    # Calculate cumulative points excluding the current game
    dataset_df['Cumulative_Points_Previous'] = dataset_df.groupby(['TEAM_ID', 'SEASON'])['PTS'].cumsum().shift(1)
    dataset_df['Cumulative_Opp_Points_Previous'] = dataset_df.groupby(['TEAM_ID', 'SEASON'])['OPP_PTS'].cumsum().shift(1)

    # Calculate cumulative Offensive Rating excluding the current game
    dataset_df['Cumulative_Off_Rating_Previous'] = dataset_df.groupby(['TEAM_ID', 'SEASON'])['Off_Rating'].cumsum().shift(1)

    # Calculate Game Number within each TEAM_ID and SEASON
    dataset_df['Game_Number'] = dataset_df.groupby(['TEAM_ID', 'SEASON']).cumcount() + 1

    # Calculate PPG and Opponent PPG up to the previous game
    dataset_df['PPG_Team'] = dataset_df['Cumulative_Points_Previous'] / (dataset_df['Game_Number'] - 1)
    dataset_df['PPG_Opponent'] = dataset_df['Cumulative_Opp_Points_Previous'] / (dataset_df['Game_Number'] - 1)

    # Calculate Offensive Rating up to the previous game
    dataset_df['Off_Rating_Avg'] = dataset_df['Cumulative_Off_Rating_Previous'] / (dataset_df['Game_Number'] - 1)

    # For the first game, use the points and Offensive Rating from that game
    first_games_mask = dataset_df['Game_Number'] == 1
    dataset_df.loc[first_games_mask, 'PPG_Team'] = dataset_df.loc[first_games_mask, 'PTS']
    dataset_df.loc[first_games_mask, 'PPG_Opponent'] = dataset_df.loc[first_games_mask, 'OPP_PTS']
    dataset_df.loc[first_games_mask, 'Off_Rating_Avg'] = dataset_df.loc[first_games_mask, 'Off_Rating']

    print("\nCalculated PPG, Opponent PPG, and Offensive Rating up to the previous game.")

    # Drop intermediate columns
    dataset_df.drop(['Cumulative_Points_Previous', 'Cumulative_Opp_Points_Previous', 
                    'Cumulative_Off_Rating_Previous', 'Game_Number'], axis=1, inplace=True)

    # Save the updated dataset
    dataset_df.to_csv('new_season_dataset_with_ppg_off_rating.csv', index=False)
    logging.info("Updated dataset with PPG and Offensive Rating saved to 'new_season_dataset_with_ppg_off_rating.csv'")
    print("Updated dataset with PPG and Offensive Rating saved to 'new_season_dataset_with_ppg_off_rating.csv'")

def load_cache(filename):
    """Loads a cache dictionary from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                cache = pickle.load(f)
            print(f"Cache loaded from '{filename}'.")
            return cache
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading cache from '{filename}': {e}")
            return {}
    else:
        print(f"No existing cache found at '{filename}'. Starting a new cache.")
        return {}

def save_cache(cache, filename):
    """Saves a cache dictionary to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to '{filename}'.")

# === Main Script ===

# Step 1: Fetch games for the most recent season
new_games_df = fetch_game_ids(most_recent_season)

if new_games_df.empty:
    logging.error("No games fetched for the most recent season. Exiting.")
    print("No games fetched for the most recent season. Exiting.")
    exit()

# Extract unique GAME_IDs
new_game_ids = new_games_df['GAME_ID'].unique()

# Define filename for new season data
data_filename = new_season_data_filename

# Load existing new season data if available
all_new_data = load_data(data_filename)
processed_game_ids = set()

# If data was loaded, extract already processed GAME_IDs
if all_new_data:
    processed_game_ids = set([data['GAME_ID'] for data in all_new_data])
    print(f"Resuming from existing data. {len(processed_game_ids)} games already processed.")
else:
    all_new_data = []
    print("Starting data collection for the most recent season from scratch.")

# Step 2: Fetch and populate new season data
for idx, game_id in enumerate(new_game_ids):
    if game_id in processed_game_ids:
        continue  # Skip already processed games

    game_data = fetch_game_data(game_id)
    if game_data:
        all_new_data.extend(game_data.values())
        processed_game_ids.add(game_id)
    else:
        logging.warning(f"No data for game {game_id}")
        print(f"No data for game {game_id}")

    # Save data after every 100 games
    if (idx + 1) % 100 == 0 or (idx + 1) == len(new_game_ids):
        save_data(all_new_data, data_filename)
        print(f"Processed {idx + 1}/{len(new_game_ids)} games for the most recent season.")

    time.sleep(0.6)  # Respect rate limits

# Save the final new season data
save_data(all_new_data, data_filename)
print("All new season data has been collected and saved.")

# Create DataFrame from new season data
new_season_dataset_df = pd.DataFrame(all_new_data)

# Calculate total first-quarter points if columns exist
if {'PTS_QTR1', 'OPP_PTS_QTR1'}.issubset(new_season_dataset_df.columns):
    new_season_dataset_df['Total_First_Quarter_Points'] = (
        new_season_dataset_df['PTS_QTR1'] + new_season_dataset_df['OPP_PTS_QTR1']
    )
else:
    logging.warning("Columns 'PTS_QTR1' and/or 'OPP_PTS_QTR1' are missing. 'Total_First_Quarter_Points' will not be calculated.")
    new_season_dataset_df['Total_First_Quarter_Points'] = np.nan

# Extract 'GAME_ID' and 'GAME_DATE' from new_games_df
if {'GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID'}.issubset(new_games_df.columns):
    game_dates = new_games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID']].drop_duplicates()
    # Merge 'GAME_DATE' and 'MATCHUP' into new_season_dataset_df
    new_season_dataset_df = new_season_dataset_df.merge(game_dates, on=['GAME_ID', 'TEAM_ID'], how='left')
else:
    logging.warning("One or more columns ['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID'] are missing in 'new_games_df'. Skipping merge.")
    # Handle missing data accordingly or assign default values
    new_season_dataset_df['GAME_DATE'] = pd.NaT
    new_season_dataset_df['MATCHUP'] = 'Unknown'

# Convert 'GAME_DATE' to datetime format if not already parsed
if 'GAME_DATE' in new_season_dataset_df.columns and new_season_dataset_df['GAME_DATE'].dtype == 'object':
    new_season_dataset_df['GAME_DATE'] = pd.to_datetime(new_season_dataset_df['GAME_DATE'], errors='coerce')
    logging.info("'GAME_DATE' converted to datetime after merging.")
    print("'GAME_DATE' converted to datetime after merging.")

# Extract 'MONTH' from 'GAME_DATE'
if 'GAME_DATE' in new_season_dataset_df.columns:
    new_season_dataset_df['MONTH'] = new_season_dataset_df['GAME_DATE'].dt.month
else:
    logging.warning("'GAME_DATE' column missing. 'MONTH' will not be extracted.")
    new_season_dataset_df['MONTH'] = np.nan

# Check for a pre-existing checkpoint
if os.path.exists(checkpoint_filename):
    # Load the checkpoint and parse 'GAME_DATE' as datetime
    new_season_dataset_df = pd.read_csv(checkpoint_filename, parse_dates=['GAME_DATE'], infer_datetime_format=True)
    print("Loaded DataFrame from checkpoint file.")
    # If 'GAME_DATE' was not parsed, attempt to convert
    if new_season_dataset_df['GAME_DATE'].dtype == 'object':
        new_season_dataset_df['GAME_DATE'] = pd.to_datetime(new_season_dataset_df['GAME_DATE'], errors='coerce')
        logging.info("'GAME_DATE' converted to datetime after loading checkpoint.")
        print("'GAME_DATE' converted to datetime after loading checkpoint.")
else:
    # Calculations for 'Home_Away', 'Recent_Trend_Team', 'Recent_Trend_Opponent', 'Head_to_Head_Q1'
    new_season_dataset_df['Home_Away'] = new_season_dataset_df.apply(
        lambda row: get_home_away(row, historical_dataset_df), axis=1
    )
    new_season_dataset_df['Recent_Trend_Team'] = new_season_dataset_df.apply(
        lambda row: get_recent_trend(row['TEAM_ID'], row['GAME_DATE'], historical_dataset_df), axis=1
    )
    new_season_dataset_df['Recent_Trend_Opponent'] = new_season_dataset_df.apply(
        lambda row: get_recent_trend(row['OPPONENT_TEAM_ID'], row['GAME_DATE'], historical_dataset_df), axis=1
    )
    new_season_dataset_df['Head_to_Head_Q1'] = new_season_dataset_df.apply(
        lambda row: get_head_to_head_q1(row['TEAM_ID'], row['OPPONENT_TEAM_ID'], row['GAME_DATE'], historical_dataset_df), axis=1
    )

    # Save the checkpoint
    new_season_dataset_df.to_csv(checkpoint_filename, index=False)
    print(f"Checkpoint saved to '{checkpoint_filename}'.")


# Step 4: Verify and Handle Missing Starting Lineups
# Since starting lineups are removed, this step is no longer necessary.

# Step 5: Additional Processing (Season, Pace, etc.)

# Apply the function to create the 'SEASON' column
new_season_dataset_df['SEASON'] = new_season_dataset_df.apply(determine_season, axis=1)

print(new_season_dataset_df[['GAME_DATE', 'SEASON']].head())

# Get unique seasons from the 'SEASON' column
seasons = new_season_dataset_df['SEASON'].dropna().unique()
print(f"Seasons to fetch pace data for: {seasons}")

# Fetch and save pace data for these seasons
pace_data = fetch_and_save_pace_data(seasons)

# Merge team pace data if 'pace_data' is not empty
if not pace_data.empty:
    new_season_dataset_df = new_season_dataset_df.merge(
        pace_data.rename(columns={'TEAM_ID': 'TEAM_ID', 'PACE': 'PACE_Team'}),
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )

    # Merge opponent pace data
    new_season_dataset_df = new_season_dataset_df.merge(
        pace_data.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'PACE': 'PACE_Opponent'}),
        on=['OPPONENT_TEAM_ID', 'SEASON'],
        how='left'
    )

    # Ensure 'GAME_DATE' is in datetime format
    new_season_dataset_df['GAME_DATE'] = pd.to_datetime(new_season_dataset_df['GAME_DATE'], errors='coerce')

    # Sort the DataFrame
    new_season_dataset_df = new_season_dataset_df.sort_values(by=['TEAM_ID', 'SEASON', 'GAME_DATE']).reset_index(drop=True)

    # Calculate cumulative season average Pace for Team
    new_season_dataset_df['Season_Avg_Pace_Team'] = (
        new_season_dataset_df.groupby(['TEAM_ID', 'SEASON'])['PACE_Team']
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    # Calculate cumulative season average Pace for Opponent
    new_season_dataset_df['Season_Avg_Pace_Opponent'] = (
        new_season_dataset_df.groupby(['OPPONENT_TEAM_ID', 'SEASON'])['PACE_Opponent']
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    # Calculate the average of season averages (optional)
    new_season_dataset_df['Average_Season_Avg_Pace'] = (
        new_season_dataset_df['Season_Avg_Pace_Team'] + new_season_dataset_df['Season_Avg_Pace_Opponent']
    ) / 2

    # Drop the original Pace columns
    new_season_dataset_df = new_season_dataset_df.drop(['PACE_Team', 'PACE_Opponent'], axis=1)

    # Verify the new columns (optional)
    print(new_season_dataset_df[['TEAM_ID', 'GAME_DATE', 'Season_Avg_Pace_Team', 
                                 'Season_Avg_Pace_Opponent', 'Average_Season_Avg_Pace']].head(10))

    # Save the dataset up to this point
    new_season_dataset_df.to_csv('new_season_dataset_up_to_pace.csv', index=False)
    logging.info("Dataset up to pace data saved to 'new_season_dataset_up_to_pace.csv'.")
    print("Dataset up to pace data saved to 'new_season_dataset_up_to_pace.csv'.")
else:
    logging.warning("No pace data available to merge.")
    print("No pace data available to merge.")

# === Step 6: Process FG_PCT Data ===

# Load the cache or initialize a new one if it doesn't exist
fg_pct_cache = load_cache('fg_pct_cache_new_season.pkl')

# Define batch size
batch_size = 100  # Adjust based on API rate limits and performance needs
total_games = len(new_season_dataset_df)
batches = (total_games // batch_size) + (1 if total_games % batch_size > 0 else 0)

# Process in batches with a progress bar
print("Processing FG_PCT data in batches...")
for batch_num in tqdm(range(batches), desc="FG_PCT Batches"):
    # Define start and end of the batch
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_games)

    # Process each row in the current batch
    for idx in range(start_idx, end_idx):
        row = new_season_dataset_df.iloc[idx]
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        if pd.isnull(game_id) or pd.isnull(team_id):
            logging.warning(f"Skipping row {idx} due to missing GAME_ID or TEAM_ID.")
            continue
        if (game_id, team_id) not in fg_pct_cache:
            # Fetch FG_PCT if it's not already cached
            fg_pct_cache[(game_id, team_id)] = fetch_fg_pct(game_id, team_id, fg_pct_cache)

    # Save cache at the end of each batch to prevent data loss
    save_cache(fg_pct_cache, 'fg_pct_cache_new_season.pkl')

print("All FG_PCT data processed and cached.")

# Convert FG_PCT cache to DataFrame for merging
fg_pct_data = [
    {'GAME_ID': game_id, 'TEAM_ID': team_id, 'FG_PCT': fg_pct}
    for (game_id, team_id), fg_pct in fg_pct_cache.items()
]
fg_pct_df = pd.DataFrame(fg_pct_data)

# Rename columns for team FG%
fg_pct_team_df = fg_pct_df.rename(columns={'TEAM_ID': 'TEAM_ID', 'FG_PCT': 'FG_PCT_Team'})

# Merge team FG% data
new_season_dataset_df = new_season_dataset_df.merge(
    fg_pct_team_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Rename columns for opponent FG%
fg_pct_opponent_df = fg_pct_df.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'FG_PCT': 'FG_PCT_Opponent'})

# Merge opponent FG% data
new_season_dataset_df = new_season_dataset_df.merge(
    fg_pct_opponent_df,
    on=['GAME_ID', 'OPPONENT_TEAM_ID'],
    how='left'
)

# Calculate cumulative season average FG% for Team
new_season_dataset_df['Season_Avg_FG_PCT_Team'] = (
    new_season_dataset_df.groupby(['TEAM_ID', 'SEASON'])['FG_PCT_Team']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Calculate cumulative season average FG% for Opponent
new_season_dataset_df['Season_Avg_FG_PCT_Opponent'] = (
    new_season_dataset_df.groupby(['OPPONENT_TEAM_ID', 'SEASON'])['FG_PCT_Opponent']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Drop the original FG% columns
new_season_dataset_df = new_season_dataset_df.drop(['FG_PCT_Team', 'FG_PCT_Opponent'], axis=1)

# Remove 'PTS_QTR1' and 'OPP_PTS_QTR1' before saving the final dataset, only if they exist
cols_to_drop = ['PTS_QTR1', 'OPP_PTS_QTR1']
existing_cols_to_drop = [col for col in cols_to_drop if col in new_season_dataset_df.columns]
if existing_cols_to_drop:
    new_season_dataset_df = new_season_dataset_df.drop(existing_cols_to_drop, axis=1)
    logging.info(f"Dropped columns: {existing_cols_to_drop}")
    print(f"Dropped columns: {existing_cols_to_drop}")
else:
    logging.info("No 'PTS_QTR1' or 'OPP_PTS_QTR1' columns to drop.")
    print("No 'PTS_QTR1' or 'OPP_PTS_QTR1' columns to drop.")

# Save the updated dataset
new_season_dataset_df.to_csv('new_season_dataset_with_season_avg_fg_pct.csv', index=False)
logging.info("Updated new season dataset with Season Average FG% saved to 'new_season_dataset_with_season_avg_fg_pct.csv'.")
print("Updated new season dataset with Season Average FG% saved to 'new_season_dataset_with_season_avg_fg_pct.csv'.")

# === Step 7: Fetch and Merge PTS and Offensive Rating ===

# Load caches
pts_cache = load_cache('pts_cache_new_season.pkl')
off_rating_cache = load_cache('off_rating_cache_new_season.pkl')

# Fetch and cache PTS and Off_Rating
unique_games_teams = new_season_dataset_df[['GAME_ID', 'TEAM_ID', 'OPPONENT_TEAM_ID']].drop_duplicates()
total_pairs = len(unique_games_teams)
print(f"Total unique (GAME_ID, TEAM_ID) pairs to process for PTS and Off_Rating: {total_pairs}")

for idx, row in unique_games_teams.iterrows():
    game_id = row['GAME_ID']
    team_id = row['TEAM_ID']
    opponent_id = row['OPPONENT_TEAM_ID']

    if pd.isnull(game_id) or pd.isnull(team_id) or pd.isnull(opponent_id):
        logging.warning(f"Skipping row {idx} due to missing GAME_ID, TEAM_ID, or OPPONENT_TEAM_ID.")
        continue

    # Fetch PTS for team
    if (game_id, team_id) not in pts_cache:
        pts_cache[(game_id, team_id)] = fetch_pts(game_id, team_id, pts_cache)

    # Fetch PTS for opponent
    if (game_id, opponent_id) not in pts_cache:
        pts_cache[(game_id, opponent_id)] = fetch_pts(game_id, opponent_id, pts_cache)

    # Fetch Off_Rating for team
    if (game_id, team_id) not in off_rating_cache:
        off_rating_cache[(game_id, team_id)] = fetch_off_rating(game_id, team_id, off_rating_cache)

    # Save caches periodically
    if (idx + 1) % 100 == 0:
        save_cache(pts_cache, 'pts_cache_new_season.pkl')
        save_cache(off_rating_cache, 'off_rating_cache_new_season.pkl')
        print(f"Processed {idx + 1}/{total_pairs} (GAME_ID, TEAM_ID) pairs")

# Save caches after processing all
save_cache(pts_cache, 'pts_cache_new_season.pkl')
save_cache(off_rating_cache, 'off_rating_cache_new_season.pkl')
print("PTS and Off_Rating data fetched and cached.")

# Convert caches to DataFrames
pts_data = [
    {'GAME_ID': game_id, 'TEAM_ID': team_id, 'PTS': pts}
    for (game_id, team_id), pts in pts_cache.items()
]

off_rating_data = [
    {'GAME_ID': game_id, 'TEAM_ID': team_id, 'Off_Rating': off_rating}
    for (game_id, team_id), off_rating in off_rating_cache.items()
]

pts_df = pd.DataFrame(pts_data)
off_rating_df = pd.DataFrame(off_rating_data)

# Merge PTS into new_season_dataset_df
new_season_dataset_df = new_season_dataset_df.merge(
    pts_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Merge Opponent PTS into new_season_dataset_df
opponent_pts_df = pts_df.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'PTS': 'OPP_PTS'})
new_season_dataset_df = new_season_dataset_df.merge(
    opponent_pts_df[['GAME_ID', 'OPPONENT_TEAM_ID', 'OPP_PTS']],
    on=['GAME_ID', 'OPPONENT_TEAM_ID'],
    how='left'
)

# Merge Off_Rating into new_season_dataset_df
new_season_dataset_df = new_season_dataset_df.merge(
    off_rating_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Handle missing values if any
for col in ['PTS', 'OPP_PTS', 'Off_Rating']:
    if col in new_season_dataset_df.columns:
        new_season_dataset_df[col] = pd.to_numeric(new_season_dataset_df[col], errors='coerce')
    else:
        logging.warning(f"Column '{col}' not found in new_season_dataset_df after merging.")
        new_season_dataset_df[col] = np.nan

# Step 8: Calculate PPG, Opponent PPG, and Offensive Rating

calculate_ppg_and_off_rating(new_season_dataset_df)

# The dataset now includes PPG, Opponent PPG, and Off_Rating_Avg

# Save the final dataset
new_season_dataset_df.to_csv('final_training_dataset_new_season.csv', index=False)
logging.info("Final training dataset for the most recent season saved to 'final_training_dataset_new_season.csv'.")
print("Final training dataset for the most recent season saved to 'final_training_dataset_new_season.csv'.")

test_pace = fetch_team_pace('2024-25')
print(test_pace.head())

print(new_season_dataset_df[['TEAM_ID', 'SEASON', 'Season_Avg_Pace_Team', 'Season_Avg_Pace_Opponent']].head())

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
from datetime import datetime

# Function to dynamically generate the list of seasons
def generate_seasons(start_year=2017):
    current_year = datetime.now().year
    if datetime.now().month >= 9:  # NBA seasons typically start in October
        end_year = current_year + 1
    else:
        end_year = current_year
    seasons = []
    for year in range(start_year, end_year):
        season = f"{year}-{str(year+1)[-2:]}"
        seasons.append(season)
    return seasons

# Use the function to get the list of seasons
training_seasons = generate_seasons()

training_games_df = pd.DataFrame()
test_games_df = pd.DataFrame()

# Function Definitions

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

            # Convert 'GAME_DATE' to datetime
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            current_date = datetime.now()
            # Filter out future games
            games = games[games['GAME_DATE'] <= current_date]

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
            print(f"No data available for game {game_id}")
            return None

        # Extract relevant columns
        line_score_df = line_score_df[['TEAM_ID', 'TEAM_ABBREVIATION', 'PTS_QTR1']]

        # Ensure two teams are present
        if len(line_score_df) != 2:
            print(f"Invalid number of teams in game {game_id}")
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
        print(f"Error fetching data for game {game_id}: {e}")
        return None

def save_data(data, filename):
    """Saves data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def load_data(filename):
    """Loads data from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"No existing data file found at {filename}")
        return []

def get_home_away(row):
    """Determines if the team was playing at home or away."""
    # Find the corresponding game entry in training_games_df
    game = training_games_df[
        (training_games_df['GAME_ID'] == row['GAME_ID']) &
        (training_games_df['TEAM_ID'] == row['TEAM_ID'])
    ]
    if not game.empty:
        matchup = game.iloc[0]['MATCHUP']
        if '@' in matchup:
            # The team is playing away
            return 'Away'
        else:
            # The team is playing at home
            return 'Home'
    else:
        return 'Unknown'

def get_recent_trend(team_id, game_date, num_games=5):
    """Fetches the recent trend of first-quarter points for a team."""
    # Filter games played by the team before the current game date
    team_games = training_dataset_df[
        (training_dataset_df['TEAM_ID'] == team_id) &
        (training_dataset_df['GAME_DATE'] < game_date)
    ].sort_values(by='GAME_DATE', ascending=False)
    # Get the last 'num_games' games
    recent_games = team_games.head(num_games)
    # Return the list of first-quarter points
    return recent_games['PTS_QTR1'].tolist()

def get_head_to_head_q1(team_id, opponent_id, game_date):
    """Calculates the average total first-quarter points in head-to-head games."""
    # Filter past games between the two teams before the current game date
    games_between = training_dataset_df[
        (
            (training_dataset_df['TEAM_ID'] == team_id) &
            (training_dataset_df['OPPONENT_TEAM_ID'] == opponent_id)
        ) &
        (training_dataset_df['GAME_DATE'] < game_date)
    ].sort_values(by='GAME_DATE', ascending=False)
    # Get the last 3 games
    recent_games = games_between.head(3)
    if not recent_games.empty:
        # Calculate the average total first-quarter points
        avg_total_q1_points = recent_games['Total_First_Quarter_Points'].mean()
        return avg_total_q1_points
    else:
        return None  # Or you can decide to return a default value

def save_starting_lineups(data, filename):
    """Saves starting lineups cache to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Starting lineups data saved to '{filename}'.")

def load_starting_lineups(filename):
    """Loads starting lineups cache from a pickle file."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Starting lineups data loaded from '{filename}'.")
            return data
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading starting lineups from '{filename}': {e}")
            return {}
    else:
        print(f"No existing starting lineups file found at '{filename}'. Starting a new cache.")
        return {}

def get_starting_lineup(game_id, team_id, starting_lineups_cache):
    """Fetches the starting lineup for a given GAME_ID and TEAM_ID."""
    key = (game_id, team_id)
    if key in starting_lineups_cache:
        # Starting lineup already fetched
        return starting_lineups_cache[key]
    else:
        # Fetch starting lineup from API
        try:
            time.sleep(0.6)  # Adjust based on rate limits
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            player_stats = boxscore.player_stats.get_data_frame()
            starters = player_stats[
                (player_stats['TEAM_ID'] == team_id) & (player_stats['START_POSITION'].notnull())
            ]
            lineup = starters['PLAYER_NAME'].tolist()
            # Save to cache
            starting_lineups_cache[key] = lineup
            print(f"Fetched starting lineup for GAME_ID {game_id}, TEAM_ID {team_id}: {lineup}")
            return lineup
        except Exception as e:
            error_msg = f"Error fetching starting lineup for game {game_id}, team {team_id}: {e}"
            print(error_msg)
            # Save empty list to avoid retrying
            starting_lineups_cache[key] = []
            return []

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
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    pace_data = team_stats[['TEAM_ID', 'PACE']]
    return pace_data

def fetch_and_save_pace_data(seasons, filename='pace_data.pkl'):
    """Fetches and saves pace data for multiple seasons."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            pace_data = pickle.load(f)
        print(f"Pace data loaded from '{filename}'.")
    else:
        pace_dfs = []
        for season in seasons:
            print(f"Fetching pace data for season {season}")
            pace_df = fetch_team_pace(season)
            pace_df['SEASON'] = season
            pace_dfs.append(pace_df)
            time.sleep(1)  # Respect API rate limits
        pace_data = pd.concat(pace_dfs, ignore_index=True)
        with open(filename, 'wb') as f:
            pickle.dump(pace_data, f)
        print(f"Pace data saved to '{filename}'.")
    return pace_data

def fetch_fg_pct(game_id, team_id, cache):
    """
    Fetches the FG_PCT for a given game and team. Utilizes a cache to prevent redundant API calls.
    """
    key = (game_id, team_id)
    if key in cache:
        # print(f"FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id} fetched from cache.")
        return cache[key]

    try:
        # Ensure game_id is formatted as a 10-digit string
        game_id_str = str(game_id).zfill(10)

        # Respect API rate limits
        time.sleep(0.6)

        # Fetch data from BoxScoreTraditionalV2 for the entire game
        boxscore = BoxScoreTraditionalV2(
            game_id=game_id_str,
            start_period=0, 
            end_period=10,   # Fetch all periods in the game
            range_type=0,
            start_range=0,
            end_range=0
        )

        # Access the team stats dataset
        team_stats = boxscore.team_stats.get_data_frame()

        # Filter by team_id to find the specific team's stats
        team_row = team_stats[team_stats['TEAM_ID'] == team_id]

        if not team_row.empty:
            fg_pct = team_row['FG_PCT'].values[0]
            cache[key] = fg_pct  # Cache the result for future calls
            # print(f"FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id} fetched from API: {fg_pct}")
            return fg_pct
        else:
            print(f"No data found for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan

    except Exception as e:
        print(f"Error fetching FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id}: {e}")
        cache[key] = np.nan
        return np.nan

def save_cache(cache, filename):
    """Saves a cache dictionary to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to '{filename}'.")

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

# === Functions to Fetch and Cache PTS and Offensive Rating ===

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
            # print(f"PTS for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {pts}")
            return pts
        else:
            print(f"No PTS data for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan
    except Exception as e:
        print(f"Error fetching PTS for GAME_ID {game_id_str}, TEAM_ID {team_id}: {e}")
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
            # print(f"Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {off_rating}")
            return off_rating
        else:
            print(f"No Off_Rating data for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            cache[key] = np.nan
            return np.nan
    except Exception as e:
        print(f"Error fetching Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id}: {e}")
        cache[key] = np.nan
        return np.nan

# === Step 8: Calculate PPG and Offensive Rating ===

def calculate_ppg_and_off_rating(training_dataset_df):
    """
    Calculates Points Per Game (PPG), Opponent PPG, and Offensive Rating (Off_Rating_Avg) for each team up to the previous game.
    """
    # Ensure 'GAME_DATE' is in datetime format
    training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

    # Sort the DataFrame
    training_dataset_df.sort_values(by=['TEAM_ID', 'SEASON', 'GAME_DATE'], inplace=True)

    # Calculate cumulative points excluding the current game
    training_dataset_df['Cumulative_Points_Previous'] = training_dataset_df.groupby(['TEAM_ID', 'SEASON'])['PTS'].cumsum().shift(1)
    training_dataset_df['Cumulative_Opp_Points_Previous'] = training_dataset_df.groupby(['TEAM_ID', 'SEASON'])['OPP_PTS'].cumsum().shift(1)

    # Calculate cumulative Offensive Rating excluding the current game
    training_dataset_df['Cumulative_Off_Rating_Previous'] = training_dataset_df.groupby(['TEAM_ID', 'SEASON'])['Off_Rating'].cumsum().shift(1)

    # Calculate Game Number within each TEAM_ID and SEASON
    training_dataset_df['Game_Number'] = training_dataset_df.groupby(['TEAM_ID', 'SEASON']).cumcount() + 1

    # Calculate PPG and Opponent PPG up to the previous game
    training_dataset_df['PPG_Team'] = training_dataset_df['Cumulative_Points_Previous'] / (training_dataset_df['Game_Number'] - 1)
    training_dataset_df['PPG_Opponent'] = training_dataset_df['Cumulative_Opp_Points_Previous'] / (training_dataset_df['Game_Number'] - 1)

    # Calculate Offensive Rating up to the previous game
    training_dataset_df['Off_Rating_Avg'] = training_dataset_df['Cumulative_Off_Rating_Previous'] / (training_dataset_df['Game_Number'] - 1)

    # For the first game, use the points and Offensive Rating from that game
    first_games_mask = training_dataset_df['Game_Number'] == 1
    training_dataset_df.loc[first_games_mask, 'PPG_Team'] = training_dataset_df.loc[first_games_mask, 'PTS']
    training_dataset_df.loc[first_games_mask, 'PPG_Opponent'] = training_dataset_df.loc[first_games_mask, 'OPP_PTS']
    training_dataset_df.loc[first_games_mask, 'Off_Rating_Avg'] = training_dataset_df.loc[first_games_mask, 'Off_Rating']

    print("\nCalculated PPG, Opponent PPG, and Offensive Rating up to the previous game.")

    # Drop intermediate columns
    training_dataset_df.drop(['Cumulative_Points_Previous', 'Cumulative_Opp_Points_Previous', 'Cumulative_Off_Rating_Previous', 'Game_Number'], axis=1, inplace=True)

    # Save the updated dataset
    training_dataset_df.to_csv('training_dataset_with_ppg_off_rating.csv', index=False)
    print("Updated training dataset with PPG and Offensive Rating saved to 'training_dataset_with_ppg_off_rating.csv'")

# === End of Step 8 ===

# Start of the main script

# Step 1: Fetch training games
training_games_df = fetch_game_ids(training_seasons)

# Extract unique GAME_IDs
training_game_ids = training_games_df['GAME_ID'].unique()

# Define filename for training data
data_filename = 'all_training_data.pkl'

# Load existing training data if available
all_training_data = load_data(data_filename)
processed_game_ids = set()

# If data was loaded, extract already processed GAME_IDs
if all_training_data:
    processed_game_ids = set([data['GAME_ID'] for data in all_training_data])
    print(f"Resuming from existing data. {len(processed_game_ids)} games already processed.")
else:
    all_training_data = []
    print("Starting data collection from scratch.")

# Step 2: Fetch and populate training data
for idx, game_id in enumerate(training_game_ids):
    if game_id in processed_game_ids:
        continue  # Skip already processed games

    game_data = fetch_game_data(game_id)
    if game_data:
        all_training_data.extend(game_data.values())
        processed_game_ids.add(game_id)
    else:
        print(f"No data for game {game_id}")

    # Save data after every 100 games
    if (idx + 1) % 100 == 0 or (idx + 1) == len(training_game_ids):
        save_data(all_training_data, data_filename)
        print(f"Processed {idx + 1}/{len(training_game_ids)} training games")

    time.sleep(0.6)  # Respect rate limits

# Save the final training data
save_data(all_training_data, data_filename)
print("All training data has been collected and saved.")

# Load data if not already in memory
if 'all_training_data' not in globals():
    all_training_data = load_data(data_filename)

# Create DataFrame from training data
training_dataset_df = pd.DataFrame(all_training_data)

# Calculate total first-quarter points
training_dataset_df['Total_First_Quarter_Points'] = (
    training_dataset_df['PTS_QTR1'] + training_dataset_df['OPP_PTS_QTR1']
)

# Extract 'GAME_ID' and 'GAME_DATE' from training_games_df
game_dates = training_games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID']].drop_duplicates()

# Merge 'GAME_DATE' and 'MATCHUP' into training_dataset_df
training_dataset_df = training_dataset_df.merge(game_dates, on=['GAME_ID', 'TEAM_ID'], how='left')

# Convert 'GAME_DATE' to datetime format
training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

# Extract 'MONTH' from 'GAME_DATE'
training_dataset_df['MONTH'] = training_dataset_df['GAME_DATE'].dt.month

# Check for a pre-existing checkpoint
checkpoint_filename = 'training_dataset_with_trends.csv'
if os.path.exists(checkpoint_filename):
    training_dataset_df = pd.read_csv(checkpoint_filename)
    print("Loaded DataFrame from checkpoint file.")
else:
    # Calculations for 'Home_Away', 'Recent_Trend_Team', 'Recent_Trend_Opponent', 'Head_to_Head_Q1'
    training_dataset_df['Home_Away'] = training_dataset_df.apply(get_home_away, axis=1)
    training_dataset_df['Recent_Trend_Team'] = training_dataset_df.apply(
        lambda row: get_recent_trend(row['TEAM_ID'], row['GAME_DATE']), axis=1
    )
    training_dataset_df['Recent_Trend_Opponent'] = training_dataset_df.apply(
        lambda row: get_recent_trend(row['OPPONENT_TEAM_ID'], row['GAME_DATE']), axis=1
    )
    training_dataset_df['Head_to_Head_Q1'] = training_dataset_df.apply(
        lambda row: get_head_to_head_q1(row['TEAM_ID'], row['OPPONENT_TEAM_ID'], row['GAME_DATE']), axis=1
    )

    # Save the checkpoint
    training_dataset_df.to_csv(checkpoint_filename, index=False)
    print(f"Checkpoint saved to '{checkpoint_filename}'.")

# Step 5: Additional Processing (Season, Pace, etc.)

# Ensure 'GAME_DATE' is in datetime format
training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'], errors='coerce')

# Apply the function to create the 'SEASON' column
training_dataset_df['SEASON'] = training_dataset_df.apply(determine_season, axis=1)

print(training_dataset_df[['GAME_DATE', 'SEASON']].head())

# Get unique seasons from the 'SEASON' column
seasons = training_dataset_df['SEASON'].unique()
print(f"Seasons to fetch pace data for: {seasons}")

# Fetch and save pace data for these seasons
pace_data = fetch_and_save_pace_data(seasons)

# Merge team pace data
training_dataset_df = training_dataset_df.merge(
    pace_data.rename(columns={'TEAM_ID': 'TEAM_ID', 'PACE': 'PACE_Team'}),
    on=['TEAM_ID', 'SEASON'],
    how='left'
)

# Merge opponent pace data
training_dataset_df = training_dataset_df.merge(
    pace_data.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'PACE': 'PACE_Opponent'}),
    on=['OPPONENT_TEAM_ID', 'SEASON'],
    how='left'
)

# Ensure 'GAME_DATE' is in datetime format
training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

# Sort the DataFrame
training_dataset_df = training_dataset_df.sort_values(by=['TEAM_ID', 'SEASON', 'GAME_DATE']).reset_index(drop=True)

# Calculate cumulative season average Pace for Team
training_dataset_df['Season_Avg_Pace_Team'] = (
    training_dataset_df.groupby(['TEAM_ID', 'SEASON'])['PACE_Team']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Calculate cumulative season average Pace for Opponent
training_dataset_df['Season_Avg_Pace_Opponent'] = (
    training_dataset_df.groupby(['OPPONENT_TEAM_ID', 'SEASON'])['PACE_Opponent']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Calculate the average of season averages (optional)
training_dataset_df['Average_Season_Avg_Pace'] = (
    training_dataset_df['Season_Avg_Pace_Team'] + training_dataset_df['Season_Avg_Pace_Opponent']
) / 2

# Drop the original Pace columns
training_dataset_df = training_dataset_df.drop(['PACE_Team', 'PACE_Opponent'], axis=1)

# Verify the new columns (optional)
print(training_dataset_df[['TEAM_ID', 'GAME_DATE', 'Season_Avg_Pace_Team', 'Season_Avg_Pace_Opponent', 'Average_Season_Avg_Pace']].head(10))

# Save the dataset up to this point
training_dataset_df.to_csv('training_dataset_up_to_pace.csv', index=False)
print("Dataset up to pace data saved to 'training_dataset_up_to_pace.csv'")

# === Step 6: Process FG_PCT Data ===

# Load the cache or initialize a new one if it doesn't exist
fg_pct_cache = load_cache('fg_pct_cache.pkl')

# Define batch size
batch_size = 100  # Adjust based on API rate limits and performance needs
total_games = len(training_dataset_df)
batches = (total_games // batch_size) + (1 if total_games % batch_size > 0 else 0)

# Process in batches with a progress bar
print("Processing FG_PCT data in batches...")
for batch_num in tqdm(range(batches), desc="FG_PCT Batches"):
    # Define start and end of the batch
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_games)

    # Process each row in the current batch
    for idx in range(start_idx, end_idx):
        row = training_dataset_df.iloc[idx]
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        if (game_id, team_id) not in fg_pct_cache:
            # Fetch FG_PCT if it's not already cached
            fg_pct_cache[(game_id, team_id)] = fetch_fg_pct(game_id, team_id, fg_pct_cache)

    # Save cache at the end of each batch to prevent data loss
    save_cache(fg_pct_cache, 'fg_pct_cache.pkl')

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
training_dataset_df = training_dataset_df.merge(
    fg_pct_team_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Rename columns for opponent FG%
fg_pct_opponent_df = fg_pct_df.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'FG_PCT': 'FG_PCT_Opponent'})

# Merge opponent FG% data
training_dataset_df = training_dataset_df.merge(
    fg_pct_opponent_df,
    on=['GAME_ID', 'OPPONENT_TEAM_ID'],
    how='left'
)

# Calculate cumulative season average FG% for Team
training_dataset_df['Season_Avg_FG_PCT_Team'] = (
    training_dataset_df
    .sort_values(['TEAM_ID', 'SEASON', 'GAME_DATE'])
    .groupby(['TEAM_ID', 'SEASON'])['FG_PCT_Team']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Calculate cumulative season average FG% for Opponent
training_dataset_df['Season_Avg_FG_PCT_Opponent'] = (
    training_dataset_df
    .sort_values(['OPPONENT_TEAM_ID', 'SEASON', 'GAME_DATE'])
    .groupby(['OPPONENT_TEAM_ID', 'SEASON'])['FG_PCT_Opponent']
    .expanding()
    .mean()
    .reset_index(level=[0,1], drop=True)
)

# Drop the original FG% columns
training_dataset_df = training_dataset_df.drop(['FG_PCT_Team', 'FG_PCT_Opponent'], axis=1)

# Remove 'PTS_QTR1' and 'OPP_PTS_QTR1' before saving the final dataset
training_dataset_df = training_dataset_df.drop(['PTS_QTR1', 'OPP_PTS_QTR1'], axis=1)

# Save the updated dataset
training_dataset_df.to_csv('training_dataset_with_season_avg_fg_pct.csv', index=False)
print("Updated training dataset with Season Average FG% saved to 'training_dataset_with_season_avg_fg_pct.csv'")

# Continue with Step 7: Fetch and Merge PTS and Offensive Rating

# Ensure 'GAME_DATE' is in datetime format
training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

# Load caches
pts_cache = load_cache('pts_cache.pkl')
off_rating_cache = load_cache('off_rating_cache.pkl')

# Fetch and cache PTS and Off_Rating
unique_games_teams = training_dataset_df[['GAME_ID', 'TEAM_ID', 'OPPONENT_TEAM_ID']].drop_duplicates()
total_pairs = len(unique_games_teams)
print(f"Total unique (GAME_ID, TEAM_ID) pairs to process for PTS and Off_Rating: {total_pairs}")

for idx, row in unique_games_teams.iterrows():
    game_id = row['GAME_ID']
    team_id = row['TEAM_ID']
    opponent_id = row['OPPONENT_TEAM_ID']

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
        save_cache(pts_cache, 'pts_cache.pkl')
        save_cache(off_rating_cache, 'off_rating_cache.pkl')
        print(f"Processed {idx + 1}/{total_pairs} (GAME_ID, TEAM_ID) pairs")

# Save caches after processing all
save_cache(pts_cache, 'pts_cache.pkl')
save_cache(off_rating_cache, 'off_rating_cache.pkl')
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

# Merge PTS into training_dataset_df
training_dataset_df = training_dataset_df.merge(
    pts_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Merge Opponent PTS into training_dataset_df
opponent_pts_df = pts_df.rename(columns={'TEAM_ID': 'OPPONENT_TEAM_ID', 'PTS': 'OPP_PTS'})
training_dataset_df = training_dataset_df.merge(
    opponent_pts_df[['GAME_ID', 'OPPONENT_TEAM_ID', 'OPP_PTS']],
    on=['GAME_ID', 'OPPONENT_TEAM_ID'],
    how='left'
)

# Merge Off_Rating into training_dataset_df
training_dataset_df = training_dataset_df.merge(
    off_rating_df,
    on=['GAME_ID', 'TEAM_ID'],
    how='left'
)

# Handle missing values if any
training_dataset_df['PTS'] = pd.to_numeric(training_dataset_df['PTS'], errors='coerce')
training_dataset_df['OPP_PTS'] = pd.to_numeric(training_dataset_df['OPP_PTS'], errors='coerce')
training_dataset_df['Off_Rating'] = pd.to_numeric(training_dataset_df['Off_Rating'], errors='coerce')

# Step 8: Calculate PPG, Opponent PPG, and Offensive Rating

calculate_ppg_and_off_rating(training_dataset_df)

# The dataset now includes PPG, Opponent PPG, and Off_Rating_Avg

# Save the final dataset
training_dataset_df.to_csv('final_training_dataset.csv', index=False)
print("Final training dataset saved to 'final_training_dataset.csv'")

import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoresummaryv2,
    boxscoretraditionalv2,
)
from nba_api.stats.endpoints import leaguedashteamstats
import pickle
import os
from nba_api.stats.endpoints import BoxScoreTraditionalV2
from tqdm import tqdm

# Define the seasons in the correct format
training_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
test_season = ['2023-24']  # If data is available for this season

training_games_df = pd.DataFrame()
test_games_df = pd.DataFrame()

# Adjust the fetch_game_ids function
def fetch_game_ids(seasons):
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
            print(f"Error fetching games for season {season}: {e}")
    if all_games:
        return pd.concat(all_games, ignore_index=True)
    else:
        print("No games fetched for any season.")
        return pd.DataFrame()

def fetch_game_data(game_id):
    game_data = {}
    try:
        # Fetch Box Score Summary
        boxscore = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        line_score_df = boxscore.line_score.get_data_frame()

        if line_score_df.empty:
            print(f"No line score data for game {game_id}")
            return None

        # Extract team and opponent data
        teams_info = line_score_df[['TEAM_ID', 'TEAM_ABBREVIATION', 'PTS_QTR1']]

        # Ensure two teams are present
        if len(teams_info) != 2:
            print(f"Invalid number of teams in game {game_id}")
            return None

        # Assign team and opponent data
        team1 = teams_info.iloc[0]
        team2 = teams_info.iloc[1]

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
                # Additional fields will be added later
            }
            key = f"{game_id}_{team['TEAM_ID']}"
            game_data[key] = data
        return game_data
    except Exception as e:
        print(f"Error fetching data for game {game_id}: {e}")
        return None

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def load_data(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"No existing data file found at {filename}")
        return []

def get_home_away(row):
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
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Starting lineups data saved to {filename}")

def load_starting_lineups(filename):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Starting lineups data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"No existing starting lineups file found at {filename}")
        return {}

def get_starting_lineup(game_id, team_id, starting_lineups_cache):
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
            return lineup
        except Exception as e:
            error_msg = f"Error fetching starting lineup for game {game_id}, team {team_id}: {e}"
            print(error_msg)
            # Save empty list to avoid retrying
            starting_lineups_cache[key] = []
            return []

# Define a function to determine the season based on 'GAME_DATE'
def determine_season(row):
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
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]
    pace_data = team_stats[['TEAM_ID', 'PACE']]
    return pace_data

def fetch_and_save_pace_data(seasons, filename='pace_data.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            pace_data = pickle.load(f)
        print(f"Pace data loaded from {filename}")
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
        print(f"Pace data saved to {filename}")
    return pace_data

def fetch_fg_pct(game_id, team_id, cache):
    """
    Fetches the FG_PCT for a given game and team. Utilizes a cache to prevent redundant API calls.
    
    Parameters:
        game_id (str): The unique 10-digit identifier for the game.
        team_id (int): The unique identifier for the team.
        cache (dict): Cache dictionary to store/retrieve FG_PCT.
    
    Returns:
        float: The FG_PCT value or NaN if not available.
    """
    key = (game_id, team_id)
    if key in cache:
        print(f"FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id} fetched from cache.")
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
            print(f"FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id} fetched from API: {fg_pct}")
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
    with open(filename, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved to '{filename}'.")

def load_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
        print(f"Cache loaded from '{filename}'.")
        return cache
    else:
        print(f"No existing cache found at '{filename}'. Starting a new cache.")
        return {}




# Start of the main script

# Fetch training games
training_games_df = fetch_game_ids(training_seasons)

training_game_ids = training_games_df['GAME_ID'].unique()

data_filename = 'all_training_data.pkl'

all_training_data = load_data(data_filename)
processed_game_ids = set()

# If data was loaded, extract already processed GAME_IDs
if all_training_data:
    processed_game_ids = set([data['GAME_ID'] for data in all_training_data])
    print(f"Resuming from existing data. {len(processed_game_ids)} games already processed.")
else:
    all_training_data = []
    print("Starting data collection from scratch.")

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

    time.sleep(0.6)

save_data(all_training_data, data_filename)
print("All training data has been collected and saved.")

# Load data if not already in memory
if 'all_training_data' not in globals():
    all_training_data = load_data(data_filename)

training_dataset_df = pd.DataFrame(all_training_data)

# Calculate total first-quarter points
training_dataset_df['Total_First_Quarter_Points'] = (
    training_dataset_df['PTS_QTR1'] + training_dataset_df['OPP_PTS_QTR1']
)

# Extract 'GAME_ID' and 'GAME_DATE' from training_games_df
game_dates = training_games_df[['GAME_ID', 'GAME_DATE']].drop_duplicates()

# Merge 'GAME_DATE' into training_dataset_df
training_dataset_df = training_dataset_df.merge(game_dates, on='GAME_ID', how='left')

# Convert 'GAME_DATE' to datetime format
training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

# Extract 'MONTH' from 'GAME_DATE'
training_dataset_df['MONTH'] = training_dataset_df['GAME_DATE'].dt.month

# Check if 'MATCHUP' column exists
if 'MATCHUP' not in training_games_df.columns:
    print("'MATCHUP' column is missing in training_games_df.")
else:
    print("'MATCHUP' column is present.")


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





starting_lineups_filename = 'starting_lineups.pkl'

starting_lineups_cache = load_starting_lineups(starting_lineups_filename)

# **Optimized Starting Lineups Processing**

# Convert starting_lineups_cache to DataFrames
print("Merging starting lineups from cache into training_dataset_df.")

# Team starting lineups
lineups_data = []
for (game_id, team_id), lineup in starting_lineups_cache.items():
    lineups_data.append({
        'GAME_ID': game_id,
        'TEAM_ID': team_id,
        'Starting_Lineup_Team': lineup
    })

starting_lineups_df = pd.DataFrame(lineups_data)

# Opponent starting lineups
opponent_lineups_data = []
for (game_id, team_id), lineup in starting_lineups_cache.items():
    opponent_lineups_data.append({
        'GAME_ID': game_id,
        'OPPONENT_TEAM_ID': team_id,
        'Starting_Lineup_Opponent': lineup
    })

opponent_starting_lineups_df = pd.DataFrame(opponent_lineups_data)

# **Ensure consistent data types for 'GAME_ID' in both DataFrames**
training_dataset_df['GAME_ID'] = training_dataset_df['GAME_ID'].astype(str)
starting_lineups_df['GAME_ID'] = starting_lineups_df['GAME_ID'].astype(str)
opponent_starting_lineups_df['GAME_ID'] = opponent_starting_lineups_df['GAME_ID'].astype(str)



training_dataset_df['Head_to_Head_Q1'] = training_dataset_df.apply(
    lambda row: row['Recent_Trend_Team'][0] if pd.isnull(row['Head_to_Head_Q1']) and row['Recent_Trend_Team'] else row['Head_to_Head_Q1'],
    axis=1
)

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

# Calculate average pace between teams
training_dataset_df['Average_Pace'] = (
    training_dataset_df['PACE_Team'] + training_dataset_df['PACE_Opponent']
) / 2

# Verify the 'Average_Pace' column
print(training_dataset_df[['TEAM_ID', 'OPPONENT_TEAM_ID', 'PACE_Team', 'PACE_Opponent', 'Average_Pace']].head())

# Save the final dataset up to this point
training_dataset_df.to_csv('training_dataset_up_to_pace.csv', index=False)
print("Final dataset up to pace data saved to 'training_dataset_up_to_pace.csv'")

############################ 


# Load the cache or initialize a new one if it doesn't exist
fg_pct_cache = load_cache('fg_pct_cache2.pkl')

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

# Merge FG_PCT data with the main dataset
training_dataset_df = training_dataset_df.merge(fg_pct_df, on=['GAME_ID', 'TEAM_ID'], how='left')

# Save the updated dataset
training_dataset_df.to_csv('training_dataset_with_fg_pct.csv', index=False)
print("Updated training dataset with FG_PCT saved to 'training_dataset_with_fg_pct.csv'")






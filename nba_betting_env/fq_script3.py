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
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2, LeagueGameFinder
from tqdm import tqdm
import json

# Define the seasons in the correct format
training_seasons = ['2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']

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
            print(f"PTS for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {pts}")
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
            print(f"Off_Rating for GAME_ID {game_id_str}, TEAM_ID {team_id} fetched: {off_rating}")
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

def add_blank_game_row(csv_filename):
    """
    Adds a blank row to the CSV dataset for a new game.
    
    Parameters:
        csv_filename (str): The path to the CSV file.
    """
    # Load the existing dataset
    df = pd.read_csv(csv_filename)
    
    # Define a blank row with NaN values
    blank_row = {column: np.nan for column in df.columns}
    
    # Convert the blank_row dictionary to a DataFrame
    blank_df = pd.DataFrame([blank_row])
    
    # Concatenate the blank row to the existing DataFrame
    df = pd.concat([df, blank_df], ignore_index=True)
    
    # Save the updated DataFrame back to CSV
    df.to_csv(csv_filename, index=False)
    print("Added a blank row for a new game.")


# === End of Step 8 ===

import sys      # For command-line arguments

if len(sys.argv) > 1:
    if sys.argv[1] == "build":
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

        df = training_dataset_df
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Ensure 'GAME_ID' is treated as a string for consistent sorting
        df['GAME_ID'] = df['GAME_ID'].astype(str)

        # Sort the DataFrame by 'GAME_ID'
        df.sort_values(by='GAME_ID', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Copy the 'Off_Rating_Avg' column
        off_rating_avg_copy = df[['Off_Rating_Avg']].copy()

        import numpy as np

        # Number of rows in the DataFrame
        num_rows = len(off_rating_avg_copy)

        # Create an array of indices
        indices = np.arange(num_rows)

        # Swap indices for every pair
        for i in range(0, num_rows - 1, 2):
            indices[i], indices[i + 1] = indices[i + 1], indices[i]

        # Reorder the 'Off_Rating_Avg' values according to the swapped indices
        off_rating_avg_swapped = off_rating_avg_copy.iloc[indices].reset_index(drop=True)

        # Add the swapped 'Off_Rating_Avg' as 'Opp_Off_Rating_Avg' to the main DataFrame
        df['Opp_Off_Rating_Avg'] = off_rating_avg_swapped['Off_Rating_Avg']

        # Display the first few rows to verify
        print(df[['GAME_ID', 'TEAM_ID', 'OPPONENT_TEAM_ID', 'Off_Rating_Avg', 'Opp_Off_Rating_Avg']].head(10))

        training_games_df = df

        # Save the final dataset
        training_dataset_df.to_csv('final_training_dataset.csv', index=False)
        print("Final training dataset saved to 'final_training_dataset.csv'")

        

else:
    count = 0
    while(count < 65):
        print("Fetching new games")

        # Define the path to your CSV file
        csv_filename = 'final_training_dataset.csv'

        # Read the CSV into a DataFrame
        try:
            df = pd.read_csv(csv_filename, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE'])
            print(f"Successfully loaded '{csv_filename}'.")
        except FileNotFoundError:
            print(f"Error: The file '{csv_filename}' does not exist in the current directory.")
            exit()
        except Exception as e:
            print(f"An error occurred while reading '{csv_filename}': {e}")
            exit()

        def latest_game(df):
            if not df.empty:
                # Extract the last GAME_ID correctly
                last_game_id = df['GAME_ID'].iloc[-1].strip()
                print(f"Extracted GAME_ID: {last_game_id}")
                return last_game_id
            else:
                print("The DataFrame is empty. No rows to display.")

        new_game_id = str(int(latest_game(df)) + 1).zfill(10)

        print(f"New game ID: {new_game_id}")  

        def get_recent_trend(csv_filename, team_id, game_date):
            """
            Retrieves the recent trend array for a team based on the most recent game played before the specified game date.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team_id (str or int): The TEAM_ID of the team.
                game_date (str or pd.Timestamp): The date of the current game. The function fetches trends from games before this date.
            
            Returns:
                list: The Recent_Trend_Team array from the most recent game before the specified game date.
                    Returns an empty list if no such game exists.
            """
            try:
                # Read the CSV file with GAME_DATE parsed as datetime
                df = pd.read_csv(csv_filename, dtype={'GAME_ID': str, 'TEAM_ID': str}, parse_dates=['GAME_DATE'])
                
                # Filter games for the specified team and before the given game_date
                filtered_games = df[
                    (df['TEAM_ID'] == str(team_id)) &
                    (df['GAME_DATE'] < pd.to_datetime(game_date))
                ]
                
                if filtered_games.empty:
                    print(f"No prior games found for TEAM_ID {team_id} before {game_date}.")
                    return []
                
                # Sort the filtered games by GAME_DATE descending to get the most recent game first
                sorted_games = filtered_games.sort_values(by='GAME_DATE', ascending=False)
                
                # Select the most recent game
                last_game = sorted_games.iloc[0]

                # Extract the GAME_ID of the last_game and save it to variable game_id
                game_id = last_game['GAME_ID']
                
                # Retrieve the Recent_Trend_Team column
                trend_json = last_game.get('Recent_Trend_Team', '[]')  # Default to empty list if key not found
                
                # Handle cases where the trend is NaN or empty
                if pd.isna(trend_json) or trend_json.strip() == '':
                    trend = []
                else:
                    try:
                        trend = json.loads(trend_json)
                        # Ensure all elements are integers (in case they are read as floats)
                        trend = [int(x) for x in trend]
                    except json.JSONDecodeError:
                        print(f"Invalid JSON format for Recent_Trend_Team in GAME_ID {last_game['GAME_ID']}.")
                        trend = []
                    except Exception as e:
                        print(f"Error parsing Recent_Trend_Team: {e}")
                        trend = []
            
                trend.pop(len(trend)-1)
                return trend
            
            except FileNotFoundError:
                print(f"CSV file '{csv_filename}' not found.")
                return []
            except pd.errors.EmptyDataError:
                print(f"CSV file '{csv_filename}' is empty.")
                return []
            except Exception as e:
                print(f"An error occurred while fetching recent trend: {e}")
                return []

        def fetch_pts_qtr1_game_id(csv_filename, team_id, game_date):
            # Read the CSV file with GAME_DATE parsed as datetime
            df = pd.read_csv(csv_filename, dtype={'GAME_ID': str, 'TEAM_ID': str}, parse_dates=['GAME_DATE'])
            
            # Filter games for the specified team and before the given game_date
            filtered_games = df[
                (df['TEAM_ID'] == str(team_id)) &
                (df['GAME_DATE'] < pd.to_datetime(game_date))
            ]

            if filtered_games.empty:
                print(f"No prior games found for TEAM_ID {team_id} before {game_date}.")
                return []
            
            # Sort the filtered games by GAME_DATE descending to get the most recent game first
            sorted_games = filtered_games.sort_values(by='GAME_DATE', ascending=False)
            
            # Select the most recent game
            last_game = sorted_games.iloc[0]

            # Extract the GAME_ID of the last_game and save it to variable game_id
            game_id = last_game['GAME_ID']

            return str(game_id).zfill(10)

        def fetch_pts_qtr1(game_id, team_id):
            # Fetch box score summary
            boxscore = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
            time.sleep(0.2)  # To respect API rate limits

            line_score = boxscore.line_score.get_data_frame()

            if line_score.empty or len(line_score) < 2:
                print(f"Insufficient data for GAME_ID {game_id}.")
                return
            
            if str(line_score.iloc[0]['TEAM_ID']) == str(team_id):
                team = line_score.iloc[0] 
            else:
                team = line_score.iloc[1] 

            return int(team['PTS_QTR1'])

        def num_of_games(csv_filename, team_id):
            # Read the CSV file with GAME_DATE parsed as datetime
            df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str, 'SEASON': str}, parse_dates=['GAME_DATE'])
            
            # Filter games for the specified team and before the given game_date
            filtered_games = df[
                (df['TEAM_ID'] == str(team_id)) &
                (df['SEASON'] == "2024-25") 
            ]

            return len(filtered_games)

        def fetch_latest_off(csv_filename, team_id, game_id):
            """
            Fetches and returns the pace value for the latest game played by the specified team
            using the BoxScoreAdvancedV2 endpoint.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team_id (str or int): The TEAM_ID of the team.
            
            Returns:
                float or None: The pace value from the latest game, or None if not found.
            """
            try:
                # Read the CSV file to find the latest game played by the team
                df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
                team_id = str(team_id)
                team_games = df[df['TEAM_ID'] == team_id]
                
                if team_games.empty:
                    print(f"No games found for TEAM_ID {team_id}.")
                    return None

                game_id = str(game_id).zfill(10)

                # Fetch advanced box score data for the latest game
                boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                time.sleep(0.2)  # Respect API rate limits
                
                # Get team stats DataFrame
                team_stats = boxscore.team_stats.get_data_frame()
                if str(team_stats.iloc[0]['TEAM_ID']) != team_id:
                    return float(team_stats.iloc[1]['OFF_RATING'])
                
                return float(team_stats.iloc[0]['OFF_RATING'])
            
            except Exception as e:
                print(f"An error occurred off rating: {e}")
                return None

        def fetch_latest_pace(csv_filename, team_id):
            """
            Fetches and returns the pace value for the latest game played by the specified team
            using the BoxScoreAdvancedV2 endpoint.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team_id (str or int): The TEAM_ID of the team.
            
            Returns:
                float or None: The pace value from the latest game, or None if not found.
            """
            try:
                # Read the CSV file to find the latest game played by the team
                df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
                team_id = str(team_id)
                team_games = df[df['TEAM_ID'] == team_id]
                
                if team_games.empty:
                    print(f"No games found for TEAM_ID {team_id}.")
                    return None
                

                # Get the latest game based on GAME_DATE
                latest_game = team_games.sort_values(by='GAME_DATE', ascending=False).iloc[0]
                game_id = str(latest_game['GAME_ID']).zfill(10)

                # Fetch advanced box score data for the latest game
                boxscore = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
                time.sleep(0.2)  # Respect API rate limits
                
                # Get team stats DataFrame
                team_stats = boxscore.team_stats.get_data_frame()
                if str(team_stats.iloc[0]['TEAM_ID']) != team_id:
                    return float(team_stats.iloc[1]['PACE'])
                
                return float(team_stats.iloc[1]['PACE'])
            
            except Exception as e:
                print(f"An error occurred pace: {e}")
                return None

        def fetch_latest_pts(csv_filename, team_id, game_id):
            """
            Fetches and returns the pace value for the latest game played by the specified team
            using the BoxScoreAdvancedV2 endpoint.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team_id (str or int): The TEAM_ID of the team.
            
            Returns:
                float or None: The pace value from the latest game, or None if not found.
            """
            try:
                if(len(game_id) < 10):
                    game_id.zfill(10)

                # Get the latest game based on GAME_DATE
                latest_game = game_id

                time.sleep(0.2)  # Respect API rate limits

                # Fetch data from BoxScoreTraditionalV2 for the entire game
                boxscore = BoxScoreTraditionalV2(
                    game_id=latest_game,
                    start_period=0, 
                    end_period=10,   # Fetch all periods in the game
                    range_type=0,
                    start_range=0,
                    end_range=0
                )

                team_id = str(team_id)

                # Access the team stats dataset
                team_stats = boxscore.team_stats.get_data_frame()

                if str(team_stats.iloc[0]['TEAM_ID']) != team_id: 
                    return float(team_stats.iloc[1]['PTS'])

                return float(team_stats.iloc[0]['PTS'])
            
            except Exception as e:
                print(f"An error occurred pts: {e}")
                return None

        def fetch_latest_fg(csv_filename, team_id):
            """
            Fetches and returns the pace value for the latest game played by the specified team
            using the BoxScoreAdvancedV2 endpoint.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team_id (str or int): The TEAM_ID of the team.
            
            Returns:
                float or None: The pace value from the latest game, or None if not found.
            """
            try:
                # Read the CSV file to find the latest game played by the team
                df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
                team_id = str(team_id)
                team_games = df[df['TEAM_ID'] == team_id]
                
                if team_games.empty:
                    print(f"No games found for TEAM_ID {team_id}.")
                    return None
                

                # Get the latest game based on GAME_DATE
                latest_game = team_games.sort_values(by='GAME_DATE', ascending=False).iloc[0]
                game_id = str(latest_game['GAME_ID']).zfill(10)

                time.sleep(0.2)  # Respect API rate limits

                # Fetch data from BoxScoreTraditionalV2 for the entire game
                boxscore = BoxScoreTraditionalV2(
                    game_id=game_id,
                    start_period=0, 
                    end_period=10,   # Fetch all periods in the game
                    range_type=0,
                    start_range=0,
                    end_range=0
                )

                # Access the team stats dataset
                team_stats = boxscore.team_stats.get_data_frame()
                if str(team_stats.iloc[0]['TEAM_ID']) != team_id:
                    return float(team_stats.iloc[1]['FG_PCT'])

                return float(team_stats.iloc[0]['FG_PCT'])
            
            except Exception as e:
                print(f"An error occurred pace: {e}")
                return None

        def calc_off(csv_filename, team_id, game_id):
            df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
            team_id = str(team_id)
            team_games = df[df['TEAM_ID'] == team_id]
            if team_games.empty:
                print(f"No games found for TEAM_ID {team_id}.")
                return None
            latest_game = team_games.sort_values(by='GAME_DATE', ascending=False).iloc[0]
            season_avg_pace = float(latest_game['Off_Rating_Avg'])

            csv_file = 'final_training_dataset.csv'

            off = ((season_avg_pace * num_of_games(csv_filename, team_id)) + fetch_latest_off(csv_file, team_id, game_id)) / (num_of_games(csv_filename, team_id) + 1)
            return off

        def calc_pace(csv_filename, team_id):
            df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
            team_id = str(team_id)
            team_games = df[df['TEAM_ID'] == team_id]
            if team_games.empty:
                print(f"No games found for TEAM_ID {team_id}.")
                return None
            latest_game = team_games.sort_values(by='GAME_DATE', ascending=False).iloc[0]
            season_avg_pace = float(latest_game['Season_Avg_Pace_Team'])

            csv_file = 'final_training_dataset.csv'

            pace = ((season_avg_pace * num_of_games(csv_filename, team_id)) + fetch_latest_pace(csv_file, team_id)) / (num_of_games(csv_filename, team_id) + 1)
            return pace

        def calc_fg(csv_filename, team_id):
            df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
            team_id = str(team_id)
            team_games = df[df['TEAM_ID'] == team_id]
            if team_games.empty:
                print(f"No games found for TEAM_ID {team_id}.")
                return None
            latest_game = team_games.sort_values(by='GAME_DATE', ascending=False).iloc[0]
            season_avg_fg = float(latest_game['Season_Avg_FG_PCT_Team'])

            csv_filename = 'final_training_dataset.csv'

            fg = ((season_avg_fg * num_of_games(csv_filename, team_id)) + fetch_latest_fg(csv_filename, team_id)) / (num_of_games(csv_filename, team_id) + 1)
            return fg

        def calc_pts(csv_filename, team_id, game_id):
            df = pd.read_csv(csv_filename, dtype={'TEAM_ID': str}, parse_dates=['GAME_DATE'])
            team_id = str(team_id)
            team_games = df[df['TEAM_ID'] == team_id]
            if team_games.empty:
                print(f"No games found for TEAM_ID {team_id}.")
                return None
            latest_game = team_games.sort_values(by='GAME_ID', ascending=False).iloc[0]
            season_avg_pts = float(latest_game['PPG_Team'])
            

            csv_filename = 'final_training_dataset.csv'

            pts = ((season_avg_pts * num_of_games(csv_filename, team_id)) + fetch_latest_pts(csv_filename, team_id, game_id)) / (num_of_games(csv_filename, team_id) + 1)
            return pts

        def get_matchup_avg_first_qtr(csv_filename, team1_id, team2_id):
            """
            Calculates the average total first quarter points for the three most recent matchups between two teams
            and stores their GAME_IDs in an array.
            
            Parameters:
                csv_filename (str): Path to the CSV file containing game data.
                team1_id (str or int): TEAM_ID of the first team.
                team2_id (str or int): TEAM_ID of the second team.
            
            Returns:
                float: Average total first quarter points.
                list: List of GAME_IDs of the selected matchups.
            """
            try:
                df = pd.read_csv(csv_filename, dtype={'GAME_ID': str, 'TEAM_ID': str, 'OPPONENT_TEAM_ID': str}, parse_dates=['GAME_DATE'])
                team1_id = str(team1_id)
                team2_id = str(team2_id)
                
                matchups = df[((df['TEAM_ID'] == team1_id) & (df['OPPONENT_TEAM_ID'] == team2_id)) |
                            ((df['TEAM_ID'] == team2_id) & (df['OPPONENT_TEAM_ID'] == team1_id))]
                
                if matchups.empty:
                    print("No matchups found.")
                    return None, []
                
                # Remove duplicate GAME_IDs
                matchups = matchups.drop_duplicates(subset='GAME_ID')

                latest_three = matchups.sort_values(by='GAME_DATE', ascending=False).head(3)
                
                game_ids = latest_three['GAME_ID'].tolist()


                total_first_qtr = 0
                for id in game_ids:
                    id = str(id).zfill(10)
                    total_first_qtr += fetch_pts_qtr1(id, team1_id) + fetch_pts_qtr1(id, team2_id)

                average = total_first_qtr/3
                
                return average
            
            except FileNotFoundError:
                print(f"CSV file '{csv_filename}' not found.")
                return None, []
            except Exception as e:
                print(f"An error occurred: {e}")
                return None, []

        from collections import deque

        def shift_right(arr):
            """
            Shifts the array to the right by one position using deque.
            
            Parameters:
                arr (list): The input array.
            
            Returns:
                list: The shifted array.
            """
            d = deque(arr)
            d.rotate(1)  # Rotate the deque to the right by 1
            return list(d)

        def append_game_data(game_id, csv_filename='final_training_dataset.csv'):
            """
            Fetches game data for the given GAME_ID and appends it to the CSV.
            
            Parameters:
                game_id (str): The GAME_ID of the game to fetch.
                csv_filename (str): Path to the CSV file.
            """
            try:
                # Fetch game details using LeagueGameFinder
                game_finder = LeagueGameFinder(game_id_nullable=game_id)
                time.sleep(0.2)  # Respect API rate limits
                games = game_finder.get_data_frames()[0]

                # Filter to the exact game_id
                game = games[games['GAME_ID'] == game_id]
                if game.empty:
                    print(f"No matching game details found for GAME_ID {game_id}.")
                    return
                game = game.iloc[0]
                game_date = game['GAME_DATE']
                matchup = game['MATCHUP']
                month = pd.to_datetime(game_date).month

                # Determine Home or Away
                team_abbrev = game['TEAM_ABBREVIATION']

                if '@' in matchup:
                    # Format: AwayTeam @ HomeTeam
                    teams = matchup.split('@')
                    away_team = teams[0].strip()
                    home_team = teams[1].strip()
                    home_away = 'Away' 
                    mtch = home_team + " vs. " + away_team
                    alt_HA = 'Home' 
                    tmp = matchup
                    matchup = mtch
                    mtch = tmp
                elif 'vs' in matchup:
                    # Format: HomeTeam vs AwayTeam
                    teams = matchup.split('vs.')
                    home_team = teams[0].strip()
                    away_team = teams[1].strip()
                    home_away = 'Home' 
                    mtch = away_team + " @ " + home_team
                    alt_HA = 'Away' 
                    tmp = matchup
                    matchup = mtch
                    mtch = tmp
                else:
                    home_away = 'Unknown'

                # Fetch box score summary
                boxscore = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
                time.sleep(0.2)  # To respect API rate limits

                line_score = boxscore.line_score.get_data_frame()
                if line_score.empty or len(line_score) < 2:
                    print(f"Insufficient data for GAME_ID {game_id}.")
                    return
                
                # Extract team and opponent data
                team = line_score.iloc[0]
                opponent = line_score.iloc[1]

                game_id = game_id.zfill(10)

                g_id = fetch_pts_qtr1_game_id(csv_filename, team['TEAM_ID'], game_date)
                g_id_opp = fetch_pts_qtr1_game_id(csv_filename, opponent['TEAM_ID'], game_date)

                
                recent_array = get_recent_trend(csv_file, team['TEAM_ID'], game_date)
                recent_array.append(fetch_pts_qtr1(g_id, team['TEAM_ID']))
                recent_array = shift_right(recent_array)

                recent_array_opp = get_recent_trend(csv_file, opponent['TEAM_ID'], game_date)
                recent_array_opp.append(fetch_pts_qtr1(g_id_opp, opponent['TEAM_ID']))
                recent_array_opp = shift_right(recent_array_opp)

                h2h = get_matchup_avg_first_qtr(csv_filename, team['TEAM_ID'], opponent['TEAM_ID'])

                pace_team = calc_pace(csv_filename, team['TEAM_ID'])
                opponent_pace_team = calc_pace(csv_filename, opponent['TEAM_ID'])

                fg_team = calc_fg(csv_filename, team['TEAM_ID'])
                fg_opponent = calc_fg(csv_filename, opponent['TEAM_ID'])

                pts_team = fetch_latest_pts(csv_file, team['TEAM_ID'], game_id)
                pts_opponent = fetch_latest_pts(csv_file, opponent['TEAM_ID'], game_id)

                ppg = calc_pts(csv_file, team['TEAM_ID'], game_id)
                opp_ppg = calc_pts(csv_file, opponent['TEAM_ID'], game_id)

                off_rating = fetch_latest_off(csv_file, team['TEAM_ID'], game_id)
                opp_off_rating = fetch_latest_off(csv_file, opponent['TEAM_ID'], game_id)
                off_rating_avg = calc_off(csv_filename, team['TEAM_ID'], game_id)
                opp_off_rating_avg = calc_off(csv_filename, opponent['TEAM_ID'], game_id)

                # Prepare game data
                game_data = {
                    'GAME_ID': game_id,
                    'TEAM_ID': team['TEAM_ID'],
                    'TEAM_ABBREVIATION': team['TEAM_ABBREVIATION'],
                    'OPPONENT_TEAM_ID': opponent['TEAM_ID'],
                    'OPPONENT_TEAM_ABBREVIATION': opponent['TEAM_ABBREVIATION'],
                    'PTS_QTR1': team['PTS_QTR1'],
                    'OPP_PTS_QTR1': opponent['PTS_QTR1'],
                    'Total_First_Quarter_Points': team['PTS_QTR1'] + opponent['PTS_QTR1'],
                    'GAME_DATE': game_date,
                    'MATCHUP': matchup,
                    'MONTH': month,
                    'Home_Away': home_away,
                    'Recent_Trend_Team': recent_array,
                    'Recent_Trend_Opponent': recent_array_opp,
                    'Head_to_Head_Q1': h2h,
                    'SEASON': '2024-25',
                    'Season_Avg_Pace_Team': pace_team,
                    'Season_Avg_Pace_Opponent': opponent_pace_team,
                    'Average_Season_Avg_Pace': (pace_team + opponent_pace_team) / 2,
                    'Season_Avg_FG_PCT_Team' : fg_team,
                    'Season_Avg_FG_PCT_Opponent': fg_opponent,
                    'PTS': pts_team,
                    'OPP_PTS': pts_opponent,
                    'Off_Rating': off_rating,
                    'PPG_Team': ppg,
                    'PPG_Opponent': opp_ppg,
                    'Off_Rating_Avg': off_rating_avg,
                    'Opp_Off_Rating_Avg': opp_off_rating_avg,
                }
                
                game_data.pop("PTS_QTR1")
                game_data.pop("OPP_PTS_QTR1")



                # Convert to DataFrame
                new_row = pd.DataFrame([game_data])
                
                # Append to CSV
                new_row.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Appended GAME_ID {game_id} to {csv_filename}.")
                
                # Prepare game data
                game_data = {
                    'GAME_ID': game_id,
                    'TEAM_ID': opponent['TEAM_ID'],
                    'TEAM_ABBREVIATION': opponent['TEAM_ABBREVIATION'],
                    'OPPONENT_TEAM_ID': team['TEAM_ID'],
                    'OPPONENT_TEAM_ABBREVIATION': team['TEAM_ABBREVIATION'],
                    'PTS_QTR1': opponent['PTS_QTR1'],
                    'OPP_PTS_QTR1': team['PTS_QTR1'],
                    'Total_First_Quarter_Points': team['PTS_QTR1'] + opponent['PTS_QTR1'],
                    'GAME_DATE': game_date,
                    'MATCHUP': mtch,
                    'MONTH': month,
                    'Home_Away': alt_HA,
                    'Recent_Trend_Team': recent_array_opp,
                    'Recent_Trend_Opponent': recent_array,
                    'Head_to_Head_Q1': h2h,
                    'SEASON': '2024-25',
                    'Season_Avg_Pace_Team': opponent_pace_team,
                    'Season_Avg_Pace_Opponent': pace_team,
                    'Average_Season_Avg_Pace': (pace_team + opponent_pace_team) / 2,
                    'Season_Avg_FG_PCT_Team' : fg_opponent,
                    'Season_Avg_FG_PCT_Opponent': fg_team,
                    'PTS': pts_opponent,
                    'OPP_PTS': pts_team,
                    'Off_Rating': opp_off_rating,
                    'PPG_Team': opp_ppg,
                    'PPG_Opponent': ppg,
                    'Off_Rating_Avg': opp_off_rating_avg,
                    'Opp_Off_Rating_Avg': off_rating_avg,
                }
                
                print("This is matchup: " + matchup)
                print("This is mtch" + mtch)

                game_data.pop("PTS_QTR1")
                game_data.pop("OPP_PTS_QTR1")

                # Convert to DataFrame
                new_row = pd.DataFrame([game_data])
                
                # Append to CSV
                new_row.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Appended GAME_ID {game_id} to {csv_filename}.")

            except Exception as e:
                print(f"Error fetching or appending data for GAME_ID {game_id}: {e}")
        

        

        csv_file = 'final_training_dataset.csv'

        # Fetch and append new game data
        append_game_data(new_game_id, csv_filename)
        count += 1
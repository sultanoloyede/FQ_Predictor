from heapq import merge
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (leaguegamefinder, boxscoresummaryv2, boxscoreadvancedv2, boxscoretraditionalv2)
import time
from tqdm import tqdm
import pickle
import os
from collections import defaultdict, deque
from statistics import mean

database = pd.DataFrame()

# FETCH GAME IDS
def fetch_game_ids():
    #seasons = np.array(['2017-18', '2018-19', '2019-20', '2020-21', '2021,22', '2022-23', '2023-24'])

    seasons = ['2022-23', '2023-24', '2024-25']
    szn_games = []


    print("Fetching NBA game ids...")

    for season in seasons:
        print(f"Processing the {season} season...")

        # Gets all the games for a specific season
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, season_type_nullable='Regular Season'
        )

        games = gamefinder.get_data_frames()[0]

        # Convert GAME_ID to numeric, handling errors
        games['GAME_ID_INT'] = pd.to_numeric(games['GAME_ID'], errors='coerce')
        games = games.dropna(subset=['GAME_ID_INT'])
        games['GAME_ID_INT'] = games['GAME_ID_INT'].astype(int)

        # Filter out G League games
        games = games[games['GAME_ID_INT'] < 100000000]

        szn_games.append(games)
        print(len(games))
        time.sleep(0.5)

    return pd.concat(szn_games, ignore_index=True)

# FETCH GAME DATA
def fetch_game_data(game_id):
    """Fetches and structures game data for a given GAME_ID."""
    game_data = {}
    try:
        # Fetch Box Score Summary
        boxscore_summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        line_score_df = boxscore_summary.line_score.get_data_frame()
        game_summary_df = boxscore_summary.game_summary.get_data_frame()

        if line_score_df.empty:
            print(f"No data available for game {game_id}")
            return None

        # Extract relevant columns
        line_score_df = line_score_df[['TEAM_ID', 'TEAM_ABBREVIATION', 'PTS_QTR1', 'GAME_DATE_EST']]
        game_summary_df = game_summary_df.iloc[0]['SEASON']

        # Ensure two teams are present
        if len(line_score_df) != 2:
            print(f"Invalid number of teams in game {game_id}")
            return None

        # Assign team and opponent data
        team1 = line_score_df.iloc[0]
        team2 = line_score_df.iloc[1]
        
        # String build season
        season = str(game_summary_df) + "-" + str(int(game_summary_df) + 1)[2:]

        # String build date
        date = str(line_score_df['GAME_DATE_EST'].iloc[0])[:10]

        # Create entries for both teams
        for team, opponent in [(team1, team2), (team2, team1)]:
            data = {
                'GAME_ID': game_id,
                'TEAM_ID': team['TEAM_ID'],
                'TEAM_ABBREVIATION': team['TEAM_ABBREVIATION'],
                'OPPONENT_TEAM_ID': opponent['TEAM_ID'],
                'OPPONENT_TEAM_ABBREVIATION': opponent['TEAM_ABBREVIATION'],
                'SEASON': season,
                'PTS_QTR1': team['PTS_QTR1'],
                'PTS_QTR1_OPP': opponent['PTS_QTR1'],
                'GAME_DATE': date
            }
            key = f"{game_id}_{team['TEAM_ID']}"
            game_data[key] = data
        return game_data
    except Exception as e:
        print(f"Error fetching data for game {game_id}: {e}")
        return None

def load_data(filename):
    #Loads data from a pickle file.
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"No existing data file found at {filename}")
        return []

def save_data(data, filename):
    #Saves data to a pickle file.
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

# FETCH ADV BOX SCORES
def fetch_adv2(game_id):
    # FQP dictionary
    adv2_data = {}

    # Fetch ADV2
    adv2 = boxscoreadvancedv2.BoxScoreAdvancedV2(
        game_id=game_id,
        start_period=1, 
        end_period=1,
        range_type=1,
        start_range=1,
        end_range=1
    )
    team_stats_df = adv2.team_stats.get_data_frame()

    # Extract relevant columns
    team_stats_df = team_stats_df[['GAME_ID', 'TEAM_ID', 'EFG_PCT', 'PACE', 'AST_TOV', 'DEF_RATING', 'OFF_RATING']]

    # Assign team and opponent data
    team1 = team_stats_df.iloc[0]
    team2 = team_stats_df.iloc[1]

    # Create entries for both teams, entries will be key value, key will be "game-id_team-id", value will be pts-team1, pts-team2
    for team, opponent in [(team1, team2), (team2, team1)]:
        data = {
            'GAME_ID': game_id,
            'TEAM_ID': team['TEAM_ID'],
            'OPPONENT_TEAM_ID': opponent['TEAM_ID'],
            'EFG_PCT': team['EFG_PCT'],
            'EFG_PCT_OPP': opponent['EFG_PCT'],
            'PACE': team['PACE'],
            'PACE_OPP': opponent['PACE'],
            'AST_TOV': team['AST_TOV'],
            'AST_TOV_OPP': opponent['AST_TOV'],
            'DEF_RATING': team['DEF_RATING'],
            'DEF_RATING_OPP': opponent['DEF_RATING'],
            'OFF_RATING': team['OFF_RATING'],
            'OFF_RATING_OPP': opponent['OFF_RATING'],
        }
        key = f"{game_id}_{team['TEAM_ID']}"
        adv2_data[key] = data

    return adv2_data

# FETCH ADV BOX SCORES
def fetch_bs_trad(game_id):
    # FQP dictionary
    bs_trad_data = {}

    # Fetch ADV2
    bs_trad = boxscoretraditionalv2.BoxScoreTraditionalV2(
        game_id=game_id,
        start_period=1, 
        end_period=1,   # First period data
        range_type=1,
        start_range=1,
        end_range=1,
        )
    team_stats_df = bs_trad.team_stats.get_data_frame()

    # Extract relevant columns
    team_stats_df = team_stats_df[['GAME_ID', 'TEAM_ID', 'FGA', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FTA', 'TO', 'OREB', 'DREB']]

    # Assign team and opponent data
    team1 = team_stats_df.iloc[0]
    team2 = team_stats_df.iloc[1]

    # Create entries for both teams, entries will be key value, key will be "game-id_team-id", value will be pts-team1, pts-team2
    for team, opponent in [(team1, team2), (team2, team1)]:
        data = {
            'GAME_ID': game_id,
            'TEAM_ID': team['TEAM_ID'],
            'OPPONENT_TEAM_ID': opponent['TEAM_ID'],
            'FGA_FQ': team['FGA'],
            'FGA_OPP_FQ': opponent['FGA'],
            'FG_PCT_FQ': team['FG_PCT'],
            'FG_PCT_OPP_FQ': opponent['FG_PCT'],
            'FG3_PCT_FQ': team['FG3_PCT'],
            'FG3_PCT_OPP_FQ': opponent['FG3_PCT'],
            'FT_PCT_FQ': team['FT_PCT'],
            'FT_PCT_OPP_FQ': opponent['FT_PCT'],
            'FTA_FQ': team['FTA'],
            'FTA_OPP_FQ': opponent['FTA'],
            'TO_FQ': team['TO'],
            'TO_OPP_FQ': opponent['TO'],
            'OREB_FQ': team['OREB'],
            'OREB_OPP_FQ': opponent['OREB'],
            'DREB_FQ': team['DREB'],
            'DREB_OPP_FQ': opponent['DREB'],
        }
        key = f"{game_id}_{team['TEAM_ID']}"
        bs_trad_data[key] = data

    return bs_trad_data

# PART 1: FETCH GAME IDS, TEAM NAMES, TEAM IDS, FQP
########################################################################################################################

def construct_games(g_ids):
    # Define filename for initial dataset save files
    data_filename = 'basic_game_data.pkl'

    # Load existing processed data if available
    game_data = load_data(data_filename)
    processed_game_ids = set()

    # If data was loaded, extract already processed GAME_IDs
    if game_data:
        processed_game_ids = set([data['GAME_ID'] for data in game_data])
        print(f"Resuming from existing data. {len(processed_game_ids)} games already processed.")
    else:
        game_data = []
        print("Starting data collection from scratch.")

    # Define batch size
    batch_size = 100  
    total_games = len(g_ids)
    batches = (total_games // batch_size) + (1 if total_games % batch_size > 0 else 0)

    # Process in batches with a progress bar
    print("Processing game data in batches...")
    for batch_num in tqdm(range(batches), desc="Game data batches"):
        # Define start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_games)

        # Go through all the game ids
        for idx in range(start_idx, end_idx):
            g_id = g_ids[idx]
            # Skip processed games
            if g_id in processed_game_ids:
                continue

            # Fetch game data for current game and add to game data array
            g_data = fetch_game_data(g_id)
            game_data.extend(g_data.values())

            # Print game data every 10 games
            if idx % 10 == 0:
                print(g_data)

            # Save data after every 100 games
            if (idx + 1) % 100 == 0 or (idx + 1) == len(g_ids):
                print("Save Checkpoint...")
                save_data(game_data, data_filename)

            time.sleep(0.3)

    # Save the final training data
    save_data(game_data, data_filename)
    print("All game data has been collected and saved.")

    database = pd.DataFrame(game_data).drop_duplicates().reset_index().sort_values(by='GAME_DATE')

    # remove duplicate rows based on all columns
    database.drop_duplicates(inplace=True)

    # Check for a pre-existing checkpoint
    checkpoint_filename = 'basic_database.csv'
    if os.path.exists(checkpoint_filename):
        database = pd.read_csv(checkpoint_filename)
        print("Loaded database from checkpoint file.")
    else:
        # Save the checkpoint
        database.to_csv(checkpoint_filename, index=False)
        print(f"Checkpoint saved to '{checkpoint_filename}'.")

def sum_fqp(csv):
    database = pd.read_csv(csv)

    database['TOTAL_FQP'] = database['PTS_QTR1'] + database['PTS_QTR1_OPP']

    print(database.tail())
    return database

# PART 2: FETCH ADV3
########################################################################################################################

def construct_adv2_db(g_ids):
    # Define filename for fqp save files
    adv2_data_filename = 'adv2_game_data.pkl'

    # Load existing processed data if available
    adv2_game_data = load_data(adv2_data_filename)

    processed_adv2_games = set()

    # If data was loaded, extract already processed GAME_IDs
    if adv2_game_data:
        processed_adv2_games = set([data['GAME_ID'] for data in adv2_game_data])
        print(f"Resuming from existing data. {len(processed_adv2_games)} games already processed.")
    else:
        adv2_game_data = []
        print("Starting data collection from scratch.")

    # Define batch size
    batch_size = 100  
    total_games = len(g_ids)
    batches = (total_games // batch_size) + (1 if total_games % batch_size > 0 else 0)

    # Process in batches with a progress bar
    print("Processing adv2 data in batches...")
    for batch_num in tqdm(range(batches), desc="adv2 game data batches"):
        # Define start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_games)

        # Go through all the game ids
        for idx in range(start_idx, end_idx):
            # Get single game id from game id array based on index pos
            g_id = g_ids[idx]
            # Skip processed games
            if g_id in processed_adv2_games:
                continue

            # Fetch game data for current game and add to game data array
            g_data = fetch_adv2(g_id)
            adv2_game_data.extend(g_data.values())

            # Print game data every 10 games
            if idx % 10 == 0:
                print(g_data)

            # Save data after every 100 games
            if (idx + 1) % 100 == 0 or (idx + 1) == len(g_ids):
                print("Save Checkpoint...")
                save_data(adv2_game_data, adv2_data_filename)

            time.sleep(0.5)

    # Save the FQP game data
    save_data(adv2_game_data, adv2_data_filename)
    print("All game data has been collected and saved.")

    fq_database = pd.DataFrame(adv2_game_data).drop_duplicates().reset_index().sort_values(by='GAME_ID')

    checkpoint_filename = 'adv2_database.csv'
    fq_database.to_csv(checkpoint_filename, index=False)

# PART 3: FETCH BOX SCORE TRAD
########################################################################################################################

def construct_bst_db(g_ids):
    # Define filename for fqp save files
    bst_data_filename = 'bst_game_data.pkl'

    # Load existing processed data if available
    bst_game_data = load_data(bst_data_filename)

    processed_bst_games = set()

    # If data was loaded, extract already processed GAME_IDs
    if bst_game_data:
        processed_bst_games = set([data['GAME_ID'] for data in bst_game_data])
        print(f"Resuming from existing data. {len(processed_bst_games)} games already processed.")
    else:
        bst_game_data = []
        print("Starting data collection from scratch.")

    # Define batch size
    batch_size = 100  
    total_games = len(g_ids)
    batches = (total_games // batch_size) + (1 if total_games % batch_size > 0 else 0)

    # Process in batches with a progress bar
    print("Processing bst data in batches...")
    for batch_num in tqdm(range(batches), desc="BST game data batches"):
        # Define start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_games)

        # Go through all the game ids
        for idx in range(start_idx, end_idx):
            # Get single game id from game id array based on index pos
            g_id = g_ids[idx]
            # Skip processed games
            if g_id in processed_bst_games:
                continue

            # Fetch game data for current game and add to game data array
            g_data = fetch_bs_trad(g_id)
            bst_game_data.extend(g_data.values())

            # Print game data every 10 games
            if idx % 10 == 0:
                print(g_data)

            # Save data after every 100 games
            if (idx + 1) % 100 == 0 or (idx + 1) == len(g_ids):
                print("Save Checkpoint...")
                save_data(bst_game_data, bst_data_filename)

            time.sleep(0.5)

    # Save the FQP game data
    save_data(bst_game_data, bst_data_filename)
    print("All game data has been collected and saved.")

    fq_database = pd.DataFrame(bst_game_data).drop_duplicates().reset_index().sort_values(by='GAME_ID')

    checkpoint_filename = 'bst_database.csv'
    fq_database.to_csv(checkpoint_filename, index=False)

# PART 4: Merge FQP to Base Dataset
########################################################################################################################
def sum_fqp(csv):
    database = pd.read_csv(csv)

    database['TOTAL_FQP'] = database['PTS_QTR1'] + database['PTS_QTR1_OPP']

    csv_new = 'merge.csv'
    database.to_csv(csv_new, index=False)
    return database

# PART 5: ROLLING AVG 10
########################################################################################################################
def rolling_avg_10(df, team):
    df = df[df['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_DATE')

    # Initialize new columns
    df['TEAM_LAST_10'] = df.iloc[0]['PTS_QTR1']
    df['TEAM_WORST_2'] = df.iloc[0]['PTS_QTR1']

    for i in range(len(df)):
        if i == 0:
            continue

        # Check if the value in col2 changes
        if df.loc[i, 'SEASON'] != df.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df.loc[i, 'TEAM_LAST_10'] = df.loc[i, 'PTS_QTR1']
            df.loc[i, 'TEAM_WORST_2'] = df.loc[i, 'PTS_QTR1']
            if i + 1 < len(df):
                df.loc[i + 1, 'TEAM_LAST_10'] = df.loc[i, 'PTS_QTR1']
                df.loc[i + 1, 'TEAM_WORST_2'] = df.loc[i, 'PTS_QTR1']
        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df.loc[i, 'SEASON']].iloc[-10:]['PTS_QTR1']
            
            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df.loc[i, 'TEAM_LAST_10'] = valid_data_avg_10.mean()
            
            # lowest_2: Use the avg_10 slice, find the lowest 2 values, and calculate their average
            if len(valid_data_avg_10) > 1:
                lowest_2_values = valid_data_avg_10.nsmallest(2)
                df.loc[i, 'TEAM_WORST_2'] = lowest_2_values.mean()
            elif len(valid_data_avg_10) == 1:
                df.loc[i, 'TEAM_WORST_2'] = valid_data_avg_10.iloc[0]

    # Print the updated DataFrame
    return df

# PART 6: ROLLING AVG 10 (PART 2)
########################################################################################################################
def add_rolling_avg(df):
    csv2 = 'last10_database.csv'

    unique_values = df['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    for team in unique_values:
        x = rolling_avg_10(df, team)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    combined_df = combined_df.sort_values(by='GAME_DATE')

    combined_df.to_csv(csv2, index=False)

    # Shift the columns
    combined_df['LAST_10_OPP'] = combined_df['TEAM_LAST_10'].shift(-1)
    combined_df['WORST_2_OPP'] = combined_df['TEAM_WORST_2'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'LAST_10_OPP'] = combined_df.loc[even_indices, 'TEAM_LAST_10'].values
    combined_df.loc[odd_indices, 'WORST_2_OPP'] = combined_df.loc[even_indices, 'TEAM_WORST_2'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    csv3 = 'last10_tmp_database.csv'
    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 7: ROLLING FGA AVG
########################################################################################################################
def rolling_fga(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_DATE')

    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_FGA'] = df2.iloc[0]['FGA_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_FGA'] = df2.loc[i, 'FGA_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_FGA'] = df2.loc[i, 'FGA_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['FGA_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_FGA'] = valid_data_avg_10.mean()

    return df1

# PART 8: ROLLING FGA AVG (PART 2)
########################################################################################################################
def add_rolling_fga():
    csv = 'last10_tmp_database.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    bst = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    bst = pd.concat([season, bst], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_fga(main, bst, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'fga.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['FGA_OPP'] = combined_df['TEAM_FGA'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'FGA_OPP'] = combined_df.loc[even_indices, 'TEAM_FGA'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 9: ROLLING PACE AVG
########################################################################################################################
def rolling_pace(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_PACE'] = df2.iloc[0]['PACE']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_PACE'] = df2.loc[i, 'PACE']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_PACE'] = df2.loc[i, 'PACE']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['PACE']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_PACE'] = valid_data_avg_10.mean()

    return df1

# PART 10: ROLLING PACE AVG (PART 2)
########################################################################################################################
def add_rolling_pace():
    csv = 'fga.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'adv2_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_pace(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'pace.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['PACE_OPP'] = combined_df['TEAM_PACE'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'PACE_OPP'] = combined_df.loc[even_indices, 'TEAM_PACE'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 11: ROLLING AVG 10 AGAINST
########################################################################################################################
def rolling_avg_10_against(df, team):
    df = df[df['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_DATE')

    # Initialize new columns
    df['TEAM_LAST_10_AGAINST'] = df.iloc[0]['PTS_QTR1_OPP']
    df['TEAM_WORST_2_AGAINST'] = df.iloc[0]['PTS_QTR1_OPP']

    for i in range(len(df)):
        if i == 0:
            continue

        # Check if the value in col2 changes
        if df.loc[i, 'SEASON'] != df.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df.loc[i, 'TEAM_LAST_10_AGAINST'] = df.loc[i, 'PTS_QTR1_OPP']
            df.loc[i, 'TEAM_WORST_2_AGAINST'] = df.loc[i, 'PTS_QTR1_OPP']
            if i + 1 < len(df):
                df.loc[i + 1, 'TEAM_LAST_10_AGAINST'] = df.loc[i, 'PTS_QTR1_OPP']
                df.loc[i + 1, 'TEAM_WORST_2_AGAINST'] = df.loc[i, 'PTS_QTR1_OPP']
        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df.loc[i, 'SEASON']].iloc[-10:]['PTS_QTR1_OPP']
            
            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df.loc[i, 'TEAM_LAST_10_AGAINST'] = valid_data_avg_10.mean()
            
            # lowest_2: Use the avg_10 slice, find the lowest 2 values, and calculate their average
            if len(valid_data_avg_10) > 1:
                lowest_2_values = valid_data_avg_10.nsmallest(2)
                df.loc[i, 'TEAM_WORST_2_AGAINST'] = lowest_2_values.mean()
            elif len(valid_data_avg_10) == 1:
                df.loc[i, 'TEAM_WORST_2_AGAINST'] = valid_data_avg_10.iloc[0]

    # Print the updated DataFrame
    return df

# PART 12: ROLLING AVG 10 AGAINST (PART 2)
########################################################################################################################
def add_rolling_avg_against(df):
    csv2 = 'pace.csv'

    unique_values = df['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    for team in unique_values:
        x = rolling_avg_10_against(df, team)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    combined_df = combined_df.sort_values(by='GAME_DATE')

    combined_df.to_csv(csv2, index=False)

    # Shift the columns
    combined_df['LAST_10_OPP_AGAINST'] = combined_df['TEAM_LAST_10_AGAINST'].shift(-1)
    combined_df['WORST_2_OPP_AGAINST'] = combined_df['TEAM_WORST_2_AGAINST'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'LAST_10_OPP_AGAINST'] = combined_df.loc[even_indices, 'TEAM_LAST_10_AGAINST'].values
    combined_df.loc[odd_indices, 'WORST_2_OPP_AGAINST'] = combined_df.loc[even_indices, 'TEAM_WORST_2_AGAINST'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    csv3 = 'last10_against_database.csv'
    combined_df.to_csv(csv3, index=False)

    print(combined_df.head())

    return combined_df

# PART 13: ROLLING EFG AVG
########################################################################################################################
def rolling_efg(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_EFG'] = df2.iloc[0]['EFG_PCT']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_EFG'] = df2.loc[i, 'EFG_PCT']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_EFG'] = df2.loc[i, 'EFG_PCT']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['EFG_PCT']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_EFG'] = valid_data_avg_10.mean()

    return df1

# PART 14: ROLLING EFG AVG (PART 2)
########################################################################################################################
def add_rolling_efg():
    csv = 'last10_against_database.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'adv2_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_efg(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'efg.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['EFG_OPP'] = combined_df['TEAM_EFG'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'EFG_OPP'] = combined_df.loc[even_indices, 'TEAM_EFG'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 15: ROLLING FG AVG
########################################################################################################################
def rolling_fg(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_FG'] = df2.iloc[0]['FG_PCT_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_FG'] = df2.loc[i, 'FG_PCT_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_FG'] = df2.loc[i, 'FG_PCT_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['FG_PCT_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_FG'] = valid_data_avg_10.mean()

    return df1

# PART 16: ROLLING FG AVG (PART 2)
########################################################################################################################
def add_rolling_fg():
    csv = 'efg.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_fg(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'fg.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['FG_OPP'] = combined_df['TEAM_FG'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'FG_OPP'] = combined_df.loc[even_indices, 'TEAM_FG'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 17: ROLLING 3P AVG (PART 2)
########################################################################################################################
def add_rolling_3p():
    csv = 'fg.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_3p(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = '3p.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['3P_PCT_OPP'] = combined_df['TEAM_3P_PCT'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, '3P_PCT_OPP'] = combined_df.loc[even_indices, 'TEAM_3P_PCT'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 18: ROLLING 3P AVG
########################################################################################################################
def rolling_3p(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')

    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_3P_PCT'] = df2.iloc[0]['FG3_PCT_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_3P_PCT'] = df2.loc[i, 'FG3_PCT_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_3P_PCT'] = df2.loc[i, 'FG3_PCT_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['FG3_PCT_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_3P_PCT'] = valid_data_avg_10.mean()

    return df1

# PART 19: ROLLING FT AVG
########################################################################################################################
def rolling_ft_pct(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_FT_PCT'] = df2.iloc[0]['FT_PCT_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_FT_PCT'] = df2.loc[i, 'FT_PCT_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_FT_PCT'] = df2.loc[i, 'FT_PCT_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['FT_PCT_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_FT_PCT'] = valid_data_avg_10.mean()

    return df1

# PART 20: ROLLING FT AVG (PART 2)
########################################################################################################################
def add_rolling_ft_pct():
    csv = '3p.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_ft_pct(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'ft_pct.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['FT_PCT_OPP'] = combined_df['TEAM_FT_PCT'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'FT_PCT_OPP'] = combined_df.loc[even_indices, 'TEAM_FT_PCT'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 21: ROLLING FTA
########################################################################################################################
def rolling_fta(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_FTA'] = df2.iloc[0]['FTA_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_FTA'] = df2.loc[i, 'FTA_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_FTA'] = df2.loc[i, 'FTA_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['FTA_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_FTA'] = valid_data_avg_10.mean()

    return df1

# PART 22: ROLLING FT AVG (PART 2)
########################################################################################################################
def add_rolling_fta():
    csv = 'ft_pct.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_fta(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'fta.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['FTA_OPP'] = combined_df['TEAM_FTA'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'FTA_OPP'] = combined_df.loc[even_indices, 'TEAM_FTA'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 23: ROLLING TOV
########################################################################################################################
def rolling_tov(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_TOV'] = df2.iloc[0]['TO_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_TOV'] = df2.loc[i, 'TO_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_TOV'] = df2.loc[i, 'TO_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['TO_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_TOV'] = valid_data_avg_10.mean()

    return df1

# PART 24: ROLLING TOV AVG (PART 2)
########################################################################################################################
def add_rolling_tov():
    csv = 'fta.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_tov(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'tov.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['TOV_OPP'] = combined_df['TEAM_TOV'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'TOV_OPP'] = combined_df.loc[even_indices, 'TEAM_TOV'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 25: ROLLING OREB
########################################################################################################################
def rolling_oreb(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_OREB'] = df2.iloc[0]['OREB_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_OREB'] = df2.loc[i, 'OREB_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_OREB'] = df2.loc[i, 'OREB_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['OREB_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_OREB'] = valid_data_avg_10.mean()

    return df1

# PART 26: ROLLING OREB AVG (PART 2)
########################################################################################################################
def add_rolling_oreb():
    csv = 'tov.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_oreb(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'oreb.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['OREB_OPP'] = combined_df['TEAM_OREB'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'OREB_OPP'] = combined_df.loc[even_indices, 'TEAM_OREB'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 27: ROLLING DREB
########################################################################################################################
def rolling_dreb(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_DREB'] = df2.iloc[0]['DREB_FQ']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_DREB'] = df2.loc[i, 'DREB_FQ']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_DREB'] = df2.loc[i, 'DREB_FQ']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['DREB_FQ']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_DREB'] = valid_data_avg_10.mean()

    return df1

# PART 28: ROLLING DREB AVG (PART 2)
########################################################################################################################
def add_rolling_dreb():
    csv = 'oreb.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'bst_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_dreb(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'dreb.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['DREB_OPP'] = combined_df['TEAM_DREB'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'DREB_OPP'] = combined_df.loc[even_indices, 'TEAM_DREB'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 29: ROLLING OFF RATING AVG
########################################################################################################################
def rolling_off_rating(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_OFF_RATING'] = df2.iloc[0]['OFF_RATING']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_OFF_RATING'] = df2.loc[i, 'OFF_RATING']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_OFF_RATING'] = df2.loc[i, 'OFF_RATING']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['OFF_RATING']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_OFF_RATING'] = valid_data_avg_10.mean()

    return df1

# PART 30: ROLLING OFF AVG (PART 2)
########################################################################################################################
def add_rolling_off_rating():
    csv = 'dreb.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'adv2_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_off_rating(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'off_rating.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['OFF_RATING_OPP'] = combined_df['TEAM_OFF_RATING'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'OFF_RATING_OPP'] = combined_df.loc[even_indices, 'TEAM_OFF_RATING'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 31: ROLLING TOV_AST AVG
########################################################################################################################
def rolling_ast_tov(df1, df2, team, team_id):

    df1 = df1[df1['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_ID')
    
    # Create an empty DataFrame to store the reordered rows
    df2_reordered = pd.DataFrame(columns=df2.columns)

    # Iterate through df1 and find matching rows in df2
    for _, row in df1.iterrows():
        game_id, team_id = row['GAME_ID'], row['TEAM_ID']
        matching_row = df2[(df2['GAME_ID'] == game_id) & (df2['TEAM_ID'] == team_id)]
        df2_reordered = pd.concat([df2_reordered, matching_row], ignore_index=True)

    df2 = df2_reordered

    # Initialize new columns
    df1['TEAM_AST_TOV'] = df2.iloc[0]['AST_TOV']

    for i in range(len(df1)):
        if i == 0:
            continue

        if df2.loc[i, 'SEASON'] != df2.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df1.loc[i, 'TEAM_AST_TOV'] = df2.loc[i, 'AST_TOV']
            if i + 1 < len(df2):
                df1.loc[i + 1, 'TEAM_AST_TOV'] = df2.loc[i, 'AST_TOV']

        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df2.loc[:i - 1]
            valid_data_avg_10 = prev_rows[prev_rows['SEASON'] == df2.loc[i, 'SEASON']].iloc[-10:]['AST_TOV']

            # avg_10: Take the average of up to 10 previous valid values
            if len(valid_data_avg_10) > 0:
                df1.loc[i, 'TEAM_AST_TOV'] = valid_data_avg_10.mean()

    return df1

# PART 32: ROLLING AST_TOV (PART 2)
########################################################################################################################
def add_rolling_ast_tov():
    csv = 'off_rating.csv'
    main = pd.read_csv(csv).sort_values(by='GAME_ID')

    csv2 = 'adv2_database.csv'
    adv2 = pd.read_csv(csv2).sort_values(by='GAME_ID')

    season = main['SEASON']
    adv2 = pd.concat([season, adv2], axis=1)

    unique_values = main['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    team_abv = unique_values[0]

    for team_abv in unique_values:
        team_id = main.loc[main['TEAM_ABBREVIATION'] == team_abv, 'TEAM_ID'].values[0]
        x = rolling_ast_tov(main, adv2, team_abv, team_id)
        if team_abv == 'ATL':
            print(x)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    csv3 = 'ast_tov.csv'

    combined_df = combined_df.sort_values(by='GAME_DATE')

    # Shift the columns
    combined_df['AST_TOV_OPP'] = combined_df['TEAM_AST_TOV'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'AST_TOV_OPP'] = combined_df.loc[even_indices, 'TEAM_AST_TOV'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    print(combined_df.columns)

    combined_df.to_csv(csv3, index=False)

    return combined_df

# PART 33: ROLLING AVG 3
########################################################################################################################
def rolling_avg_3(df, team):
    df = df[df['TEAM_ABBREVIATION'] == team].reset_index(drop=True).sort_values(by='GAME_DATE')

    # Initialize new columns
    df['TEAM_LAST_3'] = df.iloc[0]['PTS_QTR1']

    for i in range(len(df)):
        if i == 0:
            continue

        # Check if the value in col2 changes
        if df.loc[i, 'SEASON'] != df.loc[i - 1, 'SEASON']:
            # Scenario 1: Value changes in col2
            df.loc[i, 'TEAM_LAST_3'] = df.loc[i, 'PTS_QTR1']
            if i + 1 < len(df):
                df.loc[i + 1, 'TEAM_LAST_3'] = df.loc[i, 'PTS_QTR1']
        else:
            # Identify valid data slices in col1 where col2 remains the same
            prev_rows = df.loc[:i - 1]
            valid_data_avg_3 = prev_rows[prev_rows['SEASON'] == df.loc[i, 'SEASON']].iloc[-3:]['PTS_QTR1']
            
            # avg_3: Take the average of up to 3 previous valid values
            if len(valid_data_avg_3) > 0:
                df.loc[i, 'TEAM_LAST_3'] = valid_data_avg_3.mean()

    # Print the updated DataFrame
    return df

# PART 34: ROLLING AVG 3 (PART 2)
########################################################################################################################
def add_rolling_avg3(df):
    csv2 = 'ast_tov.csv'

    unique_values = df['TEAM_ABBREVIATION'].unique()

    combined_df = pd.DataFrame()

    for team in unique_values:
        x = rolling_avg_3(df, team)
        combined_df = pd.concat([combined_df, x], ignore_index=True)

    combined_df = combined_df.sort_values(by='GAME_DATE')

    combined_df.to_csv(csv2, index=False)

    # Shift the columns
    combined_df['LAST_3_OPP'] = combined_df['TEAM_LAST_3'].shift(-1)

    # Assign values to specific rows ensuring alignment of shapes
    even_indices = combined_df.index[::2]  # Indices for even rows
    odd_indices = combined_df.index[1::2]  # Indices for odd rows (to assign values)

    combined_df.loc[odd_indices, 'LAST_3_OPP'] = combined_df.loc[even_indices, 'TEAM_LAST_3'].values

    # Remove rows with NaN caused by the shift
    combined_df = combined_df.dropna().reset_index(drop=True)

    csv3 = 'last3_database.csv'
    combined_df.to_csv(csv3, index=False)

    return combined_df

# Initialize the dictionary of queues with a maximum length of 3 for each queue
queues = defaultdict(lambda: deque(maxlen=3))

# PART 35: PACE and FQ_PTS LAST 3 H2H
#######################################################################################################################
def compute_h2h_metrics(group):
    """
    Computes the LAST_3_H2H, LAST_3_PACE_H2H, and LAST_3_EFG_H2H for each game in the group.
    
    Parameters:
    - group (pd.DataFrame): A DataFrame group representing a unique matchup.
    
    Returns:
    - pd.DataFrame: The group with added 'LAST_3_H2H', 'LAST_3_PACE_H2H', and 'LAST_3_EFG_H2H' columns.
    """
    # Ensure the group is sorted by GAME_DATE
    group = group.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Extract relevant columns as lists for easy access
    total_fqp = group['TOTAL_FQP'].tolist()
    team_pace = group['TEAM_PACE'].tolist()
    pace_opp = group['PACE_OPP'].tolist()
    team_efg = group['TEAM_EFG'].tolist()
    efg_opp = group['EFG_OPP'].tolist()
    n = len(group)
    
    # Initialize lists to store the mean values
    last_3_h2h = []
    last_3_pace_h2h = []
    last_3_efg_h2h = []
    
    for i in range(n):
        # --- Compute LAST_3_H2H ---
        if i >= 1:
            # There are prior matchups
            prior_fqps = total_fqp[max(0, i-3):i]
            mean_h2h = mean(prior_fqps)
        else:
            # First occurrence; get up to the next three 'TOTAL_FQP' values
            next_fqps = total_fqp[i+1:i+4]
            if next_fqps:
                mean_h2h = mean(next_fqps)
            else:
                # If no next games exist, use the current 'TOTAL_FQP'
                mean_h2h = total_fqp[i]
        last_3_h2h.append(mean_h2h)
        
        # --- Compute LAST_3_PACE_H2H ---
        if i == 0:
            # First game; include current and next two games
            window_indices = list(range(i, min(i + 3, n)))
        elif i == 1:
            # Second game; include previous, current, and next game
            window_indices = list(range(max(i - 1, 0), min(i + 2, n)))
        else:
            # Third or later; include previous two and current game
            window_indices = list(range(max(i - 2, 0), i + 1))
        
        # Collect TEAM_PACE and PACE_OPP from the window
        pace_values = []
        for idx in window_indices:
            pace_values.extend([team_pace[idx], pace_opp[idx]])
        
        # Calculate the mean of all collected pace values
        mean_pace = mean(pace_values)
        last_3_pace_h2h.append(mean_pace)
        
        # --- Compute LAST_3_EFG_H2H ---
        # Collect TEAM_EFG and EFG_OPP from the window
        efg_values = []
        for idx in window_indices:
            efg_values.extend([team_efg[idx], efg_opp[idx]])
        
        # Calculate the mean of all collected EFG values
        mean_efg = mean(efg_values)
        last_3_efg_h2h.append(mean_efg)
    
    # Assign the computed means to the respective columns
    group['LAST_3_H2H'] = last_3_h2h
    group['LAST_3_PACE_H2H'] = last_3_pace_h2h
    group['LAST_3_EFG_H2H'] = last_3_efg_h2h
    
    return group

# PART 36: PACE and FQ_PTS LAST 3 H2H (PART 2)
#######################################################################################################################
def h2h():
    df = pd.read_csv('last3_database.csv')

    # Step 1: Sort by GAME_ID ascending
    df_sorted = df.sort_values('GAME_DATE').reset_index(drop=True)

    # Step 2: Create MATCHUP_KEY by sorting TEAM and OPPONENT abbreviations
    df_sorted['MATCHUP_KEY'] = df_sorted.apply(
        lambda row: '_vs_'.join(sorted([row['TEAM_ABBREVIATION'], row['OPPONENT_TEAM_ABBREVIATION']])),
        axis=1
    )

    # Step 3: Extract unique games based on GAME_ID and MATCHUP_KEY
    df_unique = df_sorted.drop_duplicates(subset=['GAME_ID', 'MATCHUP_KEY']).copy()

    # Step 4: Sort unique games by GAME_ID ascending
    df_unique_sorted = df_unique.sort_values('GAME_ID').reset_index(drop=True)

    # Step 6: Group by MATCHUP_KEY and apply the function
    df_unique_with_h2h = df_unique_sorted.groupby('MATCHUP_KEY').apply(compute_h2h_metrics).reset_index(drop=True)

    # Step 7: Merge 'LAST_3_H2H', 'LAST_3_PACE_H2H', and 'LAST_3_EFG_H2H' back to the original sorted DataFrame
    df_unique_h2h = df_unique_with_h2h[['GAME_ID', 'MATCHUP_KEY', 'LAST_3_H2H', 'LAST_3_PACE_H2H', 'LAST_3_EFG_H2H']]

    # Merge back to the original sorted DataFrame
    df_final = df_sorted.merge(
        df_unique_h2h,
        on=['GAME_ID', 'MATCHUP_KEY'],
        how='left'
    )

    # Drop the 'MATCHUP_KEY' if not needed
    df_final = df_final.drop(columns=['MATCHUP_KEY'])

    csv3 = 'efg_pace_h2h.csv'
    df_final.to_csv(csv3, index=False)

    print(df_final)

def xFQTP():
    df = pd.read_csv('efg_pace_h2h.csv')

    # Step 1: Sort the DataFrame by SEASON and GAME_DATE to ensure chronological order
    df_sorted = df.sort_values(['SEASON', 'GAME_DATE']).reset_index(drop=True)

    # Step 2: Calculating Cumulative Means for xFQTP
    # Group by TEAM_ID and SEASON
    team_group = df_sorted.groupby(['TEAM_ID', 'SEASON'])

    # Calculate cumulative means for TEAM_FGA, TEAM_EFG, TEAM_FTA, TEAM_FT_PCT
    df_sorted['Mean_TEAM_FGA'] = team_group['TEAM_FGA'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_TEAM_EFG'] = team_group['TEAM_EFG'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_TEAM_FTA'] = team_group['TEAM_FTA'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_TEAM_FT_PCT'] = team_group['TEAM_FT_PCT'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)

    # For first game of the season, fill NaN with current game's stats
    df_sorted['Mean_TEAM_FGA'].fillna(df_sorted['TEAM_FGA'], inplace=True)
    df_sorted['Mean_TEAM_EFG'].fillna(df_sorted['TEAM_EFG'], inplace=True)
    df_sorted['Mean_TEAM_FTA'].fillna(df_sorted['TEAM_FTA'], inplace=True)
    df_sorted['Mean_TEAM_FT_PCT'].fillna(df_sorted['TEAM_FT_PCT'], inplace=True)

    # Calculate xFQTP using the original formula
    df_sorted['xFQTP'] = (df_sorted['Mean_TEAM_FGA'] * df_sorted['Mean_TEAM_EFG'] * 2) + \
                        (df_sorted['Mean_TEAM_FTA'] * df_sorted['Mean_TEAM_FT_PCT'])

    # Step 3: Calculating Cumulative Means for xFQTP_OPP
    # Group by OPPONENT_TEAM_ID and SEASON
    opp_group = df_sorted.groupby(['OPPONENT_TEAM_ID', 'SEASON'])

    # Calculate cumulative means for FGA_OPP, EFG_OPP, FTA_OPP, FT_PCT_OPP
    df_sorted['Mean_FGA_OPP'] = opp_group['FGA_OPP'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_EFG_OPP'] = opp_group['EFG_OPP'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_FTA_OPP'] = opp_group['FTA_OPP'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)
    df_sorted['Mean_FT_PCT_OPP'] = opp_group['FT_PCT_OPP'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)

    # For first game of the season for opponent, fill NaN with current game's stats
    df_sorted['Mean_FGA_OPP'].fillna(df_sorted['FGA_OPP'], inplace=True)
    df_sorted['Mean_EFG_OPP'].fillna(df_sorted['EFG_OPP'], inplace=True)
    df_sorted['Mean_FTA_OPP'].fillna(df_sorted['FTA_OPP'], inplace=True)
    df_sorted['Mean_FT_PCT_OPP'].fillna(df_sorted['FT_PCT_OPP'], inplace=True)

    # Calculate xFQTP_OPP using the original formula
    df_sorted['xFQTP_OPP'] = (df_sorted['Mean_FGA_OPP'] * df_sorted['Mean_EFG_OPP'] * 2) + \
                            (df_sorted['Mean_FTA_OPP'] * df_sorted['Mean_FT_PCT_OPP'])

    # Step 4: Cleaning Up Intermediate Columns
    # Drop intermediate mean columns if not needed
    columns_to_drop = ['Mean_TEAM_FGA', 'Mean_TEAM_EFG', 'Mean_TEAM_FTA', 'Mean_TEAM_FT_PCT',
                    'Mean_FGA_OPP', 'Mean_EFG_OPP', 'Mean_FTA_OPP', 'Mean_FT_PCT_OPP']
    df_final = df_sorted.drop(columns=columns_to_drop)

    csv3 = 'xFQTP.csv'
    df_final.to_csv(csv3, index=False)

# STEP 1 FETCH GAME IDS

# Fetch all game ids 
g_ids = fetch_game_ids()

# Remove duplicate game ids
g_ids = g_ids['GAME_ID'].unique()


# STEP 2 BUILD BASIC DATASET

# construct_games(g_ids)

# STEP 3 BUILD ADV DATASET

# construct_adv2_db(g_ids)

# STEP 4 BUILD BOX SCORE TRAD DATASET

# construct_bst_db(g_ids)

# STEP 5 CONSTRUCT FQTP COLUMN

# csv = 'basic_database.csv'
# df = sum_fqp(csv)

# STEP 6 LAST_10_COLUMNS ADDED

# last10 = add_rolling_avg(df)

# STEP 7 SHOT ATTEMPTS

# fga = add_rolling_fga()

# STEP 8 PACE

# pace = add_rolling_pace()

# STEP 9 LAST_10_AGAINST

# df = pd.read_csv('pace.csv')
# last10_a = add_rolling_avg_against(df)

# STEP 10 EFG

# efg = add_rolling_efg()

# STEP 11 FG

# fg = add_rolling_fg()

# STEP 12 3P_PCT

# p3 = add_rolling_3p()

# STEP 13 FT_PCT

# ft_pct = add_rolling_ft_pct()

# STEP 14 FT_ATT

# fta = add_rolling_fta()

# STEP 15 TOV

# tov = add_rolling_tov()

# STEP 16 OREB

# oreb = add_rolling_oreb()

# STEP 17 DREB

# dreb = add_rolling_dreb()

# STEP 18 OFF RATING

# off_rating = add_rolling_off_rating()

# STEP 19 AST TOV

# ast_tov = add_rolling_ast_tov()

# STEP 20 LAST_3

# df = pd.read_csv('ast_tov.csv')
# last_3 = add_rolling_avg3(df)

# STEP 21 H2H PACE LAST 3

# h2h()

# STEP 22 xFQTP and xFQTP_OPP

# xFQTP()

import os
import time
import pickle
import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreSummaryV2
from datetime import timedelta
import logging

# -----------------------------
# 1. Configuration and Logging
# -----------------------------

# Define filenames with unique prefixes to prevent conflicts
LOG_FILENAME = 'nba_ai_data_collection.log'
HEAD_TO_HEAD_FILENAME = 'nba_ai_head_to_head_data.pkl'
DATA_FILENAME = 'nba_ai_all_training_data.pkl'
CHECKPOINT_FILENAME = 'nba_ai_training_dataset_with_trends.csv'
FINAL_CSV_FILENAME = 'nba_ai_input_training_dataset.csv'

# Configure logging
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# -----------------------------
# 2. Data Loading and Saving
# -----------------------------

def load_data(filename):
    """Loads data from a pickle file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded data from {filename}.")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {filename}: {e}")
            return {}
    else:
        logging.info(f"No existing data file found at {filename}.")
        return {}

def save_data(data, filename):
    """Saves data to a pickle file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data successfully saved to {filename}.")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")

# -----------------------------
# 3. Fetching Head-to-Head Games
# -----------------------------

def fetch_head_to_head_games(team_id_1, team_id_2, seasons):
    """
    Fetches head-to-head games between two teams over specified seasons.

    Args:
        team_id_1 (int): TEAM_ID of the first team.
        team_id_2 (int): TEAM_ID of the second team.
        seasons (list): List of season strings, e.g., ['2023-24', '2022-23'].

    Returns:
        pd.DataFrame: DataFrame containing head-to-head games.
    """
    head_to_head_games = []

    for season in seasons:
        logging.info(f"Fetching games for season {season} between Team {team_id_1} and Team {team_id_2}")
        try:
            # Fetch games for team 1
            gamefinder1 = LeagueGameFinder(
                team_nullable=team_id_1,
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games1 = gamefinder1.get_data_frames()[0]

            # Fetch games for team 2
            gamefinder2 = LeagueGameFinder(
                team_nullable=team_id_2,
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games2 = gamefinder2.get_data_frames()[0]

            # Merge games where team1 played team2
            merged_games = pd.merge(
                games1,
                games2,
                on='GAME_ID',
                suffixes=('_team1', '_team2')
            )

            # Only keep one entry per game
            if not merged_games.empty:
                merged_games = merged_games.drop_duplicates(subset='GAME_ID')
                head_to_head_games.append(merged_games)
                logging.info(f"Found {len(merged_games)} head-to-head games in season {season}")
            else:
                logging.info(f"No head-to-head games found between Team {team_id_1} and Team {team_id_2} in season {season}")

            time.sleep(1)  # Respect API rate limits
        except Exception as e:
            logging.error(f"Error fetching head-to-head games for season {season}: {e}")

    if head_to_head_games:
        return pd.concat(head_to_head_games, ignore_index=True)
    else:
        return pd.DataFrame()

# -----------------------------
# 4. Initializing Head-to-Head Data
# -----------------------------

def initialize_head_to_head(seasons, head_to_head_data):
    """
    Initializes the head-to-head data for all team pairs.

    Args:
        seasons (list): List of season strings.
        head_to_head_data (dict): Existing head-to-head data.

    Returns:
        dict: Updated head-to-head data.
    """
    # Get unique team IDs
    gamefinder = LeagueGameFinder(season_nullable=seasons, season_type_nullable='Regular Season')
    teams_df = gamefinder.get_data_frames()[1]  # Team stats are usually in the second DataFrame
    team_ids = teams_df['TEAM_ID'].unique()

    total_pairs = len(team_ids) * (len(team_ids) - 1) // 2
    logging.info(f"Total unique team pairs to process: {total_pairs}")

    # Iterate over all unique team pairs
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            team_id_1 = team_ids[i]
            team_id_2 = team_ids[j]
            team_pair = tuple(sorted([team_id_1, team_id_2]))

            if team_pair not in head_to_head_data:
                # Fetch head-to-head games
                games_df = fetch_head_to_head_games(team_id_1, team_id_2, seasons)

                if not games_df.empty:
                    # Sort games by GAME_DATE
                    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
                    games_df = games_df.sort_values(by='GAME_DATE')

                    # Calculate total PTS_QTR1 for each game
                    games_df['Total_PTS_QTR1'] = games_df['PTS_QTR1_team1'] + games_df['PTS_QTR1_team2']

                    # Initialize the rolling window with the last three games
                    recent_games = games_df['Total_PTS_QTR1'].tolist()[-3:]

                    head_to_head_data[team_pair] = recent_games

                    logging.info(f"Initialized head-to-head data for Team Pair {team_pair}: {recent_games}")
                else:
                    head_to_head_data[team_pair] = []
                    logging.info(f"No head-to-head games to initialize for Team Pair {team_pair}")

    return head_to_head_data

# -----------------------------
# 5. Fetching and Processing Game Data
# -----------------------------

def fetch_and_process_games(current_season, previous_seasons):
    """
    Fetches and processes game data for the current and previous seasons.

    Args:
        current_season (str): Current season string, e.g., '2023-24'.
        previous_seasons (list): List of previous season strings.

    Returns:
        pd.DataFrame: Combined DataFrame of all relevant games.
    """
    all_seasons = [current_season] + previous_seasons
    logging.info(f"Fetching games for seasons: {all_seasons}")
    training_games_df = fetch_game_ids(all_seasons)

    if 'GAME_DATE' in training_games_df.columns:
        training_games_df['GAME_DATE'] = pd.to_datetime(training_games_df['GAME_DATE'])
    else:
        logging.error("GAME_DATE column is missing from the fetched games data.")

    return training_games_df

# -----------------------------
# 6. Main Execution Flow
# -----------------------------

def main():
    # Define seasons
    CURRENT_SEASON = '2023-24'        # Replace with the actual current season
    PREVIOUS_SEASONS = ['2022-23', '2021-22']  # Up to two previous seasons

    # Step 1: Fetch and process games for current and previous seasons
    training_games_df = fetch_and_process_games(CURRENT_SEASON, PREVIOUS_SEASONS)

    # Extract unique GAME_IDs
    training_game_ids = training_games_df['GAME_ID'].unique()

    # Load existing training data if available
    all_training_data = load_data(DATA_FILENAME)
    processed_game_ids = set()

    # If data was loaded, extract already processed GAME_IDs
    if all_training_data:
        processed_game_ids = set([data['GAME_ID'] for data in all_training_data])
        logging.info(f"Resuming from existing data. {len(processed_game_ids)} games already processed.")
    else:
        all_training_data = []
        logging.info("Starting data collection from scratch.")

    # Load existing head-to-head data
    head_to_head_data = load_data(HEAD_TO_HEAD_FILENAME)
    if not head_to_head_data:
        head_to_head_data = {}
        logging.info("No existing head-to-head data found. Initializing new head-to-head data.")
        head_to_head_data = initialize_head_to_head(PREVIOUS_SEASONS, head_to_head_data)
        save_data(head_to_head_data, HEAD_TO_HEAD_FILENAME)
    else:
        logging.info("Loaded existing head-to-head data.")

    # Step 2: Fetch and populate training data with Head_to_Head_Q1
    for idx, game_id in enumerate(training_game_ids):
        if game_id in processed_game_ids:
            continue  # Skip already processed games

        game_data = fetch_game_data(game_id)
        if game_data:
            for key, data in game_data.items():
                team_id = data['TEAM_ID']
                opponent_team_id = data['OPPONENT_TEAM_ID']
                # Fetch GAME_DATE from training_games_df
                game_date_series = training_games_df.loc[training_games_df['GAME_ID'] == game_id, 'GAME_DATE']
                if not game_date_series.empty:
                    game_date = game_date_series.iloc[0]
                else:
                    logging.warning(f"GAME_DATE not found for game {game_id}. Skipping Head_to_Head_Q1 calculation.")
                    game_date = None

                if game_date is not None:
                    team_pair = tuple(sorted([team_id, opponent_team_id]))
                    total_pts_q1 = data['PTS_QTR1'] + data['OPP_PTS_QTR1']

                    # Calculate Head_to_Head_Q1
                    recent_games = head_to_head_data.get(team_pair, [])
                    if len(recent_games) >= 3:
                        avg_q1 = sum(recent_games[-3:]) / 3
                    elif len(recent_games) > 0:
                        avg_q1 = sum(recent_games) / len(recent_games)
                    else:
                        avg_q1 = None  # Not enough data

                    # Assign Head_to_Head_Q1 to the data
                    data['Head_to_Head_Q1'] = avg_q1

                    # Update head_to_head_data with the current game
                    recent_games.append(total_pts_q1)
                    if len(recent_games) > 3:
                        recent_games.pop(0)  # Remove the oldest game
                    head_to_head_data[team_pair] = recent_games

                    # Add updated Head_to_Head_Q1 to the game data
                    all_training_data.append(data)
                else:
                    logging.warning(f"No GAME_DATE for game {game_id}. Skipping Head_to_Head_Q1 calculation.")

            processed_game_ids.add(game_id)
        else:
            logging.warning(f"No data for game {game_id}")

        # Save data after every 100 games or at the end
        if (idx + 1) % 100 == 0 or (idx + 1) == len(training_game_ids):
            save_data(all_training_data, DATA_FILENAME)
            logging.info(f"Processed {idx + 1}/{len(training_game_ids)} training games")

        time.sleep(0.6)  # Respect API rate limits

    # Save the final training data
    save_data(all_training_data, DATA_FILENAME)
    logging.info("All training data has been collected and saved.")

    # Create DataFrame from training data
    training_dataset_df = pd.DataFrame(all_training_data)

    # Calculate total first-quarter points
    training_dataset_df['Total_First_Quarter_Points'] = (
        training_dataset_df['PTS_QTR1'] + training_dataset_df['OPP_PTS_QTR1']
    )

    # Extract 'GAME_ID', 'GAME_DATE', and 'MATCHUP' from training_games_df
    game_dates = training_games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_ID']].drop_duplicates()

    # Merge 'GAME_DATE' and 'MATCHUP' into training_dataset_df
    training_dataset_df = training_dataset_df.merge(
        game_dates,
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )

    # Ensure 'GAME_DATE' is in datetime format
    training_dataset_df['GAME_DATE'] = pd.to_datetime(training_dataset_df['GAME_DATE'])

    # Extract 'MONTH' from 'GAME_DATE'
    training_dataset_df['MONTH'] = training_dataset_df['GAME_DATE'].dt.month

    # Step 3: Process head-to-head data (already handled during data collection)

    # Step 4: Calculate additional features
    logging.info("Calculating additional features: Home_Away, Recent_Trend_Team, Recent_Trend_Opponent.")

    # Apply Home_Away
    training_dataset_df['Home_Away'] = training_dataset_df.apply(get_home_away, axis=1)

    # Apply Recent_Trend_Team
    training_dataset_df['Recent_Trend_Team'] = training_dataset_df.apply(
        lambda row: get_recent_trend(row['TEAM_ID'], row['GAME_DATE'], training_dataset_df), axis=1
    )

    # Apply Recent_Trend_Opponent
    training_dataset_df['Recent_Trend_Opponent'] = training_dataset_df.apply(
        lambda row: get_recent_trend(row['OPPONENT_TEAM_ID'], row['GAME_DATE'], training_dataset_df), axis=1
    )

    # Step 5: Persist Head-to-Head Data
    save_data(head_to_head_data, HEAD_TO_HEAD_FILENAME)

    # Step 6: Save Checkpoint with Additional Features
    if os.path.exists(CHECKPOINT_FILENAME):
        training_dataset_df = pd.read_csv(CHECKPOINT_FILENAME, parse_dates=['GAME_DATE'])
        logging.info("Loaded DataFrame from checkpoint file.")
    else:
        # 'Head_to_Head_Q1' has already been calculated during data collection
        # Save the checkpoint with additional features
        try:
            training_dataset_df.to_csv(CHECKPOINT_FILENAME, index=False)
            logging.info(f"Checkpoint saved to '{CHECKPOINT_FILENAME}'.")
        except Exception as e:
            logging.error(f"Error saving checkpoint to '{CHECKPOINT_FILENAME}': {e}")

    # Final DataFrame is ready for analysis or modeling
    logging.info("Dataset preparation is complete.")

    # Output the head of the dataset
    print("Head of the Training Dataset:")
    print(training_dataset_df.head())

    # Save the final dataset as a CSV for AI input
    try:
        training_dataset_df.to_csv(FINAL_CSV_FILENAME, index=False)
        logging.info(f"Final training dataset saved to '{FINAL_CSV_FILENAME}'.")
        print(f"\nFinal training dataset saved to '{FINAL_CSV_FILENAME}'.")
    except Exception as e:
        logging.error(f"Error saving final training dataset to '{FINAL_CSV_FILENAME}': {e}")
        print(f"\nError saving final training dataset to '{FINAL_CSV_FILENAME}': {e}")

if __name__ == "__main__":
    main()

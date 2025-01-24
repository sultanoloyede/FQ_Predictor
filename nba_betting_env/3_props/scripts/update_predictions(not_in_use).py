import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import joblib
import logging
import re  # Import regular expressions
import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    filename='../logs/update_predictions.log',
    level=logging.DEBUG,  # Set to DEBUG to capture all logs
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def fetch_rotowire_page_text():
    """
    Fetches Rotowire NBA lineups page once, returns the entire HTML as lowercase text.
    """
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logging.error(f"Failed to fetch {url} (status: {resp.status_code}).")
            return ""  # Return empty text on error

        page_text = resp.text.lower()  # store everything in lowercase
        return page_text
    except Exception as e:
        logging.error(f"Error fetching rotowire: {e}")
        return ""

def is_player_healthy(player_name, rotowire_text, window_size=1000):
    """
    Returns 1 if the player's name is found AND there's no 'gtd' or 'out' 
    within the next `window_size` characters. Otherwise returns 0.

    - `player_name` should be the exact full name from your CSV (e.g. "Anthony Edwards").
    - `rotowire_text` is the entire HTML from fetch_rotowire_page_text(), lowercased.
    - `window_size` is how many characters after the name we search for 'gtd' or 'out'.
    """
    if not player_name:
        return 0

    full_name = player_name.lower().strip()
    if not full_name:
        return 0

    # Find the index of the player's name in the page text
    idx = rotowire_text.find(full_name)
    if idx == -1:
        # Name not found => 0
        return 0

    # If found, examine the next `window_size` chars
    start_sub = idx + len(full_name)
    end_sub = min(len(rotowire_text), start_sub + window_size)
    near_context = rotowire_text[start_sub:end_sub]

    print()
    print(near_context)
    # If "gtd" or "out" is in near_context => 0, else 1
    if "span" in near_context:
        return 0
    return 1


def load_best_models(models_dir):
    """
    Loads the best scaler and trained models for each target from the models directory.
    Returns a dictionary with target names as keys and their respective scaler and model.
    """
    targets = ['1_plus', '2_plus', '3_plus']
    best_models = {}

    for target in targets:
        # Find all model files for the target
        model_files = [f for f in os.listdir(models_dir) if f.startswith(f'model_{target}') and f.endswith('.pkl')]
        if not model_files:
            logging.error(f"No model files found for target {target} in {models_dir}.")
            continue

        # Initialize variables to track the best model
        best_model_file = None
        highest_roc_auc = -1.0

        # Iterate through model files to find the one with the highest ROC-AUC
        for filename in model_files:
            # Use regex to extract ROC-AUC score
            match = re.search(r'roc_auc_(\d+\.\d+)', filename)
            if match:
                try:
                    roc_auc = float(match.group(1))
                    logging.debug(f"Found ROC-AUC {roc_auc} in filename '{filename}'.")
                    if roc_auc > highest_roc_auc:
                        highest_roc_auc = roc_auc
                        best_model_file = filename
                except ValueError:
                    logging.error(f"Invalid ROC-AUC score in filename '{filename}'. Skipping this file.")
            else:
                logging.error(f"ROC-AUC score not found in filename '{filename}'. Skipping this file.")

        if best_model_file:
            model_path = os.path.join(models_dir, best_model_file)

            # Corresponding scaler filename
            scaler_file = f"scaler_{target}.pkl"
            scaler_path = os.path.join(models_dir, scaler_file)

            if not os.path.exists(scaler_path):
                logging.error(f"Scaler file '{scaler_file}' not found for target {target}.")
                continue

            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                best_models[target] = {
                    'model': model,
                    'scaler': scaler
                }
                logging.info(f"Loaded best model for target '{target}': {best_model_file} with ROC-AUC {highest_roc_auc:.4f}")
            except Exception as e:
                logging.error(f"Error loading model or scaler for target '{target}': {e}")
        else:
            logging.error(f"No valid model files found for target '{target}'.")

    return best_models

def load_player_data(players_dir):
    """
    Loads all player CSVs from the players directory.
    """
    all_data = []
    player_files = [f for f in os.listdir(players_dir) if f.endswith('.csv')]

    for file in player_files:
        try:
            player_id = file.split('_')[1].split('.')[0]
        except IndexError:
            logging.error(f"Filename format incorrect for file '{file}'. Expected format: 'player_<id>.csv'. Skipping.")
            continue

        df = pd.read_csv(os.path.join(players_dir, file))
        df['player_id'] = player_id

        # Standardize column names to lowercase to avoid case sensitivity issues
        df.columns = [col.lower() for col in df.columns]

        # Define required columns
        required_columns = ['game_date', '3points', 'fg3a', 'minutes', 'fouls', 'player_name', 'team_id']

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns {missing_columns} in {file}. Skipping this file.")
            continue  # Skip this file and proceed with others

        # Convert 'game_date' to datetime
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

        # Check for any NaT (Not a Time) values after conversion
        if df['game_date'].isnull().any():
            logging.warning(f"Some 'game_date' entries could not be parsed in {file}. These rows will be dropped.")
            df = df.dropna(subset=['game_date'])

        # Sort by 'game_date'
        df = df.sort_values('game_date').reset_index(drop=True)
        all_data.append(df)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logging.info(f"Loaded data from {len(all_data)} player files. Total games loaded: {len(combined_data)}")
    else:
        combined_data = pd.DataFrame()
        logging.warning("No player data was loaded. Please check the player CSV files.")
    return combined_data

def compute_bottom_20_avg(series):
    """
    Computes the average of the bottom 20% values in a pandas Series.
    If the number of games is less than 5, it calculates based on available games.
    """
    if series.empty:
        return np.nan
    n = max(int(len(series) * 0.2), 1)  # At least one game
    bottom_games = series.nsmallest(n)
    return bottom_games.mean()

def prepare_features(combined_data, target_threshold):
    """
    Prepares the features required for prediction based on each player's latest game.
    Returns a DataFrame with 'player_id' and FEATURE_COLUMNS.
    """
    features = []
    players_processed = 0

    # Define the feature columns directly
    FEATURE_COLUMNS = [
        '3ppg_last_10_games',
        '3ppg_last_20_games',
        'pct_1_plus_last_10_games',
        'pct_2_plus_last_10_games',
        'pct_3_plus_last_10_games',
        'avg_score_bottom_20_percent_games',
        'fg3a_avg_last_10_games',
        'fg3a_avg_season',
        'fg3a_avg_last_5_games',
        'minutes_avg_last_10_games',
        'minutes_avg_season',
        'minutes_avg_last_5_games',
        'fouls_avg_last_10_games',
        'fouls_avg_last_20_games',
        'foul_rate_last_10_games',
        'foul_rate_last_20_games'
    ]

    # Group data by player
    grouped = combined_data.groupby('player_id')

    for player_id, group in grouped:
        group = group.sort_values('game_date').reset_index(drop=True)

        # Check if the player has at least two games to make a prediction
        if len(group) < 2:
            logging.warning(f'Player {player_id} has less than 2 games. Skipping prediction.')
            continue

        # Extract all games except the latest one
        past_games = group.iloc[:-1].reset_index(drop=True)

        # Compute rolling averages based on past_games
        for window in [5, 10, 20]:
            past_games[f'3ppg_last_{window}_games'] = past_games['3points'].rolling(window=window, min_periods=window).mean()
            past_games[f'fg3a_avg_last_{window}_games'] = past_games['fg3a'].rolling(window=window, min_periods=window).mean()
            past_games[f'minutes_avg_last_{window}_games'] = past_games['minutes'].rolling(window=window, min_periods=window).mean()
            past_games[f'fouls_avg_last_{window}_games'] = past_games['fouls'].rolling(window=window, min_periods=window).mean()

        # Compute overall season averages up to the latest game
        past_games['fg3a_avg_season'] = past_games['fg3a'].expanding(min_periods=20).mean()
        past_games['minutes_avg_season'] = past_games['minutes'].expanding(min_periods=20).mean()

        # Compute percentage of games exceeding scoring thresholds in the last 10 games
        past_games['pct_1_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 1).mean())
        past_games['pct_2_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 2).mean())
        past_games['pct_3_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 3).mean())

        # Compute average of bottom 20% scores up to the latest game
        past_games['avg_score_bottom_20_percent_games'] = past_games['3points'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)

        # Compute foul rates
        past_games['foul_rate_last_10_games'] = past_games['fouls'].rolling(window=10, min_periods=10).mean()
        past_games['foul_rate_last_20_games'] = past_games['fouls'].rolling(window=20, min_periods=20).mean()

        # Drop rows with NaN values resulting from rolling calculations
        past_games = past_games.dropna(subset=FEATURE_COLUMNS)

        if past_games.empty:
            logging.warning(f'Player {player_id} has no data after feature calculations. Skipping.')
            continue

        # Use the latest game for prediction
        latest_game = past_games.iloc[-1]

        # Features
        feature_values = latest_game[FEATURE_COLUMNS].values
        features.append([player_id] + list(feature_values))

        players_processed += 1

    if features:
        # Create DataFrame with 'player_id' and features
        columns = ['player_id'] + FEATURE_COLUMNS
        X = pd.DataFrame(features, columns=columns)
        logging.info(f'Prepared features for {players_processed} players for target > {target_threshold} 3points.')
        logging.debug(f'Features DataFrame columns: {X.columns.tolist()}')
        logging.debug(f'Sample features:\n{X.head()}')
    else:
        X = pd.DataFrame()
        logging.warning('No features were prepared. Please check the player data.')

    return X

def make_predictions(X, best_models):
    """
    Makes probability predictions using the loaded best models.
    Returns a DataFrame with predictions and player information, including '3ppg_last_10_games'.
    """
    if X.empty:
        logging.error('Input features DataFrame is empty. Cannot make predictions.')
        return pd.DataFrame()

    # Start with 'player_id' and '3ppg_last_10_games'
    if '3ppg_last_10_games' not in X.columns:
        logging.error("'3ppg_last_10_games' not found in input features. Cannot include it in predictions.")
        return pd.DataFrame()

    predictions = X[['player_id', '3ppg_last_10_games']].copy()

    # Iterate over each target to make predictions
    for target in ['1_plus', '2_plus', '3_plus']:
        if target not in best_models:
            logging.error(f"No model available for target {target}. Skipping predictions for this target.")
            predictions[target] = np.nan
            continue

        model = best_models[target]['model']
        scaler = best_models[target]['scaler']

        # Extract features
        feature_columns = [
            '3ppg_last_10_games',
            '3ppg_last_20_games',
            'pct_1_plus_last_10_games',
            'pct_2_plus_last_10_games',
            'pct_3_plus_last_10_games',
            'avg_score_bottom_20_percent_games',
            'fg3a_avg_last_10_games',
            'fg3a_avg_season',
            'fg3a_avg_last_5_games',
            'minutes_avg_last_10_games',
            'minutes_avg_season',
            'minutes_avg_last_5_games',
            'fouls_avg_last_10_games',
            'fouls_avg_last_20_games',
            'foul_rate_last_10_games',
            'foul_rate_last_20_games'
        ]

        # Verify that all required feature columns are present
        missing_features = [col for col in feature_columns if col not in X.columns]
        if missing_features:
            logging.error(f"Missing feature columns for target '{target}': {missing_features}. Skipping predictions for this target.")
            predictions[target] = np.nan
            continue

        X_features = X[feature_columns]

        # Scale features
        try:
            X_scaled = scaler.transform(X_features)
        except NotFittedError as e:
            logging.error(f"Scaler for target {target} is not fitted: {e}")
            predictions[target] = np.nan
            continue
        except Exception as e:
            logging.error(f"Error during scaling for target {target}: {e}")
            predictions[target] = np.nan
            continue

        # Make predictions
        try:
            proba = model.predict_proba(X_scaled)[:, 1]  # Probability of class '1'
            predictions[target] = proba
            logging.info(f"Predictions made for target '{target}'.")
            logging.debug(f"Sample predictions for target '{target}':\n{proba[:5]}")
        except Exception as e:
            logging.error(f"Error making predictions with model for target '{target}': {e}")
            predictions[target] = np.nan

    logging.debug(f'Predictions DataFrame columns: {predictions.columns.tolist()}')
    logging.debug(f'Sample predictions:\n{predictions.head()}')

    return predictions

def generate_top_scorers(combined_data, predictions, top_n=25):
    """
    Generates top N scorers for each scoring threshold based on predictions.
    Ensures that each player appears only once across all lists, with higher-priority lists taking precedence.
    Returns a dictionary of DataFrames.
    """
    # Extract the latest game for each player
    latest_games = combined_data.sort_values('game_date').groupby('player_id').tail(1).reset_index(drop=True)
    logging.debug(f'Latest games DataFrame columns: {latest_games.columns.tolist()}')
    logging.debug(f'Sample latest games:\n{latest_games.head()}')

    # Merge predictions with latest_games on 'player_id'
    merged = pd.merge(latest_games, predictions, on='player_id', how='left')
    logging.debug(f'Merged DataFrame columns: {merged.columns.tolist()}')
    logging.debug(f'Sample merged data:\n{merged.head()}')

    # Check for any missing values after merge
    missing_info = merged[['player_name', 'team_id']].isnull().any()
    if missing_info.any():
        logging.warning(f"Some player information is missing after merging: {missing_info.to_dict()}")

    top_scorers = {}
    excluded_player_ids = set()  # To keep track of players already assigned to higher-priority lists

    # Define the priority order
    priority_order = ['3_plus', '2_plus', '1_plus']

    # Calculate the total number of unique players needed
    total_needed = top_n * len(priority_order)  # 25 * 3 = 75

    unique_players_available = merged['player_id'].nunique()
    if unique_players_available < total_needed:
        logging.warning(f"Not enough unique players to generate all lists without overlap. Needed: {total_needed}, Available: {unique_players_available}")

    for target in priority_order:
        if target not in merged.columns:
            logging.warning(f"Predictions for target '{target}' not found.")
            continue

        # Ensure '3ppg_last_10_games' exists for sorting
        if '3ppg_last_10_games' not in merged.columns:
            logging.error(f"'3ppg_last_10_games' column not found in merged DataFrame for target '{target}'.")
            continue

        # Filter out excluded players
        target_df = merged[~merged['player_id'].isin(excluded_player_ids)].copy()

        # Sort by predicted probability descending and then by '3ppg_last_10_games' descending
        top = target_df[['player_id', 'player_name', 'team_id', target, '3ppg_last_10_games']].sort_values(
            by=[target, '3ppg_last_10_games'], ascending=[False, False]
        ).head(top_n)

        # Rename the probability column
        top.rename(columns={target: 'predicted_probability'}, inplace=True)

        # Drop '3ppg_last_10_games' as it's no longer needed
        top = top.drop(columns=['3ppg_last_10_games'])

        # Save to dictionary
        top_scorers[target] = top

        logging.info(f"Top {top_n} scorers identified for target '{target}' excluding previously selected players.")
        logging.debug(f"Top scorers for target '{target}':\n{top}")

        # Update the excluded_player_ids set
        excluded_player_ids.update(top['player_id'].tolist())

        logging.debug(f"Excluded player IDs after '{target}' list: {excluded_player_ids}")

    return top_scorers

def save_top_scorers(top_scorers, output_dir):
    """
    Saves the top scorers DataFrames as CSV files in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for target, df in top_scorers.items():
        filename = f"top_25_{target}.csv"
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"Top scorers for '{target}' saved to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving top scorers to {filepath}: {e}")

def verify_high_performers(top_scorers, high_performers, players_dir, best_models):
    """
    Verifies that high-performing players are included in the Top Scorers lists.
    
    Parameters:
    - top_scorers: Dictionary containing Top Scorers DataFrames.
    - high_performers: List of player_ids expected to be in the Top Scorers.
    - players_dir: Directory containing player CSVs.
    - best_models: Dictionary containing best models and scalers for each target.
    
    Returns:
    - None (logs verification results)
    """
    # Define the feature columns directly
    FEATURE_COLUMNS = [
        '3ppg_last_10_games',
        '3ppg_last_20_games',
        'pct_1_plus_last_10_games',
        'pct_2_plus_last_10_games',
        'pct_3_plus_last_10_games',
        'avg_score_bottom_20_percent_games',
        'fg3a_avg_last_10_games',
        'fg3a_avg_season',
        'fg3a_avg_last_5_games',
        'minutes_avg_last_10_games',
        'minutes_avg_season',
        'minutes_avg_last_5_games',
        'fouls_avg_last_10_games',
        'fouls_avg_last_20_games',
        'foul_rate_last_10_games',
        'foul_rate_last_20_games'
    ]

    for player_id in high_performers:
        file_name = f'player_{player_id}.csv'
        file_path = os.path.join(players_dir, file_name)
        
        if not os.path.exists(file_path):
            logging.error(f"Player file '{file_name}' not found for verification.")
            continue
        
        df = pd.read_csv(file_path)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df = df.dropna(subset=['game_date'])
        df = df.sort_values('game_date').reset_index(drop=True)
        
        if len(df) < 2:
            logging.warning(f'Player {player_id} has less than 2 games. Cannot verify prediction.')
            continue
        
        past_games = df.iloc[:-1].reset_index(drop=True)
        
        # Compute rolling features
        for window in [5, 10, 20]:
            past_games[f'3ppg_last_{window}_games'] = past_games['3points'].rolling(window=window, min_periods=window).mean()
            past_games[f'fg3a_avg_last_{window}_games'] = past_games['fg3a'].rolling(window=window, min_periods=window).mean()
            past_games[f'minutes_avg_last_{window}_games'] = past_games['minutes'].rolling(window=window, min_periods=window).mean()
            past_games[f'fouls_avg_last_{window}_games'] = past_games['fouls'].rolling(window=window, min_periods=window).mean()
        
        # Compute overall season averages up to the latest game
        past_games['fg3a_avg_season'] = past_games['fg3a'].expanding(min_periods=20).mean()
        past_games['minutes_avg_season'] = past_games['minutes'].expanding(min_periods=20).mean()

        # Compute percentage of games exceeding scoring thresholds in the last 10 games
        past_games['pct_1_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 1).mean())
        past_games['pct_2_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 2).mean())
        past_games['pct_3_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 3).mean())

        # Compute average of bottom 20% scores up to the latest game
        past_games['avg_score_bottom_20_percent_games'] = past_games['3points'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)

        # Compute foul rates
        past_games['foul_rate_last_10_games'] = past_games['fouls'].rolling(window=10, min_periods=10).mean()
        past_games['foul_rate_last_20_games'] = past_games['fouls'].rolling(window=20, min_periods=20).mean()

        # Drop rows with NaN values resulting from rolling calculations
        past_games = past_games.dropna(subset=FEATURE_COLUMNS)

        if past_games.empty:
            logging.warning(f'Player {player_id} has no data after feature calculations. Cannot verify prediction.')
            continue

        # Use the latest game for prediction
        latest_game = past_games.iloc[-1]

        # Features
        feature_values = latest_game[FEATURE_COLUMNS].values
        feature_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
        
        # Iterate over each target to verify
        for target, threshold in [('3_plus', 3), ('2_plus', 2), ('1_plus', 1)]:  # Adjusted priority for verification
            if target not in best_models:
                logging.error(f"No model available for target '{target}'. Cannot verify player {player_id} for this target.")
                continue

            model = best_models[target]['model']
            scaler = best_models[target]['scaler']

            # Scale features
            try:
                X_scaled = scaler.transform(feature_df)
            except NotFittedError as e:
                logging.error(f"Scaler for target '{target}' is not fitted: {e}")
                continue
            except Exception as e:
                logging.error(f"Error during scaling for target '{target}': {e}")
                continue

            # Make prediction
            try:
                prob = model.predict_proba(X_scaled)[:, 1][0]
            except Exception as e:
                logging.error(f"Error making prediction for player {player_id} on target '{target}': {e}")
                prob = np.nan

            # Check if player is in Top Scorers
            top_df = top_scorers.get(target, pd.DataFrame())
            if top_df.empty:
                logging.warning(f"Top scorers list for '{target}' is empty.")
                continue
            in_top = top_df[top_df['player_id'] == player_id]
            if not in_top.empty:
                logging.info(f"Player ID {player_id} is correctly included in Top {target} scorers with probability {prob:.2f}.")
            else:
                logging.warning(f"Player ID {player_id} is NOT included in Top {target} scorers. Predicted probability: {prob:.2f}.")

def save_top_scorers(top_scorers, output_dir):
    """
    Saves the top scorers DataFrames as CSV files in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for target, df in top_scorers.items():
        filename = f"top_15_{target}.csv"
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"Top scorers for '{target}' saved to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving top scorers to {filepath}: {e}")

def verify_high_performers(top_scorers, high_performers, players_dir, best_models):
    """
    Verifies that high-performing players are included in the Top Scorers lists.
    
    Parameters:
    - top_scorers: Dictionary containing Top Scorers DataFrames.
    - high_performers: List of player_ids expected to be in the Top Scorers.
    - players_dir: Directory containing player CSVs.
    - best_models: Dictionary containing best models and scalers for each target.
    
    Returns:
    - None (logs verification results)
    """
    # Define the feature columns directly
    FEATURE_COLUMNS = [
        '3ppg_last_10_games',
        '3ppg_last_20_games',
        'pct_1_plus_last_10_games',
        'pct_2_plus_last_10_games',
        'pct_3_plus_last_10_games',
        'avg_score_bottom_20_percent_games',
        'fg3a_avg_last_10_games',
        'fg3a_avg_season',
        'fg3a_avg_last_5_games',
        'minutes_avg_last_10_games',
        'minutes_avg_season',
        'minutes_avg_last_5_games',
        'fouls_avg_last_10_games',
        'fouls_avg_last_20_games',
        'foul_rate_last_10_games',
        'foul_rate_last_20_games'
    ]

    for player_id in high_performers:
        file_name = f'player_{player_id}.csv'
        file_path = os.path.join(players_dir, file_name)
        
        if not os.path.exists(file_path):
            logging.error(f"Player file '{file_name}' not found for verification.")
            continue
        
        df = pd.read_csv(file_path)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df = df.dropna(subset=['game_date'])
        df = df.sort_values('game_date').reset_index(drop=True)
        
        if len(df) < 2:
            logging.warning(f'Player {player_id} has less than 2 games. Cannot verify prediction.')
            continue
        
        past_games = df.iloc[:-1].reset_index(drop=True)
        
        # Compute rolling features
        for window in [5, 10, 20]:
            past_games[f'3ppg_last_{window}_games'] = past_games['3points'].rolling(window=window, min_periods=window).mean()
            past_games[f'fg3a_avg_last_{window}_games'] = past_games['fg3a'].rolling(window=window, min_periods=window).mean()
            past_games[f'minutes_avg_last_{window}_games'] = past_games['minutes'].rolling(window=window, min_periods=window).mean()
            past_games[f'fouls_avg_last_{window}_games'] = past_games['fouls'].rolling(window=window, min_periods=window).mean()
        
        # Compute overall season averages up to the latest game
        past_games['fg3a_avg_season'] = past_games['fg3a'].expanding(min_periods=20).mean()
        past_games['minutes_avg_season'] = past_games['minutes'].expanding(min_periods=20).mean()

        # Compute percentage of games exceeding scoring thresholds in the last 10 games
        past_games['pct_1_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 1).mean())
        past_games['pct_2_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 2).mean())
        past_games['pct_3_plus_last_10_games'] = past_games['3points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 3).mean())

        # Compute average of bottom 20% scores up to the latest game
        past_games['avg_score_bottom_20_percent_games'] = past_games['3points'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)

        # Compute foul rates
        past_games['foul_rate_last_10_games'] = past_games['fouls'].rolling(window=10, min_periods=10).mean()
        past_games['foul_rate_last_20_games'] = past_games['fouls'].rolling(window=20, min_periods=20).mean()

        # Drop rows with NaN values resulting from rolling calculations
        past_games = past_games.dropna(subset=FEATURE_COLUMNS)

        if past_games.empty:
            logging.warning(f'Player {player_id} has no data after feature calculations. Cannot verify prediction.')
            continue

        # Use the latest game for prediction
        latest_game = past_games.iloc[-1]

        # Features
        feature_values = latest_game[FEATURE_COLUMNS].values
        feature_df = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
        
        # Iterate over each target to verify
        for target, threshold in [('3_plus', 3), ('2_plus', 2), ('1_plus', 1)]:  # Adjusted priority for verification
            if target not in best_models:
                logging.error(f"No model available for target '{target}'. Cannot verify player {player_id} for this target.")
                continue

            model = best_models[target]['model']
            scaler = best_models[target]['scaler']

            # Scale features
            try:
                X_scaled = scaler.transform(feature_df)
            except NotFittedError as e:
                logging.error(f"Scaler for target '{target}' is not fitted: {e}")
                continue
            except Exception as e:
                logging.error(f"Error during scaling for target '{target}': {e}")
                continue

            # Make prediction
            try:
                prob = model.predict_proba(X_scaled)[:, 1][0]
            except Exception as e:
                logging.error(f"Error making prediction for player {player_id} on target '{target}': {e}")
                prob = np.nan

            # Check if player is in Top Scorers
            top_df = top_scorers.get(target, pd.DataFrame())
            if top_df.empty:
                logging.warning(f"Top scorers list for '{target}' is empty.")
                continue
            in_top = top_df[top_df['player_id'] == player_id]
            if not in_top.empty:
                logging.info(f"Player ID {player_id} is correctly included in Top {target} scorers with probability {prob:.2f}.")
            else:
                logging.warning(f"Player ID {player_id} is NOT included in Top {target} scorers. Predicted probability: {prob:.2f}.")

def main():
    try:
        logging.info('Prediction process started.')

        players_dir = '../data/players/'
        models_dir = '../models/'
        output_dir = '../data/top_scorers/'
        teams_csv_path = '../data/teams.csv'

        # Load player data
        combined_data = load_player_data(players_dir)

        if combined_data.empty:
            logging.error('No player data available. Exiting prediction process.')
            return

        # Load the best models
        best_models = load_best_models(models_dir)

        if not best_models:
            logging.error('No best models loaded. Exiting prediction process.')
            return

        # Define targets and thresholds
        targets = {
            '1_plus': 1,
            '2_plus': 2,
            '3_plus': 3
        }

        # Prepare features for each target and make predictions
        all_predictions = {}
        for target, threshold in targets.items():
            logging.info(f'\n=== Making predictions for target: {target} (> {threshold} 3points) ===')
            X = prepare_features(combined_data, threshold)
            
            # Output the combined features and predictions to a CSV
            final_output_path = os.path.join(output_dir, 'combined_features_predictions.csv')
            os.makedirs(output_dir, exist_ok=True)
            try:
                X.to_csv(final_output_path, index=False)
                logging.info(f"Combined features and predictions saved to {final_output_path}.")
            except Exception as e:
                logging.error(f"Error saving combined features and predictions to {final_output_path}: {e}")

            if X.empty:
                logging.error(f'No features prepared for target {target}. Skipping predictions for this target.')
                continue

            # Make predictions
            predictions = make_predictions(X, {target: best_models[target]})
            all_predictions[target] = predictions[target]

        if not all_predictions:
            logging.error('No predictions were made. Exiting prediction process.')
            return

        # Combine all predictions into a single DataFrame
        # To ensure that all '3ppg_last_10_games' are included, merge on 'player_id'
        # Each target has its own 'predicted_probability', but '3ppg_last_10_games' is shared
        # Ensure '3ppg_last_10_games' is consistent across targets
        # Take '3ppg_last_10_games' from the first target that has it

        # For clarity, ensure '3ppg_last_10_games' is present
        # Take it from the first non-empty prediction
        ppg_column_found = False
        for target in ['3_plus', '2_plus', '1_plus']:
            if target in all_predictions and not all_predictions[target].isnull().all():
                # 'ppg_last_10_games' is already in 'predictions', so no need to take it separately
                ppg_column_found = True
                break

        if not ppg_column_found:
            logging.error("'3ppg_last_10_games' not found in any predictions. Cannot proceed with Top Scorers generation.")
            return

        # Create combined_predictions with 'player_id' and '3ppg_last_10_games'
        # Assuming '3ppg_last_10_games' is the same across all targets
        # Use '3ppg_last_10_games' from the last target that has it
        # Here, we'll assume it's consistent and take it from '3_plus' if available, else '2_plus', else '1_plus'

        combined_predictions = pd.DataFrame()

        for target in ['3_plus', '2_plus', '1_plus']:
            if target in all_predictions and not all_predictions[target].isnull().all():
                # Extract 'player_id' and '3ppg_last_10_games' from predictions
                # Assuming '3ppg_last_10_games' is the same across all targets
                combined_predictions['player_id'] = X['player_id']
                combined_predictions['3ppg_last_10_games'] = X['3ppg_last_10_games']
                break


        # Add predicted probabilities
        for target in ['3_plus', '2_plus', '1_plus']:
            if target in all_predictions:
                combined_predictions[target] = all_predictions[target]
            else:
                combined_predictions[target] = np.nan

        logging.debug(f'Combined Predictions DataFrame columns: {combined_predictions.columns.tolist()}')
        logging.debug(f'Sample Combined Predictions:\n{combined_predictions.head()}')

        # Generate top scorers
        top_scorers = generate_top_scorers(combined_data, combined_predictions, top_n=25)

        if not top_scorers:
            logging.error('No top scorers were generated. Exiting prediction process.')
            return

        # Save top scorers
        save_top_scorers(top_scorers, output_dir)

        # 2) Fetch Rotowire HTML once
        rotowire_text = fetch_rotowire_page_text()
        if not rotowire_text:
            logging.warning("Rotowire text is empty. All players default to 0 (unhealthy).")

        # 3) For each threshold's DataFrame, add a new 'is_playing_today' column
        for threshold, df in top_scorers.items():
            if df.empty:
                continue

            def row_health(row):
                player_name = row['player_name']  # e.g. "Anthony Edwards"
                return is_player_healthy(player_name, rotowire_text, window_size=90)

            df['is_playing_today'] = df.apply(row_health, axis=1)

            # 4) Save updated CSV
            out_path = os.path.join("../data/top_scorers", f"top_25_{threshold}.csv")
            df.to_csv(out_path, index=False)
            logging.info(f"Saved updated top scorers for {threshold} to {out_path}")

        logging.info("Done updating predictions with 'is_playing_today' column.")

        # Define high-performing players to verify (example player IDs)
        high_performers = [
            1628983,  # Shai Gilgeous-Alexander
            1629029,  # Trae Young
            1629025   # Dejounte Murray
            # Add more player IDs as needed
        ]

        # Verify high-performing players are in Top Scorers
        verify_high_performers(top_scorers, high_performers, players_dir, best_models)

        logging.info('Prediction process completed successfully.')

    except Exception as e:
        logging.error(f'An error occurred during prediction: {e}')
        raise

if __name__ == '__main__':
    main()

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

def is_player_healthy(player_name, rotowire_text, window_size=150):
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

# Mapping from team_id to team_name
TEAM_ID_TO_NAME = {
    1610612737: 'Atlanta Hawks',
    1610612738: 'Boston Celtics',
    1610612751: 'Brooklyn Nets',
    1610612766: 'Charlotte Hornets',
    1610612741: 'Chicago Bulls',
    1610612739: 'Cleveland Cavaliers',
    1610612742: 'Dallas Mavericks',
    1610612743: 'Denver Nuggets',
    1610612765: 'Detroit Pistons',
    1610612744: 'Golden State Warriors',
    1610612745: 'Houston Rockets',
    1610612754: 'Indiana Pacers',
    1610612746: 'Los Angeles Clippers',
    1610612747: 'Los Angeles Lakers',
    1610612763: 'Memphis Grizzlies',
    1610612748: 'Miami Heat',
    1610612749: 'Milwaukee Bucks',
    1610612750: 'Minnesota Timberwolves',
    1610612740: 'New Orleans Pelicans',
    1610612752: 'New York Knicks',
    1610612760: 'Oklahoma City Thunder',
    1610612753: 'Orlando Magic',
    1610612755: 'Philadelphia 76ers',
    1610612756: 'Phoenix Suns',
    1610612757: 'Portland Trail Blazers',
    1610612758: 'Sacramento Kings',
    1610612759: 'San Antonio Spurs',
    1610612761: 'Toronto Raptors',
    1610612762: 'Utah Jazz',
    1610612764: 'Washington Wizards',
    # Add more mappings if necessary
}

def load_best_models(models_dir):
    """
    Loads the best scaler and trained models for each target from the models directory.
    Returns a dictionary with target names as keys and their respective scaler and model.
    """
    targets = ['10_plus', '8_plus', '6_plus', '4_plus']  # Higher thresholds first
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
        required_columns = ['game_date', 'rebounds', 'minutes', 'fouls', 'player_name', 'team_id']

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

def prepare_features(combined_data):
    """
    Prepares the features required for prediction based on each player's latest game.
    Returns a DataFrame with 'player_id' and FEATURE_COLUMNS.
    """
    features = []
    players_processed = 0

    # Define the feature columns directly, updated for rebounds
    FEATURE_COLUMNS = [
        'rpg_last_10_games',
        'rpg_last_20_games',
        'pct_4_plus_last_10_games',
        'pct_6_plus_last_10_games',
        'pct_8_plus_last_10_games',
        'pct_10_plus_last_10_games',
        'avg_rebounds_bottom_20_percent_games',
        'rebounds_avg_season',
        'rebounds_avg_last_5_games',
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

        # Skip players with less than 20 games to have meaningful statistics
        if len(group) < 20:
            logging.warning(f'Player {player_id} has less than 20 games. Skipping.')
            continue

        # Compute rolling statistics for rebounds
        group['rpg_last_10_games'] = group['rebounds'].rolling(window=10, min_periods=10).mean()
        group['rpg_last_20_games'] = group['rebounds'].rolling(window=20, min_periods=20).mean()
        group['pct_4_plus_last_10_games'] = group['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 4).mean())
        group['pct_6_plus_last_10_games'] = group['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 6).mean())
        group['pct_8_plus_last_10_games'] = group['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 8).mean())
        group['pct_10_plus_last_10_games'] = group['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 10).mean())
        group['avg_rebounds_bottom_20_percent_games'] = group['rebounds'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)
        group['rebounds_avg_season'] = group['rebounds'].expanding(min_periods=20).mean()
        group['rebounds_avg_last_5_games'] = group['rebounds'].rolling(window=5, min_periods=5).mean()
        group['minutes_avg_last_10_games'] = group['minutes'].rolling(window=10, min_periods=10).mean()
        group['minutes_avg_season'] = group['minutes'].expanding(min_periods=20).mean()
        group['minutes_avg_last_5_games'] = group['minutes'].rolling(window=5, min_periods=5).mean()
        group['fouls_avg_last_10_games'] = group['fouls'].rolling(window=10, min_periods=10).mean()
        group['fouls_avg_last_20_games'] = group['fouls'].rolling(window=20, min_periods=20).mean()
        group['foul_rate_last_10_games'] = group['fouls'].rolling(window=10, min_periods=10).mean()
        group['foul_rate_last_20_games'] = group['fouls'].rolling(window=20, min_periods=20).mean()

        # Drop rows with NaN values resulting from rolling calculations
        group = group.dropna(subset=FEATURE_COLUMNS)

        if group.empty:
            logging.warning(f'Player {player_id} has no data after feature calculations. Skipping.')
            continue

        # Use the latest game for prediction
        latest_game = group.iloc[-1]

        # Features
        feature_values = latest_game[FEATURE_COLUMNS].values
        features.append([player_id] + list(feature_values))

        players_processed += 1

    if features:
        # Create DataFrame with 'player_id' and features
        columns = ['player_id'] + FEATURE_COLUMNS
        X = pd.DataFrame(features, columns=columns)
        logging.info(f'Prepared features for {players_processed} players.')
        logging.debug(f'Features DataFrame columns: {X.columns.tolist()}')
        logging.debug(f'Sample features:\n{X.head()}')
    else:
        X = pd.DataFrame()
        logging.warning('No features were prepared. Please check the player data.')

    return X

def make_predictions(X, best_models):
    """
    Makes probability predictions using the loaded best models.
    Returns a DataFrame with predictions and player information, including all features.
    """
    if X.empty:
        logging.error('Input features DataFrame is empty. Cannot make predictions.')
        return pd.DataFrame()

    # Start with 'player_id'
    predictions = X[['player_id']].copy()

    # Iterate over each target to make predictions
    for target in ['10_plus', '8_plus', '6_plus', '4_plus']:  # Higher thresholds first
        if target not in best_models:
            logging.error(f"No model available for target {target}. Skipping predictions for this target.")
            predictions[target] = np.nan
            continue

        model = best_models[target]['model']
        scaler = best_models[target]['scaler']

        # Define feature columns specific to rebounds
        feature_columns = [
            'rpg_last_10_games',
            'rpg_last_20_games',
            'pct_4_plus_last_10_games',
            'pct_6_plus_last_10_games',
            'pct_8_plus_last_10_games',
            'pct_10_plus_last_10_games',
            'avg_rebounds_bottom_20_percent_games',
            'rebounds_avg_season',
            'rebounds_avg_last_5_games',
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
            logging.error(f"Scaler for target '{target}' is not fitted: {e}")
            predictions[target] = np.nan
            continue
        except Exception as e:
            logging.error(f"Error during scaling for target '{target}': {e}")
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

def generate_top_rebounders(combined_predictions, top_n=10):
    """
    Generates top N rebounders for each rebound threshold based on predictions.
    Ensures that each player appears only once across all lists, with higher-priority lists taking precedence.
    Returns a dictionary of DataFrames.
    """
    merged = combined_predictions.copy()

    logging.debug(f'Merged DataFrame columns: {merged.columns.tolist()}')
    logging.debug(f'Sample merged data:\n{merged.head()}')

    # Check for any missing values after merge
    if set(['player_name', 'team_id']).issubset(merged.columns):
        missing_info = merged[['player_name', 'team_id']].isnull().any()
        if missing_info.any():
            logging.warning(f"Some player information is missing after merging: {missing_info.to_dict()}")
    else:
        logging.error("Columns 'player_name' and/or 'team_id' are missing in the merged DataFrame.")

    top_rebounders = {}
    excluded_player_ids = set()  # To keep track of players already assigned to higher-priority lists

    # Define the priority order
    priority_order = ['10_plus', '8_plus', '6_plus', '4_plus']  # Higher thresholds have higher priority

    # Calculate the total number of unique players needed
    total_needed = top_n * len(priority_order)  # 10 * 4 = 40

    unique_players_available = merged['player_id'].nunique()
    if unique_players_available < total_needed:
        logging.warning(f"Not enough unique players to generate all lists without overlap. Needed: {total_needed}, Available: {unique_players_available}")

    for target in priority_order:
        if target not in merged.columns:
            logging.warning(f"Predictions for target '{target}' not found.")
            continue

        # Ensure 'rpg_last_10_games' exists for sorting
        if 'rpg_last_10_games' not in merged.columns:
            logging.error(f"'rpg_last_10_games' column not found in merged DataFrame for target '{target}'.")
            continue

        # Filter out excluded players
        target_df = merged[~merged['player_id'].isin(excluded_player_ids)].copy()

        # Sort by predicted probability descending and then by 'rpg_last_10_games' descending
        top = target_df[['player_id', 'player_name', 'team_id', target, 'rpg_last_10_games']].sort_values(
            by=[target, 'rpg_last_10_games'], ascending=[False, False]
        ).head(top_n)

        # Rename the probability column
        top.rename(columns={target: 'predicted_probability'}, inplace=True)

        # Drop 'rpg_last_10_games' as it's no longer needed
        top = top.drop(columns=['rpg_last_10_games'])

        # Map 'team_id' to 'team_name'
        top['team_name'] = top['team_id'].map(TEAM_ID_TO_NAME)
        # Handle unmapped team_ids
        top['team_name'].fillna('Unknown', inplace=True)

        # Add 'player_name' and 'team_name' as the first two columns
        cols = ['player_name', 'team_name', 'predicted_probability'] + [col for col in top.columns if col not in ['player_name', 'team_name', 'predicted_probability']]
        top = top[cols]

        # Save to dictionary
        top_rebounders[target] = top

        logging.info(f"Top {top_n} rebounders identified for target '{target}' excluding previously selected players.")
        logging.debug(f"Top rebounders for target '{target}':\n{top}")

        # Update the excluded_player_ids set
        excluded_player_ids.update(top['player_id'].tolist())

        logging.debug(f"Excluded player IDs after '{target}' list: {excluded_player_ids}")

    return top_rebounders

def save_top_rebounders(top_rebounders, output_dir):
    """
    Saves the top rebounders DataFrames as CSV files in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for target, df in top_rebounders.items():
        filename = f"top_10_{target}.csv"  # Changed from top_25 to top_10
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"Top rebounders for '{target}' saved to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving top rebounders to {filepath}: {e}")

def verify_high_performers(top_rebounders, high_performers, players_dir, best_models):
    """
    Verifies that high-performing players are included in the Top Rebounders lists.
    
    Parameters:
    - top_rebounders: Dictionary containing Top Rebounders DataFrames.
    - high_performers: List of player_ids expected to be in the Top Rebounders.
    - players_dir: Directory containing player CSVs.
    - best_models: Dictionary containing best models and scalers for each target.
    
    Returns:
    - None (logs verification results)
    """
    # Define the feature columns directly
    FEATURE_COLUMNS = [
        'rpg_last_10_games',
        'rpg_last_20_games',
        'pct_4_plus_last_10_games',
        'pct_6_plus_last_10_games',
        'pct_8_plus_last_10_games',
        'pct_10_plus_last_10_games',
        'avg_rebounds_bottom_20_percent_games',
        'rebounds_avg_season',
        'rebounds_avg_last_5_games',
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
        
        if len(df) < 20:
            logging.warning(f'Player {player_id} has less than 20 games. Cannot verify prediction.')
            continue
        
        past_games = df.iloc[:-1].reset_index(drop=True)
        
        # Compute rolling features for rebounds
        for window in [5, 10, 20]:
            past_games[f'rpg_last_{window}_games'] = past_games['rebounds'].rolling(window=window, min_periods=window).mean()
            past_games[f'rebounds_avg_last_{window}_games'] = past_games['rebounds'].rolling(window=window, min_periods=window).mean()
            past_games[f'minutes_avg_last_{window}_games'] = past_games['minutes'].rolling(window=window, min_periods=window).mean()
            past_games[f'fouls_avg_last_{window}_games'] = past_games['fouls'].rolling(window=window, min_periods=window).mean()
        
        # Compute overall season averages up to the latest game
        past_games['rebounds_avg_season'] = past_games['rebounds'].expanding(min_periods=20).mean()
        past_games['minutes_avg_season'] = past_games['minutes'].expanding(min_periods=20).mean()

        # Compute percentage of games exceeding rebound thresholds in the last 10 games
        past_games['pct_4_plus_last_10_games'] = past_games['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 4).mean())
        past_games['pct_6_plus_last_10_games'] = past_games['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 6).mean())
        past_games['pct_8_plus_last_10_games'] = past_games['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 8).mean())
        past_games['pct_10_plus_last_10_games'] = past_games['rebounds'].rolling(window=10, min_periods=10).apply(lambda x: (x > 10).mean())

        # Compute average of bottom 20% rebounds up to the latest game
        past_games['avg_rebounds_bottom_20_percent_games'] = past_games['rebounds'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)

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
        for target, threshold in [('10_plus', 10), ('8_plus', 8), ('6_plus', 6), ('4_plus', 4)]:  # Adjusted priority for verification
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

            # Check if player is in Top Rebounders
            top_df = top_rebounders.get(target, pd.DataFrame())
            if top_df.empty:
                logging.warning(f"Top rebounders list for '{target}' is empty.")
                continue
            in_top = top_df[top_df['player_id'] == player_id]
            if not in_top.empty:
                player_name = in_top.iloc[0]['player_name'] if 'player_name' in in_top.columns else 'Unknown'
                team_name = in_top.iloc[0]['team_name'] if 'team_name' in in_top.columns else 'Unknown'
                logging.info(f"Player ID {player_id} ({player_name}, {team_name}) is correctly included in Top {target} rebounders with probability {prob:.2f}.")
            else:
                logging.warning(f"Player ID {player_id} is NOT included in Top {target} rebounders. Predicted probability: {prob:.2f}.")

def main():
    try:
        logging.info('Prediction process started.')

        players_dir = '../data/players/'
        models_dir = '../models/'
        output_dir = '../data/top_rebounders/'

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

        # Prepare features once
        X = prepare_features(combined_data)
        if X.empty:
            logging.error('No features prepared. Exiting prediction process.')
            return

        # Make predictions for all targets
        predictions = make_predictions(X, best_models)
        if predictions.empty:
            logging.error('No predictions were made. Exiting prediction process.')
            return

        # Combine all predictions into a single DataFrame
        # This includes 'player_id' and all predicted probabilities
        combined_predictions = pd.merge(X, predictions, on='player_id', how='left')

        # Merge with player names and team IDs for better interpretability
        latest_games = combined_data.sort_values('game_date').groupby('player_id').tail(1).reset_index(drop=True)
        combined_predictions = pd.merge(combined_predictions, latest_games[['player_id', 'player_name', 'team_id']], on='player_id', how='left')

        # Map 'team_id' to 'team_name'
        combined_predictions['team_name'] = combined_predictions['team_id'].map(TEAM_ID_TO_NAME)
        # Handle unmapped team_ids
        combined_predictions['team_name'].fillna('Unknown', inplace=True)

        # Output the combined features and predictions to a CSV
        final_output_path = os.path.join(output_dir, 'combined_features_predictions.csv')
        os.makedirs(output_dir, exist_ok=True)
        try:
            combined_predictions.to_csv(final_output_path, index=False)
            logging.info(f"Combined features and predictions saved to {final_output_path}.")
        except Exception as e:
            logging.error(f"Error saving combined features and predictions to {final_output_path}: {e}")

        # Generate top rebounders with top_n=10
        top_rebounders = generate_top_rebounders(combined_predictions, top_n=10)  # Changed from 25 to 10

        if not top_rebounders:
            logging.error('No top rebounders were generated. Exiting prediction process.')
            return

        # Save top rebounders
        save_top_rebounders(top_rebounders, output_dir)

        # 2) Fetch Rotowire HTML once
        rotowire_text = fetch_rotowire_page_text()
        if not rotowire_text:
            logging.warning("Rotowire text is empty. All players default to 0 (unhealthy).")

        # 3) For each threshold's DataFrame, add a new 'is_playing_today' column
        for threshold, df in top_rebounders.items():
            if df.empty:
                continue

            def row_health(row):
                player_name = row['player_name']  # e.g. "Anthony Edwards"
                return is_player_healthy(player_name, rotowire_text, window_size=90)

            df['is_playing_today'] = df.apply(row_health, axis=1)

            # 4) Save updated CSV
            out_path = os.path.join("../data/top_rebounders", f"top_10_{threshold}.csv")
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

        # Verify high-performing players are in Top Rebounders
        verify_high_performers(top_rebounders, high_performers, players_dir, best_models)

        logging.info('Prediction process completed successfully.')

    except Exception as e:
        logging.error(f'An error occurred during prediction: {e}')
        raise

if __name__ == '__main__':
    main()

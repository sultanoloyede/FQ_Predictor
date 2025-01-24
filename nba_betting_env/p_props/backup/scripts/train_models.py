import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import logging
from scipy.stats import randint, uniform

# Setup logging
logging.basicConfig(
    filename='../logs/train_models.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

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
        required_columns = ['game_date', 'points', 'fga', 'minutes', 'fouls', 'player_name', 'team_id']

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
    Prepares the features required for training based on each player's game history.
    Returns feature matrix X and target vector y.
    """
    features = []
    targets = []
    players_processed = 0

    # Define the feature columns directly
    FEATURE_COLUMNS = [
        'ppg_last_10_games',
        'ppg_last_20_games',
        'pct_10_plus_last_10_games',
        'pct_15_plus_last_10_games',
        'pct_20_plus_last_10_games',
        'avg_score_bottom_20_percent_games',
        'fga_avg_last_10_games',
        'fga_avg_season',
        'fga_avg_last_5_games',
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

        # Compute rolling statistics
        group['ppg_last_10_games'] = group['points'].rolling(window=10, min_periods=10).mean()
        group['ppg_last_20_games'] = group['points'].rolling(window=20, min_periods=20).mean()
        group['pct_10_plus_last_10_games'] = group['points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 10).mean())
        group['pct_15_plus_last_10_games'] = group['points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 15).mean())
        group['pct_20_plus_last_10_games'] = group['points'].rolling(window=10, min_periods=10).apply(lambda x: (x > 20).mean())
        group['avg_score_bottom_20_percent_games'] = group['points'].rolling(window=20, min_periods=20).apply(compute_bottom_20_avg)
        group['fga_avg_last_10_games'] = group['fga'].rolling(window=10, min_periods=10).mean()
        group['fga_avg_season'] = group['fga'].expanding(min_periods=20).mean()
        group['fga_avg_last_5_games'] = group['fga'].rolling(window=5, min_periods=5).mean()
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

        # Use the latest game for training
        latest_game = group.iloc[-1]

        # Features
        feature_values = latest_game[FEATURE_COLUMNS].values
        features.append(feature_values)

        # Target: Whether the player scored above the target threshold in the latest game
        target = 1 if latest_game['points'] > target_threshold else 0
        targets.append(target)

        players_processed += 1

    if features:
        X = np.array(features)
        y = np.array(targets)
        logging.info(f'Prepared features and targets for {players_processed} players for target > {target_threshold} points.')
    else:
        X = np.array([])
        y = np.array([])
        logging.warning('No features or targets were prepared. Check the player data and feature calculations.')

    return X, y

def get_model_search_space(model_name):
    """
    Defines the hyperparameter search space for different models.
    """
    if model_name == 'XGBoost':
        return {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 5),
            'min_child_weight': randint(1, 10),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
    elif model_name == 'GradientBoosting':
        return {
            'n_estimators': randint(100, 1000),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'subsample': uniform(0.5, 0.5),
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model_name == 'LogisticRegression':
        return {
            'C': uniform(0.01, 10),
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        }
    elif model_name == 'SVC':
        return {
            'C': uniform(0.1, 10),
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    elif model_name == 'KNeighbors':
        return {
            'n_neighbors': randint(3, 30),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    else:
        return {}

def train_and_tune_model(X, y, model_name):
    """
    Trains and tunes a model using RandomizedSearchCV.
    Returns the best estimator and its ROC-AUC score.
    """
    if model_name == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == 'SVC':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'KNeighbors':
        model = KNeighborsClassifier()
    else:
        logging.error(f"Unsupported model: {model_name}")
        return None, 0

    param_distributions = get_model_search_space(model_name)

    if not param_distributions:
        logging.error(f"No hyperparameter space defined for model: {model_name}")
        return None, 0

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define ROC-AUC scorer
    scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=50,
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV
    random_search.fit(X, y)

    best_model = random_search.best_estimator_
    best_score = random_search.best_score_

    logging.info(f"Best {model_name} ROC-AUC: {best_score:.4f}")
    logging.info(f"Best {model_name} Parameters: {random_search.best_params_}")

    return best_model, best_score

def main():
    try:
        logging.info('Training process started.')

        players_dir = '../data/players/'
        models_dir = '../models/'

        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Load and prepare data
        combined_data = load_player_data(players_dir)
        if combined_data.empty:
            logging.error('No player data available. Exiting training process.')
            return

        # Define targets and thresholds
        targets = {
            '10_plus': 10,
            '15_plus': 15,
            '20_plus': 20
        }

        # Define models to evaluate
        model_names = ['XGBoost', 'GradientBoosting', 'LogisticRegression', 'SVC', 'KNeighbors']

        # Dictionary to store best models and their scores
        best_models = {}

        for target_name, threshold in targets.items():
            logging.info(f'\n=== Training for Target: {target_name} (> {threshold} points) ===')

            # Prepare features and target
            X, y = prepare_features(combined_data, threshold)

            if X.size == 0:
                logging.error(f'No data available for target {target_name}. Skipping.')
                continue

            # Initialize variables to track the best model
            overall_best_model = None
            overall_best_score = 0
            overall_best_model_name = ''

            # Evaluate each model
            for model_name in model_names:
                logging.info(f'\n--- Evaluating Model: {model_name} ---')
                model, score = train_and_tune_model(X, y, model_name)

                if model is not None and score > overall_best_score:
                    overall_best_model = model
                    overall_best_score = score
                    overall_best_model_name = model_name

            if overall_best_model is not None:
                # Fit scaler on the entire dataset
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Retrain the best model on the entire dataset
                overall_best_model.fit(X_scaled, y)
                logging.info(f'{overall_best_model_name} model retrained on the entire dataset.')

                # Save the scaler and the best model
                scaler_filename = f"scaler_{target_name}.pkl"
                scaler_path = os.path.join(models_dir, scaler_filename)
                joblib.dump(scaler, scaler_path)
                logging.info(f"Scaler saved to {scaler_path}.")

                # Ensure ROC-AUC score is rounded to 4 decimal places
                model_filename = f"model_{target_name}_{overall_best_model_name}_roc_auc_{overall_best_score:.4f}.pkl"
                model_path = os.path.join(models_dir, model_filename)
                joblib.dump(overall_best_model, model_path)
                logging.info(f"Best model ({overall_best_model_name}) saved to {model_path}.")

                # Store the best model information
                best_models[target_name] = {
                    'model': overall_best_model,
                    'scaler': scaler,
                    'roc_auc': overall_best_score,
                    'model_name': overall_best_model_name,
                    'model_path': model_path,
                    'scaler_path': scaler_path
                }
            else:
                logging.error(f'No valid model was trained for target {target_name}.')

        logging.info('\n=== Training process completed successfully ===')

    except Exception as e:
        logging.error(f'An error occurred during training: {e}')
        raise

if __name__ == '__main__':
    main()

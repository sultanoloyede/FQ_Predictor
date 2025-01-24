import os
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

#############################
# 1) HELPER FUNCTIONS
#############################

def determine_season(game_date: pd.Timestamp) -> str:
    """
    Returns a string like '2023-24' for an NBA season.
    """
    year = game_date.year
    month = game_date.month
    
    if month >= 10:  # Oct..Dec => year-year+1
        season_start = year
        season_end = year + 1
    else:
        # Jan..June => belongs to prior year
        season_start = year - 1
        season_end = year
    return f"{season_start}-{str(season_end)[-2:]}"

def load_and_consolidate_game_logs(players_dir: str, output_path: str):
    """
    Loads all player CSVs in `players_dir`, merges into one DataFrame,
    saves to `output_path`.
    """
    logging.info("Starting consolidation of player game logs.")
    all_dfs = []

    for file_name in os.listdir(players_dir):
        if not file_name.endswith('.csv'):
            continue
        
        file_path = os.path.join(players_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            df.columns = [col.lower() for col in df.columns]
            logging.debug(f"Loaded {len(df)} rows from {file_name}.")
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file_name}: {e}")

    if not all_dfs:
        logging.warning("No valid player data found. The consolidated DataFrame will be empty.")
        consolidated_df = pd.DataFrame()
    else:
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"Successfully concatenated {len(all_dfs)} files. "
                     f"Total rows: {len(consolidated_df)}")

    try:
        consolidated_df.to_csv(output_path, index=False)
        logging.info(f"Consolidated logs saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving consolidated logs: {e}")

    return consolidated_df

def compute_best_worst_avg(series, window_size, best_count, worst_count):
    """Compute mean of top `best_count` games & bottom `worst_count` games in a series."""
    sorted_desc = series.nlargest(best_count)
    sorted_asc = series.nsmallest(worst_count)
    best_avg = sorted_desc.mean() if not sorted_desc.empty else np.nan
    worst_avg = sorted_asc.mean() if not sorted_asc.empty else np.nan
    return best_avg, worst_avg

#############################
# 2) APPROACH B FEATURE ENGINEERING
#############################

def create_feature_engineered_dataset_approach_b(df, min_games_for_season_avg=1):
    """
    Approach B:
      - Each row is for game i, but label is derived from game i+1.
      - Features use data up to & including game i (no future data).
      - If game i+1 doesn't exist (the last game of the season), we skip row i.

    We track label_1_plus, label_2_plus, label_3_plus based on the *next* game (i+1).
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df.dropna(subset=["game_date"], inplace=True)

    # If no season column, add it
    if "season" not in df.columns:
        df["season"] = df["game_date"].apply(determine_season)

    # Sort by player, season, date
    df = df.sort_values(["player_id", "season", "game_date"]).reset_index(drop=True)

    engineered_rows = []

    grouped = df.groupby(["player_id", "season"], sort=False)
    for (player_id, season), group in grouped:
        group = group.sort_values("game_date").reset_index(drop=True)

        # We'll only iterate to the *second to last* row (i.e., up to len(group)-2)
        # because row i's label depends on row i+1
        for i in range(len(group) - 1):
            current_game = group.iloc[i]      # game i
            next_game = group.iloc[i + 1]     # game i+1

            # All games up to (and including) i => prev_games
            # So rolling stats can include the current_game's own stats for approach B
            # (some do it up to i-1, but typically approach B includes i in feature set).
            # It's up to you—some prefer features up to i-1. We'll do up to i for maximum info.
            prev_games = group.iloc[: i+1]

            # Rolling stats from prev_games
            # current_game_3points = current_game["3points"] is only used for computations if we want
            # but it's not the label. The label is next_game.

            # Next game 3points => used for labels
            next_3points = next_game["3points"]

            # ============ Rolling Stats ============

            # pg3 last 5, 10, 20
            pg3_5 = (prev_games["3points"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["3points"].mean())
            pg3_10 = (prev_games["3points"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["3points"].mean())
            pg3_20 = (prev_games["3points"].rolling(20).mean().iloc[-1]
                      if len(prev_games) >= 20 else prev_games["3points"].mean())

            # fg3a last 5, 10
            fg3a_5 = (prev_games["fg3a"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["fg3a"].mean())
            fg3a_10 = (prev_games["fg3a"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["fg3a"].mean())

            # minutes last 5, 10
            min_5 = (prev_games["minutes"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["minutes"].mean())
            min_10 = (prev_games["minutes"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["minutes"].mean())

            # Season averages
            if len(prev_games) >= min_games_for_season_avg:
                pg3_season = prev_games["3points"].mean()
                fg3a_season = prev_games["fg3a"].mean()
                min_season = prev_games["minutes"].mean()
            else:
                pg3_season = np.nan
                fg3a_season = np.nan
                min_season = np.nan

            # Best/Worst slices last 10
            window_10 = (prev_games.iloc[-10:]["3points"]
                         if len(prev_games) >= 10 else prev_games["3points"])
            best_2_avg_10, worst_2_avg_10 = compute_best_worst_avg(window_10, 10, 2, 2)

            # Best/Worst slices last 20
            window_20 = (prev_games.iloc[-20:]["3points"]
                         if len(prev_games) >= 20 else prev_games["3points"])
            best_4_avg_20, worst_4_avg_20 = compute_best_worst_avg(window_20, 20, 4, 4)

            # Pct clearing thresholds in last 10
            last_10_slice = (prev_games.iloc[-10:]
                             if len(prev_games) >= 10 else prev_games)
            pct_1_plus = (last_10_slice["3points"] >= 1).mean()
            pct_2_plus = (last_10_slice["3points"] >= 2).mean()
            pct_3_plus = (last_10_slice["3points"] >= 3).mean()

            # Worst 20% of the season so far
            season_3points = prev_games["3points"]
            n_20pct = max(int(len(season_3points) * 0.2), 1)
            worst_20pct_season = season_3points.nsmallest(n_20pct).mean()

            # Deviation from rolling means (using current_game's 3points)
            current_3points = current_game["3points"]
            # In Approach B, this is still “historic” by the time we predict game i+1.
            points3_deviation_5 = current_3points - (pg3_5 if pg3_5 is not np.nan else 0)
            points3_deviation_10 = current_3points - (pg3_10 if pg3_10 is not np.nan else 0)

            # Average 3points deviation in season
            if not np.isnan(pg3_season):
                avg_3points_dev_season = prev_games["3points"].apply(lambda x: x - pg3_season).mean()
            else:
                avg_3points_dev_season = np.nan

            # Did they clear X in the *last* game?
            # That means game (i-1), if i >= 1
            if i > 0:
                last_game_3points = group.iloc[i - 1]["3points"]
                cleared_1_last_game = 1 if last_game_3points >= 1 else 0
                cleared_2_last_game = 1 if last_game_3points >= 2 else 0
                cleared_3_last_game = 1 if last_game_3points >= 3 else 0
            else:
                cleared_1_last_game = np.nan
                cleared_2_last_game = np.nan
                cleared_3_last_game = np.nan

            # ============ Label from Next Game (i+1) ============
            label_1_plus = 1 if next_3points >= 1 else 0
            label_2_plus = 1 if next_3points >= 2 else 0
            label_3_plus = 1 if next_3points >= 3 else 0

            row_dict = {
                "player_id": player_id,
                "player_name": current_game.get("player_name", None),
                "team_id": current_game.get("team_id", None),
                "season": season,
                "game_date": current_game["game_date"],
                "game_id": current_game.get("game_id", None),

                # Features reflect up to & including game i
                "3points_current_game": current_3points,   # *optional* if you want i’s 3points
                "pg3_last_5": pg3_5,
                "pg3_last_10": pg3_10,
                "pg3_last_20": pg3_20,
                "pg3_season": pg3_season,
                "best_2_avg_last_10": best_2_avg_10,
                "worst_2_avg_last_10": worst_2_avg_10,
                "best_4_avg_last_20": best_4_avg_20,
                "worst_4_avg_last_20": worst_4_avg_20,
                "pct_1_plus_last_10": pct_1_plus,
                "pct_2_plus_last_10": pct_2_plus,
                "pct_3_plus_last_10": pct_3_plus,
                "worst_20pct_season": worst_20pct_season,
                "3points_deviation_from_5": points3_deviation_5,
                "3points_deviation_from_10": points3_deviation_10,
                "avg_3points_deviation_season": avg_3points_dev_season,
                "fg3a_last_5": fg3a_5,
                "fg3a_last_10": fg3a_10,
                "fg3a_season": fg3a_season,
                "minutes_last_5": min_5,
                "minutes_last_10": min_10,
                "minutes_season": min_season,
                "cleared_1_in_last_game": cleared_1_last_game,
                "cleared_2_in_last_game": cleared_2_last_game,
                "cleared_3_in_last_game": cleared_3_last_game,

                # Labels are from game i+1
                "label_1_plus": label_1_plus,
                "label_2_plus": label_2_plus,
                "label_3_plus": label_3_plus
            }

            engineered_rows.append(row_dict)

    # Return the new Approach B DataFrame
    return pd.DataFrame(engineered_rows)

#############################
# 3) TRAIN MODELS (Same as Before)
#############################

def train_and_evaluate_models(X, y, threshold_label, do_oversampling=True):
    """
    Splits data, optionally applies SMOTE, trains multiple classifiers, returns best model by .score().
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"[{threshold_label}] Train set: {X_train.shape}, Test set: {X_test.shape}")
    logging.info(f"[{threshold_label}] Class distribution in TRAIN: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
    logging.info(f"[{threshold_label}] Class distribution in TEST: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")

    if do_oversampling:
        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            logging.info(f"[{threshold_label}] After SMOTE => Train shape: {X_train.shape}")
            logging.info(f"[{threshold_label}] Class distribution after SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
        except Exception as e:
            logging.error(f"[{threshold_label}] SMOTE oversampling error: {e}")

    # Candidate models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "KNeighbors": KNeighborsClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    best_model = None
    best_score = -np.inf

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            logging.info(f"[{threshold_label}] {model_name} -> Test Accuracy: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = (model_name, model)
        except Exception as e:
            logging.error(f"[{threshold_label}] Error training {model_name}: {e}")

    return best_model, best_score


#############################
# 4) MAIN SCRIPT EXAMPLE
#############################

if __name__ == "__main__":
    logging.basicConfig(
        filename='../logs/train_models.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

    # A) Consolidate
    players_directory = "../data/players"
    consolidated_csv_path = "../data/consolidated_database.csv"
    consolidated_data = load_and_consolidate_game_logs(players_directory, consolidated_csv_path)

    if consolidated_data.empty:
        logging.error("No data found. Exiting.")
        exit(1)

    # B) Create Approach B dataset
    if "game_date" in consolidated_data.columns:
        consolidated_data["game_date"] = pd.to_datetime(consolidated_data["game_date"], errors="coerce")
        consolidated_data.dropna(subset=["game_date"], inplace=True)
        consolidated_data["season"] = consolidated_data["game_date"].apply(determine_season)

    dataset_for_training = create_feature_engineered_dataset_approach_b(consolidated_data)
    dataset_out_path = "../data/dataset_for_training_approachB.csv"
    dataset_for_training.to_csv(dataset_out_path, index=False)
    logging.info(f"Approach B dataset saved to {dataset_out_path}")

    # C) Check class distribution
    for col in ["label_1_plus", "label_2_plus", "label_3_plus"]:
        vc = dataset_for_training[col].value_counts()
        logging.info(f"[Approach B: {col}] => {vc.to_dict()}")

    # D) Train each threshold
    feature_cols = [
        # Exclude label_XX, but include your rolling stats, etc.
        "pg3_last_5", "pg3_last_10", "pg3_last_20",
        "pg3_season",
        "best_2_avg_last_10", "worst_2_avg_last_10",
        "best_4_avg_last_20", "worst_4_avg_last_20",
        "pct_1_plus_last_10", "pct_2_plus_last_10", "pct_3_plus_last_10",
        "worst_20pct_season",
        "3points_deviation_from_5", "3points_deviation_from_10",
        "avg_3points_deviation_season",
        "fg3a_last_5", "fg3a_last_10", "fg3a_season",
        "minutes_last_5", "minutes_last_10", "minutes_season",
        "cleared_1_in_last_game", "cleared_2_in_last_game", "cleared_3_in_last_game"
        # "3points_current_game" -> optional if you consider that "known" at time i (some do, some don't).
    ]

    # Drop rows with missing
    df_b = dataset_for_training.dropna(subset=feature_cols).reset_index(drop=True)

    # Loop thresholds
    thresholds = {
        "1_plus": "label_1_plus",
        "2_plus": "label_2_plus",
        "3_plus": "label_3_plus"
    }

    X = df_b[feature_cols]
    os.makedirs("../models", exist_ok=True)

    for tname, label_col in thresholds.items():
        y = df_b[label_col]

        # Decide SMOTE usage
        if tname in ["1_plus", "2_plus"]:
            do_oversampling = True
        else:
            do_oversampling = False

        logging.info(f"=== Training Approach B for threshold: {tname} ===")
        best_model_info, best_score = train_and_evaluate_models(X, y, tname, do_oversampling=do_oversampling)

        if best_model_info is None:
            logging.error(f"No successful model found for {tname} (Approach B).")
            continue

        best_model_name, best_model = best_model_info
        logging.info(f"[{tname} - Approach B] Best Model: {best_model_name}, Score: {best_score:.4f}")

        model_path = f"../models/model_{tname}_approachB_{best_model_name}_acc_{best_score:.4f}.pkl"
        joblib.dump(best_model, model_path)
        logging.info(f"Saved best Approach B model for {tname} to {model_path}")
    
    logging.info("Approach B training complete.")

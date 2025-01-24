import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

# NEW: for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# NEW: Import LightGBM
from lightgbm import LGBMClassifier  # <--- Added import for LightGBM

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
    removes players who haven't played in the last 8 days, saves to `output_path`,
    and returns the consolidated DataFrame.
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
        logging.info(f"Successfully concatenated {len(all_dfs)} files. Total rows: {len(consolidated_df)}")

        # Ensure 'game_date' is datetime
        if 'game_date' in consolidated_df.columns:
            consolidated_df['game_date'] = pd.to_datetime(consolidated_df['game_date'], errors='coerce')

            # Determine cutoff date for last 8 days
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=8)

            # Identify players who have played in the last 8 days
            recent_players = set(consolidated_df.loc[
                consolidated_df['game_date'] >= cutoff_date, 'player_id'
            ].unique())

            # Filter to keep only logs for players active in the last 8 days
            before_filtering = len(consolidated_df)
            consolidated_df = consolidated_df[consolidated_df['player_id'].isin(recent_players)]
            after_filtering = len(consolidated_df)

            logging.info(f"Filtered out players inactive in the last 8 days. "
                         f"Rows before: {before_filtering}, after: {after_filtering}.")
        else:
            logging.warning("'game_date' column not found in consolidated data; skipping activity filtering.")

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
        for i in range(len(group) - 1):
            current_game = group.iloc[i]
            next_game = group.iloc[i + 1]

            # Rolling stats up to & including game i
            prev_games = group.iloc[: i+1]

            next_points = next_game["points"]
    
            # ============ Rolling Stats ============
            ppg_5 = (prev_games["points"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["points"].mean())
            ppg_10 = (prev_games["points"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["points"].mean())
            ppg_20 = (prev_games["points"].rolling(20).mean().iloc[-1]
                      if len(prev_games) >= 20 else prev_games["points"].mean())

            diff = ppg_5 - ppg_10
            trend_5_10 = diff

            fga_5 = (prev_games["fga"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["fga"].mean())
            fga_10 = (prev_games["fga"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["fga"].mean())

            min_5 = (prev_games["minutes"].rolling(5).mean().iloc[-1]
                     if len(prev_games) >= 5 else prev_games["minutes"].mean())
            min_10 = (prev_games["minutes"].rolling(10).mean().iloc[-1]
                      if len(prev_games) >= 10 else prev_games["minutes"].mean())

            if len(prev_games) >= min_games_for_season_avg:
                ppg_season = prev_games["points"].mean()
                fga_season = prev_games["fga"].mean()
                min_season = prev_games["minutes"].mean()
            else:
                ppg_season = np.nan
                fga_season = np.nan
                min_season = np.nan

            window_10 = (prev_games.iloc[-10:]["points"]
                         if len(prev_games) >= 10 else prev_games["points"])
            best_2_avg_10, worst_2_avg_10 = compute_best_worst_avg(window_10, 10, 2, 2)

            window_20 = (prev_games.iloc[-20:]["points"]
                         if len(prev_games) >= 20 else prev_games["points"])
            best_4_avg_20, worst_4_avg_20 = compute_best_worst_avg(window_20, 20, 4, 4)

            last_5_slice = (prev_games.iloc[-5:]
                             if len(prev_games) >= 5 else prev_games)
            pct_10_plus_l5 = (last_5_slice["points"] >= 10).mean()
            pct_15_plus_l5 = (last_5_slice["points"] >= 15).mean()
            pct_20_plus_l5 = (last_5_slice["points"] >= 20).mean()

            last_10_slice = (prev_games.iloc[-10:]
                             if len(prev_games) >= 10 else prev_games)
            pct_10_plus_l10 = (last_10_slice["points"] >= 10).mean()
            pct_15_plus_l10 = (last_10_slice["points"] >= 15).mean()
            pct_20_plus_l10 = (last_10_slice["points"] >= 20).mean()

            last_20_slice = (prev_games.iloc[-20:]
                             if len(prev_games) >= 20 else prev_games)
            pct_10_plus_l20 = (last_20_slice["points"] >= 10).mean()
            pct_15_plus_l20 = (last_20_slice["points"] >= 15).mean()
            pct_20_plus_l20 = (last_20_slice["points"] >= 20).mean()

            pct_10_plus_season = (prev_games["points"] >= 10).mean() if len(prev_games) > 0 else np.nan
            pct_15_plus_season = (prev_games["points"] >= 15).mean() if len(prev_games) > 0 else np.nan
            pct_20_plus_season = (prev_games["points"] >= 20).mean() if len(prev_games) > 0 else np.nan

            season_points = prev_games["points"]
            n_20pct = max(int(len(season_points) * 0.2), 1)
            worst_20pct_season = season_points.nsmallest(n_20pct).mean()

            current_points = current_game["points"]
            points_deviation_5 = current_points - (ppg_5 if ppg_5 is not np.nan else 0)
            points_deviation_10 = current_points - (ppg_10 if ppg_10 is not np.nan else 0)

            if not np.isnan(ppg_season):
                avg_points_dev_season = prev_games["points"].apply(lambda x: x - ppg_season).mean()
            else:
                avg_points_dev_season = np.nan

            if i > 0:
                last_game_points = group.iloc[i - 1]["points"]
                cleared_10_last_game = 1 if last_game_points >= 10 else 0
                cleared_15_last_game = 1 if last_game_points >= 15 else 0
                cleared_20_last_game = 1 if last_game_points >= 20 else 0
            else:
                cleared_10_last_game = np.nan
                cleared_15_last_game = np.nan
                cleared_20_last_game = np.nan

            label_10_plus = 1 if next_points >= 10 else 0
            label_15_plus = 1 if next_points >= 15 else 0
            label_20_plus = 1 if next_points >= 20 else 0

            row_dict = {
                "player_id": player_id,
                "player_name": current_game.get("player_name", None),
                "team_id": current_game.get("team_id", None),
                "season": season,
                "game_date": current_game["game_date"],
                "game_id": current_game.get("game_id", None),

                # Features
                "points_current_game": current_points,
                "ppg_last_5": ppg_5,
                "ppg_last_10": ppg_10,
                "ppg_last_20": ppg_20,
                "ppg_season": ppg_season,
                "trend_5_10": trend_5_10,
                "best_2_avg_last_10": best_2_avg_10,
                "worst_2_avg_last_10": worst_2_avg_10,
                "best_4_avg_last_20": best_4_avg_20,
                "worst_4_avg_last_20": worst_4_avg_20,
                "pct_10_plus_last_5": pct_10_plus_l5,
                "pct_15_plus_last_5": pct_15_plus_l5,
                "pct_20_plus_last_5": pct_20_plus_l5,
                "pct_10_plus_last_10": pct_10_plus_l10,
                "pct_15_plus_last_10": pct_15_plus_l10,
                "pct_20_plus_last_10": pct_20_plus_l10,
                "pct_10_plus_last_20": pct_10_plus_l20,
                "pct_15_plus_last_20": pct_15_plus_l20,
                "pct_20_plus_last_20": pct_20_plus_l20,
                "pct_10_plus_season": pct_10_plus_season,
                "pct_15_plus_season": pct_15_plus_season,
                "pct_20_plus_season": pct_20_plus_season,
                "worst_20pct_season": worst_20pct_season,
                "points_deviation_from_5": points_deviation_5,
                "points_deviation_from_10": points_deviation_10,
                "avg_points_deviation_season": avg_points_dev_season,
                "fga_last_5": fga_5,
                "fga_last_10": fga_10,
                "fga_season": fga_season,
                "minutes_last_5": min_5,
                "minutes_last_10": min_10,
                "minutes_season": min_season,
                "cleared_10_in_last_game": cleared_10_last_game,
                "cleared_15_in_last_game": cleared_15_last_game,
                "cleared_20_in_last_game": cleared_20_last_game,

                # Labels
                "label_10_plus": label_10_plus,
                "label_15_plus": label_15_plus,
                "label_20_plus": label_20_plus
            }

            engineered_rows.append(row_dict)

    return pd.DataFrame(engineered_rows)

#############################
# NEW: PLOT FEATURE IMPORTANCE OR CORRELATION
#############################

def plot_feature_importance_or_correlation(model, X, y, feature_columns, threshold_label, output_dir):
    """
    Plots feature importance if the model supports it, otherwise plots
    the correlation of features with the target.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # 1) If the model has feature_importances_ (tree-based)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_features = [feature_columns[i] for i in sorted_idx]

        sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
        plt.title(f"Feature Importances ({type(model).__name__}) - {threshold_label}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")

    # 2) If the model has coef_ (e.g. LogisticRegression)
    elif hasattr(model, 'coef_'):
        # For binary classification, model.coef_ is shape (1, n_features)
        coefs = np.abs(model.coef_[0])
        sorted_idx = np.argsort(coefs)[::-1]
        sorted_coefs = coefs[sorted_idx]
        sorted_features = [feature_columns[i] for i in sorted_idx]

        sns.barplot(x=sorted_coefs, y=sorted_features, palette='viridis')
        plt.title(f"Absolute Coefficients ({type(model).__name__}) - {threshold_label}")
        plt.xlabel("Coefficient Magnitude")
        plt.ylabel("Features")

    # 3) Otherwise, fallback: correlation with target
    else:
        df_temp = pd.DataFrame(X, columns=feature_columns).copy()
        df_temp["target"] = y.values if hasattr(y, "values") else y
        corrs = df_temp.corr()["target"].drop("target")
        corrs_abs = corrs.abs().sort_values(ascending=False)
        sns.barplot(x=corrs_abs.values, y=corrs_abs.index, palette='viridis')
        plt.title(f"Feature Correlation with Target - {threshold_label}")
        plt.xlabel("Absolute Correlation")
        plt.ylabel("Features")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"feature_importance_{threshold_label}.png")
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"[{threshold_label}] Feature importance/correlation plot saved to {plot_path}")

#############################
# 3) TRAIN MODELS
#############################

def train_and_evaluate_models(X, y, threshold_label, do_oversampling=True):
    """
    Splits data, optionally applies SMOTE, trains multiple classifiers,
    returns best model by .score().
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
        "GradientBoosting": GradientBoostingClassifier(),
        # NEW: Add LightGBM to the models dictionary
        "LightGBM": LGBMClassifier()  # <--- Added LightGBM
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

    # OPTIONAL: You can also create plots for each candidate model here, if you want.
    # But typically you'd plot only for the best model.

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
    for col in ["label_10_plus", "label_15_plus", "label_20_plus"]:
        vc = dataset_for_training[col].value_counts()
        logging.info(f"[Approach B: {col}] => {vc.to_dict()}")

    # D) Train each threshold
    feature_cols = [
        "ppg_last_5", "ppg_last_10", "ppg_last_20", "trend_5_10",
        "ppg_season",
        "best_2_avg_last_10", "worst_2_avg_last_10",
        "best_4_avg_last_20", "worst_4_avg_last_20",
        "pct_10_plus_last_5", "pct_15_plus_last_5", "pct_20_plus_last_5",
        "pct_10_plus_last_10", "pct_15_plus_last_10", "pct_20_plus_last_10",
        "pct_10_plus_last_20", "pct_15_plus_last_20", "pct_20_plus_last_20",
        "pct_10_plus_season", "pct_15_plus_season", "pct_20_plus_season",
        "worst_20pct_season",
        "points_deviation_from_5", "points_deviation_from_10",
        "avg_points_deviation_season",
        "fga_last_5", "fga_last_10", "fga_season",
        "minutes_last_5", "minutes_last_10", "minutes_season",
        "cleared_10_in_last_game", "cleared_15_in_last_game", "cleared_20_in_last_game"
    ]

    # Drop rows with missing
    df_b = dataset_for_training.dropna(subset=feature_cols).reset_index(drop=True)

    # Loop thresholds
    thresholds = {
        "10_plus": "label_10_plus",
        "15_plus": "label_15_plus",
        "20_plus": "label_20_plus"
    }

    X = df_b[feature_cols]
    os.makedirs("../models", exist_ok=True)

    # We can store plots here
    plots_dir = "../plots"
    os.makedirs(plots_dir, exist_ok=True)

    for tname, label_col in thresholds.items():
        y = df_b[label_col]

        # Decide SMOTE usage
        if tname in ["10_plus", "15_plus"]:
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

        # SAVE the best model
        model_path = f"../models/model_{tname}_approachB_{best_model_name}_acc_{best_score:.4f}.pkl"
        joblib.dump(best_model, model_path)
        logging.info(f"Saved best Approach B model for {tname} to {model_path}")

        # ---- NEW: Plot Feature Importance / Correlation for the best model ----
        plot_feature_importance_or_correlation(
            model=best_model,
            X=X,
            y=y,
            feature_columns=feature_cols,
            threshold_label=tname,
            output_dir=plots_dir
        )

    logging.info("Approach B training complete.")

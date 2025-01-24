import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import joblib
import requests
from bs4 import BeautifulSoup

from t_models import determine_season, compute_best_worst_avg  # Ensure these are accessible

# Define any additional helpers if needed (fetch_rotowire_page_text, is_player_healthy)

def fetch_rotowire_page_text():
    url = "https://www.rotowire.com/basketball/nba-lineups.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logging.error(f"Failed to fetch {url} (status: {resp.status_code}).")
            return ""
        return resp.text.lower()
    except Exception as e:
        logging.error(f"Error fetching rotowire: {e}")
        return ""

def is_player_healthy(player_name, rotowire_text, window_size=1000):
    if not player_name:
        return 0
    full_name = player_name.lower().strip()
    if not full_name:
        return 0
    idx = rotowire_text.find(full_name)
    if idx == -1:
        return 0
    start_sub = idx + len(full_name)
    end_sub = min(len(rotowire_text), start_sub + window_size)
    near_context = rotowire_text[start_sub:end_sub]
    if "span" in near_context:
        return 0
    return 1

# Reuse the no-season feature builder for a single player
def build_latest_features_for_player(df):
    """
    Uses build_latest_features_approach_b_no_season on a single player's DataFrame.
    """
    # Since build_latest_features_approach_b_no_season returns one row per player,
    # passing a single-player dataframe will give us exactly one row.
    return build_latest_features_approach_b_no_season(df)

def build_latest_features_approach_b_no_season(df):
    # ... include your previously defined build_latest_features_approach_b_no_season code here ...
    if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df.dropna(subset=["game_date"], inplace=True)

    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    latest_rows = []
    grouped = df.groupby("player_id", sort=False)

    for player_id, group in grouped:
        if group.empty:
            continue
        group = group.sort_values("game_date").reset_index(drop=True)
        i = len(group) - 1
        prev_games = group.iloc[: i+1]
        current_game = group.iloc[i]

        pg3_5 = (prev_games["3points"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["3points"].mean())
        pg3_10 = (prev_games["3points"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["3points"].mean())
        pg3_20 = (prev_games["3points"].rolling(20).mean().iloc[-1]
                  if len(prev_games) >= 20 else prev_games["3points"].mean())

        diff = pg3_5 - pg3_10
        trend_5_10 = diff

        fg3a_5 = (prev_games["fg3a"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["fg3a"].mean())
        fg3a_10 = (prev_games["fg3a"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["fg3a"].mean())

        min_5 = (prev_games["minutes"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["minutes"].mean())
        min_10 = (prev_games["minutes"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["minutes"].mean())

        pg3_season = prev_games["3points"].mean()
        fg3a_season = prev_games["fg3a"].mean()
        min_season = prev_games["minutes"].mean()

        window_10 = prev_games.iloc[-10:]["3points"] if len(prev_games) >= 10 else prev_games["3points"]
        best_2_avg_10, worst_2_avg_10 = compute_best_worst_avg(window_10, 10, 2, 2)

        window_20 = prev_games.iloc[-20:]["3points"] if len(prev_games) >= 20 else prev_games["3points"]
        best_4_avg_20, worst_4_avg_20 = compute_best_worst_avg(window_20, 20, 4, 4)

        last_10_slice = prev_games.iloc[-10:] if len(prev_games) >= 10 else prev_games
        pct_1_plus = (last_10_slice["3points"] >= 1).mean()
        pct_2_plus = (last_10_slice["3points"] >= 2).mean()
        pct_3_plus = (last_10_slice["3points"] >= 3).mean()

        # Seasonal percentages over all games for this player
        pct_1_plus_season = (prev_games["3points"] >= 1).mean() if len(prev_games) > 0 else np.nan
        pct_2_plus_season = (prev_games["3points"] >= 2).mean() if len(prev_games) > 0 else np.nan
        pct_3_plus_season = (prev_games["3points"] >= 3).mean() if len(prev_games) > 0 else np.nan

        season_3points = prev_games["3points"]
        n_20pct = max(int(len(season_3points) * 0.2), 1)
        worst_20pct_season = season_3points.nsmallest(n_20pct).mean()

        current_3points = current_game["3points"]
        points3_deviation_5 = current_3points - (pg3_5 if pg3_5 is not np.nan else 0)
        points3_deviation_10 = current_3points - (pg3_10 if pg3_10 is not np.nan else 0)

        avg_3points_dev_season = (prev_games["3points"].apply(lambda x: x - pg3_season).mean()
                                 if not np.isnan(pg3_season) else np.nan)

        if i > 0:
            last_game_3points = group.iloc[i - 1]["3points"]
            cleared_1_last_game = 1 if last_game_3points >= 1 else 0
            cleared_2_last_game = 1 if last_game_3points >= 2 else 0
            cleared_3_last_game = 1 if last_game_3points >= 3 else 0
        else:
            cleared_1_last_game = np.nan
            cleared_2_last_game = np.nan
            cleared_3_last_game = np.nan

        row_dict = {
            "player_id": player_id,
            "player_name": current_game.get("player_name", None),
            "team_id": current_game.get("team_id", None),
            "game_date": current_game["game_date"],
            "game_id": current_game.get("game_id", None),

            "pg3_last_5": pg3_5,
            "pg3_last_10": pg3_10,
            "pg3_last_20": pg3_20,
            "pg3_season": pg3_season,
            "trend_5_10": trend_5_10,
            "best_2_avg_last_10": best_2_avg_10,
            "worst_2_avg_last_10": worst_2_avg_10,
            "best_4_avg_last_20": best_4_avg_20,
            "worst_4_avg_last_20": worst_4_avg_20,
            "pct_1_plus_last_10": pct_1_plus,
            "pct_2_plus_last_10": pct_2_plus,
            "pct_3_plus_last_10": pct_3_plus,

            # Add new seasonal features
            "pct_1_plus_season": pct_1_plus_season,
            "pct_2_plus_season": pct_2_plus_season,
            "pct_3_plus_season": pct_3_plus_season,
        
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
            "cleared_3_in_last_game": cleared_3_last_game
        }
        latest_rows.append(row_dict)

    return pd.DataFrame(latest_rows)

def generate_non_overlapping_top_scorers(predictions_df, top_n=25):
    """
    Non-overlapping top scorers in priority order (3+, 2+, 1+), 
    sorted only by probability descending.
    Each player can appear in at most one threshold's CSV.
    """
    os.makedirs("../data/top_scorers", exist_ok=True)
    thresholds = [
        ("3_plus", "prob_3_plus"),
        ("2_plus", "prob_2_plus"),
        ("1_plus", "prob_1_plus")
    ]
    excluded_player_ids = set()

    # Fetch Rotowire page (assuming you use this for extra info, optional)
    rotowire_text = fetch_rotowire_page_text()
    if not rotowire_text:
        logging.warning("Rotowire text is empty. All players default to 0 (unhealthy).")

    for (threshold_name, prob_col) in thresholds:
        if prob_col not in predictions_df.columns:
            continue

        df_remaining = predictions_df[~predictions_df["player_id"].isin(excluded_player_ids)].copy()
        df_sorted = df_remaining.sort_values(by=prob_col, ascending=False)
        top_n_players = df_sorted.head(top_n).copy()

        def row_health(row):
            player_name = row['player_name']
            return is_player_healthy(player_name, rotowire_text, window_size=90)

        top_n_players['is_playing_today'] = top_n_players.apply(row_health, axis=1)

        out_path = f"../data/top_scorers/top_{top_n}_{threshold_name}.csv"
        top_n_players.to_csv(out_path, index=False)
        print(f"Saved top {top_n} for {threshold_name} to {out_path}")

        excluded_player_ids.update(top_n_players["player_id"].unique())

def main():
    # Argument parsing for player ID
    parser = argparse.ArgumentParser(description="Predict upcoming game performance for a single player.")
    parser.add_argument("player_id", type=str, help="Player ID to predict for")
    args = parser.parse_args()
    player_id = args.player_id

    # Logging setup
    logging.basicConfig(
        filename='../logs/predict_player.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logging.info(f"Starting prediction for player {player_id}.")

    # Load consolidated data
    consolidated_path = "../data/consolidated_database.csv"
    if not os.path.exists(consolidated_path):
        logging.error(f"Consolidated data not found at {consolidated_path}. Exiting.")
        sys.exit(1)

    df = pd.read_csv(consolidated_path)
    if df.empty:
        logging.error("Consolidated data is empty. Exiting.")
        sys.exit(1)

    # Filter for the specific player
    player_df = df[df["player_id"] == int(player_id)]
    print(player_df)
    if player_df.empty:
        logging.error(f"No data found for player {player_id}. Exiting.")
        sys.exit(1)

    # Build latest features for this player
    latest_features = build_latest_features_approach_b_no_season(player_df)
    if latest_features.empty:
        logging.error(f"No features generated for player {player_id}. Exiting.")
        sys.exit(1)

    # Drop NaNs from feature columns
    feature_cols = [
        "pg3_last_5", "pg3_last_10", "pg3_last_20", "trend_5_10",
        "pg3_season", "best_2_avg_last_10", "worst_2_avg_last_10",
        "best_4_avg_last_20", "worst_4_avg_last_20",
        "pct_1_plus_last_10", "pct_2_plus_last_10", "pct_3_plus_last_10",
        "pct_1_plus_season", "pct_2_plus_season", "pct_3_plus_season",  # Added seasonal features
        "worst_20pct_season", "3points_deviation_from_5", "3points_deviation_from_10",
        "avg_3points_deviation_season", "fg3a_last_5", "fg3a_last_10", "fg3a_season",
        "minutes_last_5", "minutes_last_10", "minutes_season",
        "cleared_1_in_last_game", "cleared_2_in_last_game", "cleared_3_in_last_game"
    ]
    latest_features = latest_features.dropna(subset=feature_cols).reset_index(drop=True)
    if latest_features.empty:
        logging.error("No valid features after dropping NaNs. Exiting.")
        sys.exit(1)

    # Load models
    thresholds = ["3_plus", "2_plus", "1_plus"]
    models = {}
    models_dir = "../models"

    for tname in thresholds:
        possible_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(f"model_{tname}_approachB") and f.endswith(".pkl")
        ]
        if not possible_files:
            logging.warning(f"No model file found for {tname} in {models_dir}.")
            continue
        possible_files.sort(reverse=True)
        best_file = possible_files[0]
        try:
            model_path = os.path.join(models_dir, best_file)
            model = joblib.load(model_path)
            models[tname] = model
            logging.info(f"Loaded model for {tname}: {best_file}")
        except Exception as e:
            logging.error(f"Error loading model {best_file} for {tname}: {e}")

    if not models:
        logging.error("No models loaded. Exiting.")
        sys.exit(1)

    # Predict for the single player
    X_input = latest_features[feature_cols]
    predictions = {}
    for tname in thresholds:
        if tname in models:
            model = models[tname]
            prob = model.predict_proba(X_input)[:, 1][0]  # single row prediction
            predictions[tname] = prob
        else:
            predictions[tname] = None

    # Print results to terminal
    print(f"Predictions for player {player_id}:")
    for tname in thresholds:
        prob = predictions[tname]
        if prob is not None:
            print(f"Probability of {tname} 3points: {prob:.4f}")
        else:
            print(f"No model available for {tname} threshold.")

if __name__ == "__main__":
    main()

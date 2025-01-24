import os
import pandas as pd
import numpy as np
import logging
import joblib
import requests
from bs4 import BeautifulSoup

from t_models import (  # or wherever you keep them
    determine_season,
    compute_best_worst_avg
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

def build_latest_features_approach_b_no_season(df):
    """
    Approach B with no season grouping:
      For each player_id, find the absolute last game in the dataset (covering all seasons).
      Compute rolling stats up to that game i, ignoring "season" boundaries.
      Return exactly ONE row per player.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df.dropna(subset=["game_date"], inplace=True)

    # Sort by player, then date
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    latest_rows = []
    grouped = df.groupby("player_id", sort=False)

    for player_id, group in grouped:
        if group.empty:
            continue

        group = group.sort_values("game_date").reset_index(drop=True)
        i = len(group) - 1  # The last row => absolute last game
        prev_games = group.iloc[: i+1]
        current_game = group.iloc[i]

        # -------------- Rolling Stats (Approach B) --------------
        ppg_5 = (prev_games["points"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["points"].mean())
        ppg_10 = (prev_games["points"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["points"].mean())
        ppg_20 = (prev_games["points"].rolling(20).mean().iloc[-1]
                  if len(prev_games) >= 20 else prev_games["points"].mean())

        fga_5 = (prev_games["fga"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["fga"].mean())
        fga_10 = (prev_games["fga"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["fga"].mean())

        min_5 = (prev_games["minutes"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["minutes"].mean())
        min_10 = (prev_games["minutes"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["minutes"].mean())

        ppg_season = prev_games["points"].mean()
        fga_season = prev_games["fga"].mean()
        min_season = prev_games["minutes"].mean()

        # Best/Worst slices
        window_10 = prev_games.iloc[-10:]["points"] if len(prev_games) >= 10 else prev_games["points"]
        best_2_avg_10, worst_2_avg_10 = compute_best_worst_avg(window_10, 10, 2, 2)

        window_20 = prev_games.iloc[-20:]["points"] if len(prev_games) >= 20 else prev_games["points"]
        best_4_avg_20, worst_4_avg_20 = compute_best_worst_avg(window_20, 20, 4, 4)

        last_10_slice = prev_games.iloc[-10:] if len(prev_games) >= 10 else prev_games
        pct_10_plus = (last_10_slice["points"] >= 10).mean()
        pct_15_plus = (last_10_slice["points"] >= 15).mean()
        pct_20_plus = (last_10_slice["points"] >= 20).mean()

        season_points = prev_games["points"]
        n_20pct = max(int(len(season_points) * 0.2), 1)
        worst_20pct_season = season_points.nsmallest(n_20pct).mean()

        current_points = current_game["points"]
        points_deviation_5 = current_points - (ppg_5 if ppg_5 is not np.nan else 0)
        points_deviation_10 = current_points - (ppg_10 if ppg_10 is not np.nan else 0)

        avg_points_dev_season = (
            prev_games["points"].apply(lambda x: x - ppg_season).mean()
            if not np.isnan(ppg_season) else np.nan
        )

        if i > 0:
            last_game_points = group.iloc[i - 1]["points"]
            cleared_10_last_game = 1 if last_game_points >= 10 else 0
            cleared_15_last_game = 1 if last_game_points >= 15 else 0
            cleared_20_last_game = 1 if last_game_points >= 20 else 0
        else:
            cleared_10_last_game = np.nan
            cleared_15_last_game = np.nan
            cleared_20_last_game = np.nan

        row_dict = {
            "player_id": player_id,
            "player_name": current_game.get("player_name", None),
            "team_id": current_game.get("team_id", None),
            # We'll keep the game_date for reference, but it doesn't control grouping
            "game_date": current_game["game_date"],
            "game_id": current_game.get("game_id", None),

            "ppg_last_5": ppg_5,
            "ppg_last_10": ppg_10,
            "ppg_last_20": ppg_20,
            "ppg_season": ppg_season,
            "best_2_avg_last_10": best_2_avg_10,
            "worst_2_avg_last_10": worst_2_avg_10,
            "best_4_avg_last_20": best_4_avg_20,
            "worst_4_avg_last_20": worst_4_avg_20,
            "pct_10_plus_last_10": pct_10_plus,
            "pct_15_plus_last_10": pct_15_plus,
            "pct_20_plus_last_10": pct_20_plus,
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
        }
        latest_rows.append(row_dict)

    return pd.DataFrame(latest_rows)


def generate_non_overlapping_top_scorers(predictions_df, top_n=25):
    """
    Non-overlapping top scorers in priority order (20+, 15+, 10+), 
    sorted only by probability descending.
    Each player can appear in at most one threshold's CSV.
    """
    os.makedirs("../data/top_scorers", exist_ok=True)
    thresholds = [
        ("20_plus", "prob_20_plus"),
        ("15_plus", "prob_15_plus"),
        ("10_plus", "prob_10_plus")
    ]
    excluded_player_ids = set()

    # 2) Fetch Rotowire HTML once
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
                player_name = row['player_name']  # e.g. "Anthony Edwards"
                return is_player_healthy(player_name, rotowire_text, window_size=90)

        top_n_players['is_playing_today'] = top_n_players.apply(row_health, axis=1)

        out_path = f"../data/top_scorers/top_{top_n}_{threshold_name}.csv"
        top_n_players.to_csv(out_path, index=False)
        print(f"Saved top {top_n} for {threshold_name} to {out_path}")

        excluded_player_ids.update(top_n_players["player_id"].unique())


def main():
    logging.basicConfig(
        filename='../logs/update_predictions.log',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s:%(message)s'
    )
    logging.info("Starting update_predictions with single-instance-per-player approach.")

    # 1) Load consolidated data
    consolidated_path = "../data/consolidated_database.csv"
    if not os.path.exists(consolidated_path):
        logging.error(f"No consolidated data found at {consolidated_path}. Exiting.")
        return

    df = pd.read_csv(consolidated_path)
    if df.empty:
        logging.error("Consolidated data is empty. Exiting.")
        return

    # 2) Build latest features with NO season grouping => exactly one row per player
    latest_features = build_latest_features_approach_b_no_season(df)
    if latest_features.empty:
        logging.warning("No latest features to predict on (maybe no data). Exiting.")
        return

    # 3) Drop NaNs from the feature columns
    feature_cols = [
        "ppg_last_5", "ppg_last_10", "ppg_last_20",
        "ppg_season", "best_2_avg_last_10", "worst_2_avg_last_10",
        "best_4_avg_last_20", "worst_4_avg_last_20",
        "pct_10_plus_last_10", "pct_15_plus_last_10", "pct_20_plus_last_10",
        "worst_20pct_season", "points_deviation_from_5", "points_deviation_from_10",
        "avg_points_deviation_season", "fga_last_5", "fga_last_10", "fga_season",
        "minutes_last_5", "minutes_last_10", "minutes_season",
        "cleared_10_in_last_game", "cleared_15_in_last_game", "cleared_20_in_last_game"
    ]
    latest_features = latest_features.dropna(subset=feature_cols).reset_index(drop=True)
    if latest_features.empty:
        logging.warning("All features are NaN after dropping => no rows. Exiting.")
        return

    # 4) Load models (20_plus, 15_plus, 10_plus)
    thresholds = ["20_plus", "15_plus", "10_plus"]
    models = {}
    models_dir = "../models"

    for tname in thresholds:
        possible_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(f"model_{tname}_approachB") and f.endswith(".pkl")
        ]
        if not possible_files:
            logging.warning(f"No model file found for {tname} in {models_dir}. Skipping.")
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
        return

    # 5) Predict
    predictions_df = latest_features[["player_id", "player_name", "team_id", "game_date"]].copy()

    X_input = latest_features[feature_cols]

    for tname in thresholds:
        if tname not in models:
            predictions_df[f"prob_{tname}"] = np.nan
            continue

        model = models[tname]
        probs = model.predict_proba(X_input)[:, 1]
        predictions_df[f"prob_{tname}"] = probs

    # 6) Generate top-25 for 20_plus, 15_plus, 10_plus without duplicates
    generate_non_overlapping_top_scorers(predictions_df, top_n=25)

    logging.info("update_predictions complete. One row per player => no duplicates.")


if __name__ == "__main__":
    main()

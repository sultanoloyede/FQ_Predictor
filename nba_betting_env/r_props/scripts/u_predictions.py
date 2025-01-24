import os
import pandas as pd
import numpy as np
import logging
import joblib
import requests
import unicodedata
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
    """
    if not player_name:
        return 0

    # Normalize the name: remove accents and specified suffixes
    normalized_name = unicodedata.normalize('NFKD', player_name) \
                          .encode('ASCII', 'ignore') \
                          .decode('ASCII')
    # Remove generational suffixes and trim whitespace
    normalized_name = normalized_name.replace(" Jr.", "").replace(" jr.", "")
    normalized_name = normalized_name.replace(" II", "").replace(" III", "") \
                                     .replace(" ii", "").replace(" iii", "") \
                                     .strip()

    full_name = normalized_name.lower()
    if not full_name:
        return 0

    # Find the index of the player's name in the page text
    idx = rotowire_text.find(full_name)
    if idx == -1:
        # Name not found => 0
        return 0

    # If found, examine the next `window_size` characters
    start_sub = idx + len(full_name)
    end_sub = min(len(rotowire_text), start_sub + window_size)
    near_context = rotowire_text[start_sub:end_sub]

    print()
    print(near_context)
    # If "span" is found in near_context, treat player as not healthy
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
        rpg_5 = (prev_games["rebounds"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["rebounds"].mean())
        rpg_10 = (prev_games["rebounds"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["rebounds"].mean())
        rpg_20 = (prev_games["rebounds"].rolling(20).mean().iloc[-1]
                  if len(prev_games) >= 20 else prev_games["rebounds"].mean())

        # Compute trend_5_10 based on ppg_5 and ppg_10
        diff = rpg_5 - rpg_10
        trend_5_10 = diff

        min_5 = (prev_games["minutes"].rolling(5).mean().iloc[-1]
                 if len(prev_games) >= 5 else prev_games["minutes"].mean())
        min_10 = (prev_games["minutes"].rolling(10).mean().iloc[-1]
                  if len(prev_games) >= 10 else prev_games["minutes"].mean())

        rpg_season = prev_games["rebounds"].mean()
        min_season = prev_games["minutes"].mean()

        # Best/Worst slices
        window_10 = prev_games.iloc[-10:]["rebounds"] if len(prev_games) >= 10 else prev_games["rebounds"]
        best_2_avg_10, worst_2_avg_10 = compute_best_worst_avg(window_10, 10, 2, 2)

        window_20 = prev_games.iloc[-20:]["rebounds"] if len(prev_games) >= 20 else prev_games["rebounds"]
        best_4_avg_20, worst_4_avg_20 = compute_best_worst_avg(window_20, 20, 4, 4)

        last_5_slice = prev_games.iloc[-5:] if len(prev_games) >= 5 else prev_games
        pct_4_plus_l5 = (last_5_slice["rebounds"] >= 4).mean()
        pct_6_plus_l5 = (last_5_slice["rebounds"] >= 6).mean()
        pct_8_plus_l5 = (last_5_slice["rebounds"] >= 8).mean()
        pct_10_plus_l5 = (last_5_slice["rebounds"] >= 10).mean()

        last_10_slice = prev_games.iloc[-10:] if len(prev_games) >= 10 else prev_games
        pct_4_plus_l10 = (last_10_slice["rebounds"] >= 4).mean()
        pct_6_plus_l10 = (last_10_slice["rebounds"] >= 6).mean()
        pct_8_plus_l10 = (last_10_slice["rebounds"] >= 8).mean()
        pct_10_plus_l10 = (last_10_slice["rebounds"] >= 10).mean()

        last_20_slice = prev_games.iloc[-20:] if len(prev_games) >= 20 else prev_games
        pct_4_plus_l20 = (last_20_slice["rebounds"] >= 4).mean()
        pct_6_plus_l20 = (last_20_slice["rebounds"] >= 6).mean()
        pct_8_plus_l20 = (last_20_slice["rebounds"] >= 8).mean()
        pct_10_plus_l20 = (last_20_slice["rebounds"] >= 10).mean()

        # ---- New: Seasonal percentages ----
        pct_4_plus_season = (prev_games["rebounds"] >= 4).mean() if len(prev_games) > 0 else np.nan
        pct_6_plus_season = (prev_games["rebounds"] >= 6).mean() if len(prev_games) > 0 else np.nan
        pct_8_plus_season = (prev_games["rebounds"] >= 8).mean() if len(prev_games) > 0 else np.nan
        pct_10_plus_season = (prev_games["rebounds"] >= 10).mean() if len(prev_games) > 0 else np.nan

        season_rebounds = prev_games["rebounds"]
        n_20pct = max(int(len(season_rebounds) * 0.2), 1)
        worst_20pct_season = season_rebounds.nsmallest(n_20pct).mean()

        current_rebounds = current_game["rebounds"]
        rebounds_deviation_5 = current_rebounds - (rpg_5 if rpg_5 is not np.nan else 0)
        rebounds_deviation_10 = current_rebounds - (rpg_10 if rpg_10 is not np.nan else 0)

        avg_rebounds_dev_season = (
            prev_games["rebounds"].apply(lambda x: x - rpg_season).mean()
            if not np.isnan(rpg_season) else np.nan
        )

        if i > 0:
            last_game_rebounds = group.iloc[i - 1]["rebounds"]
            cleared_4_last_game = 1 if last_game_rebounds >= 4 else 0
            cleared_6_last_game = 1 if last_game_rebounds >= 6 else 0
            cleared_8_last_game = 1 if last_game_rebounds >= 8 else 0
            cleared_10_last_game = 1 if last_game_rebounds >= 10 else 0
        else:
            cleared_4_last_game = np.nan
            cleared_6_last_game = np.nan
            cleared_8_last_game = np.nan
            cleared_10_last_game = np.nan

        row_dict = {
            "player_id": player_id,
            "player_name": current_game.get("player_name", None),
            "team_id": current_game.get("team_id", None),
            # We'll keep the game_date for reference, but it doesn't control grouping
            "game_date": current_game["game_date"],
            "game_id": current_game.get("game_id", None),

            "rpg_last_5": rpg_5,
            "rpg_last_10": rpg_10,
            "rpg_last_20": rpg_20,
            "rpg_season": rpg_season,

            # New feature:
            "trend_5_10": trend_5_10,

            "best_2_avg_last_10": best_2_avg_10,
            "worst_2_avg_last_10": worst_2_avg_10,
            "best_4_avg_last_20": best_4_avg_20,
            "worst_4_avg_last_20": worst_4_avg_20,
            "pct_4_plus_last_5": pct_4_plus_l5,
            "pct_6_plus_last_5": pct_6_plus_l5,
            "pct_8_plus_last_5": pct_8_plus_l5,
            "pct_10_plus_last_5": pct_10_plus_l5,
            "pct_4_plus_last_10": pct_4_plus_l10,
            "pct_6_plus_last_10": pct_6_plus_l10,
            "pct_8_plus_last_10": pct_8_plus_l10,
            "pct_10_plus_last_10": pct_10_plus_l10,
            "pct_4_plus_last_20": pct_4_plus_l20,
            "pct_6_plus_last_20": pct_6_plus_l20,
            "pct_8_plus_last_20": pct_8_plus_l20,
            "pct_10_plus_last_20": pct_10_plus_l20,

            # New seasonal percentages
            "pct_4_plus_season": pct_4_plus_season,
            "pct_6_plus_season": pct_6_plus_season,
            "pct_8_plus_season": pct_8_plus_season,
            "pct_10_plus_season": pct_10_plus_season,

            "worst_20pct_season": worst_20pct_season,
            "rebounds_deviation_from_5": rebounds_deviation_5,
            "rebounds_deviation_from_10": rebounds_deviation_10,
            "avg_rebounds_deviation_season": avg_rebounds_dev_season,
            "minutes_last_5": min_5,
            "minutes_last_10": min_10,
            "minutes_season": min_season,
            "cleared_4_in_last_game": cleared_4_last_game,
            "cleared_6_in_last_game": cleared_6_last_game,
            "cleared_8_in_last_game": cleared_8_last_game,
            "cleared_10_in_last_game": cleared_10_last_game,
        }
        latest_rows.append(row_dict)

    return pd.DataFrame(latest_rows)


def generate_non_overlapping_top_rebounders(predictions_df, top_n=10):
    """
    Non-overlapping top rebounders in priority order (20+, 15+, 10+), 
    sorted only by probability descending.
    Each player can appear in at most one threshold's CSV.
    """
    os.makedirs("../data/top_rebounders", exist_ok=True)
    thresholds = [
        ("10_plus", "prob_10_plus"),
        ("8_plus", "prob_8_plus"),
        ("6_plus", "prob_6_plus"),
        ("4_plus", "prob_4_plus")
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

        out_path = f"../data/top_rebounders/top_{top_n}_{threshold_name}.csv"
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
        "rpg_last_5", "rpg_last_10", "rpg_last_20", "trend_5_10",
        "rpg_season", "best_2_avg_last_10", "worst_2_avg_last_10",
        "best_4_avg_last_20", "worst_4_avg_last_20",
        "pct_4_plus_last_5", "pct_6_plus_last_5", "pct_8_plus_last_5", "pct_10_plus_last_5",
        "pct_4_plus_last_10", "pct_6_plus_last_10", "pct_8_plus_last_10", "pct_10_plus_last_10",
        "pct_4_plus_last_20", "pct_6_plus_last_20", "pct_8_plus_last_20", "pct_10_plus_last_20",
        "pct_4_plus_season", "pct_6_plus_season", "pct_8_plus_season", "pct_10_plus_season",
        "worst_20pct_season", "rebounds_deviation_from_5", "rebounds_deviation_from_10",
        "avg_rebounds_deviation_season",
        "minutes_last_5", "minutes_last_10", "minutes_season",
        "cleared_4_in_last_game", "cleared_6_in_last_game", "cleared_8_in_last_game", "cleared_10_in_last_game"
    ]
    latest_features = latest_features.dropna(subset=feature_cols).reset_index(drop=True)
    if latest_features.empty:
        logging.warning("All features are NaN after dropping => no rows. Exiting.")
        return

    # 4) Load models (10_plus, 8_plus, 6_plus, 4_plus)
    thresholds = ["10_plus", "8_plus", "6_plus", "4_plus"]
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

    # 6) Generate top-10 for 10_plus, 8_plus, 6_plus, 4_plus without duplicates
    generate_non_overlapping_top_rebounders(predictions_df, top_n=10)

    logging.info("update_predictions complete. One row per player => no duplicates.")


if __name__ == "__main__":
    main()

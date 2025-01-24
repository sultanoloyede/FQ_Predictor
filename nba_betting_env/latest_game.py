import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder
from datetime import datetime
import time
from nba_api.stats.endpoints import boxscoretraditionalv2

def fetch_box_score(game_id):
    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    player_stats = boxscore.player_stats.get_data_frame()
    team_stats = boxscore.team_stats.get_data_frame()
    return player_stats, team_stats

def get_last_3_regular_season_games(season='2024-25'):
    """
    Fetches the last 3 completed NBA regular season game IDs for the specified season.
    
    Parameters:
        season (str): The NBA season in the format 'YYYY-YY', e.g., '2022-23'.
    
    Returns:
        List[str]: A list of the last 3 unique GAME_IDs.
    """
    # Initialize the LeagueGameFinder with the desired season type
    game_finder = LeagueGameFinder(season_nullable=season, season_type_nullable='Regular Season')
    
    # Retrieve the DataFrame containing the game data
    games_df = game_finder.get_data_frames()[0]
    
    if games_df.empty:
        print(f"No games found for the {season} Regular Season.")
        return []
    
    # Sort the DataFrame by GAME_DATE descending to get the most recent games first
    games_sorted = games_df.sort_values('GAME_ID', ascending=False)
    
    # Drop duplicate GAME_IDs to ensure each game is only counted once
    unique_games = games_sorted.drop_duplicates(subset='GAME_ID', keep='first')
    
    # Select the top 3 GAME_IDs
    last_3_game_ids = unique_games.head(3)['GAME_ID'].tolist()
    
    return last_3_game_ids

def main():
    # Specify the season you are interested in
    # Example formats: '2022-23', '2021-22', etc.
    season = '2024-25'  # Update this to the current season as needed
    
    print(f"Fetching the last 3 regular season game IDs for the {season} season...")
    
    # Fetch the last 20 regular season game IDs
    last_3_games = get_last_3_regular_season_games(season=season)
    
    if last_3_games:
        print("\nLast 3 Regular Season Completed Game IDs:")
        for idx, game_id in reversed(list(enumerate(last_3_games, start=1))):
            print(f"{idx}: {game_id}")
            player_stats, team_stats = fetch_box_score(game_id)
            print(f"\nBox Score for Game ID {game_id}:")
            print("Player Stats:")
            print(player_stats.head())
            print("Team Stats:")
            print(team_stats)
            # Optional: Add delays to prevent rate limiting
            time.sleep(1)
    else:
        print("No game IDs retrieved.")
        

if __name__ == "__main__":
    main()

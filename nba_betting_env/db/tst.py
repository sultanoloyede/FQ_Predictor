import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoresummaryv2,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
)
from nba_api.stats.endpoints import leaguedashteamstats
import pickle
import os
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2, LeagueGameFinder
from tqdm import tqdm
import json

def fetch_fg_pct(game_id, team_id):
    """
    Fetches the FG_PCT for a given game and team. Utilizes a cache to prevent redundant API calls.
    
    Parameters:
        game_id (str): The unique 10-digit identifier for the game.
        team_id (int): The unique identifier for the team.
        cache (dict): Cache dictionary to store/retrieve FG_PCT.
    
    Returns:
        float: The FG_PCT value or NaN if not available.
    """
    
    try:
        # Ensure game_id is formatted as a 10-digit string
        game_id_str = str(game_id).zfill(10)
        
        # Respect API rate limits
        time.sleep(0.6)
        
        # Fetch data from BoxScoreTraditionalV2 for the entire game
        boxscore = BoxScoreAdvancedV2(
            game_id=game_id_str,
            start_period=1, 
            end_period=1,
            range_type=1,
            start_range=1,
            end_range=1
        )
        
        # Access the team stats dataset
        team_stats = boxscore.team_stats.get_data_frame()
        
        print(team_stats)

        # Filter by team_id to find the specific team's stats
        team_row = team_stats[team_stats['TEAM_ID'] == team_id]
        
        if not team_row.empty:
            fg_pct = team_row['FG_PCT'].values[0]
            print(f"FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id} fetched from API: {fg_pct}")
            return fg_pct
        else:
            print(f"No data found for TEAM_ID {team_id} in GAME_ID {game_id_str}.")
            return np.nan
    
    except Exception as e:
        print(f"Error fetching FG_PCT for GAME_ID {game_id}, TEAM_ID {team_id}: {e}")
        return np.nan

game_id = str(22401219).zfill(10)
team_id ='1610612756'
fetch_fg_pct(game_id, team_id)
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (leaguegamefinder)

database = pd.DataFrame()

# Returns game ids
def fetch_games():
    #seasons = np.array(['2017-18', '2018-19', '2019-20', '2020-21', '2021,22', '2022-23', '2023-24'])

    seasons = np.array(['2017-18'])

    games = np.array([])

    print("Fetching NBA games...")

    for season in seasons:
        print(f"Processing the {season} season...")

        # Gets all the games for a specific season
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, season_type_nullable='Regular Season'
        )

        games = gamefinder.get_data_frames()[0]

        print(f"Number of games fetched for season {season} before filtering: {len(games)}")

def fetch_efg(game_id):
    print("%")


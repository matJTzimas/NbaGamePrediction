import os 
import sys
import pandas as pd
from nba.entity.config_entity import *
from nba.entity.artifact_entity import *
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from nba.utils.main_utils import *
from nba.utils.transformation_utils import home_away_id

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            self.players_df = pd.read_csv(self.data_validation_artifact.validated_players_path)
            self.games_df = pd.read_csv(self.data_validation_artifact.validated_games_path)
            logging.info(f'Read players and games data for transformation')

        except Exception as e:
            raise NbaException(e, sys)

    def get_players_game_stats(self, game_id):
        """
        For the specified game_id, get the stats of all players on both teams playing in that
        """

        home_id, away_id = home_away_id(self.games_df, game_id)
        training_cols = ["PLAYER_ID", "TEAM_ID", "GAME_ID","SEASON_YEAR"] + self.data_transformation_config.important_player_stats

        # players in this game
        game_players_stats = self.players_df[self.players_df["GAME_ID"] == game_id]
        home_players_game_stats = game_players_stats[game_players_stats["TEAM_ID"] == home_id][training_cols].copy()
        away_players_game_stats = game_players_stats[game_players_stats["TEAM_ID"] == away_id][training_cols].copy()

        home_players_game_stats['HOME_ID'] =  away_players_game_stats['HOME_ID'] = home_id
        
        away_players_game_stats['AWAY_ID'] =  home_players_game_stats['AWAY_ID'] = away_id

        return home_players_game_stats, away_players_game_stats


    def get_player_season_stats_pre_gameid(self, game_id):
        """
        For the specified game_id, for the two teams playing, get the season stats of all players on both teams
        up to (but not including) the game date.  

        Args:
            - game_id (int): The game ID of the specific game.
        Returns:
            - tuple: (home_team_player_stats_dataframe, away_team_player_stats_dataframe) 

        """
        home_id, away_id = home_away_id(self.games_df, game_id)
        training_cols = ["PLAYER_ID", "TEAM_ID"] + self.data_transformation_config.important_player_stats

        home_players_game_stats, away_players_game_stats = self.get_players_game_stats(game_id)

        game_date = self.games_df.loc[self.games_df["GAME_ID"] == game_id, "GAME_DATE"].iloc[0]

        # season stats before this game
        mask = (
            (self.players_df["GAME_DATE"] < game_date)
            & (self.players_df["TEAM_ID"].isin([home_id, away_id]))
        )
        season_df = self.players_df.loc[mask, training_cols]

        players_season_stats = (
            season_df
            .groupby(["PLAYER_ID", "TEAM_ID"], as_index=False)[training_cols]
            .mean()
        )

        # match only players who appeared in this specific game
        home_players_season_stats = players_season_stats[
            players_season_stats["PLAYER_ID"].isin(home_players_game_stats["PLAYER_ID"])
        ].copy()
        away_players_season_stats = players_season_stats[
            players_season_stats["PLAYER_ID"].isin(away_players_game_stats["PLAYER_ID"])
        ].copy()

        home_players_season_stats['HOME_ID'] = away_players_season_stats['HOME_ID'] = home_id
        
        away_players_season_stats['AWAY_ID'] = home_players_season_stats['AWAY_ID'] = away_id

        return home_players_season_stats, away_players_season_stats





    def initialize_data_transformation(self):
        try: 
            home, away = self.get_player_season_stats_pre_gameid(game_id=12300002)
            print(home.head())
            print(away.head())
        except Exception as e:
            raise NbaException(e, sys)
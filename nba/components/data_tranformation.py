import os 
import sys
import pandas as pd
from nba.entity.config_entity import *
from nba.entity.artifact_entity import *
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from nba.utils.main_utils import *
from nba.utils.transformation_utils import home_away_id, add_prefix, moneyline_to_probability
from tqdm import tqdm
from nba_api.stats.static import teams
import numpy as np

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            self.teams_df = pd.DataFrame(teams.get_teams())
            self.players_df = pd.read_csv(self.data_validation_artifact.validated_players_path)
            self.games_df = pd.read_csv(self.data_validation_artifact.validated_games_path)
            self.odds_df = pd.read_csv(self.data_validation_artifact.validated_odds_path)
            logging.info(f'Read players and games data for transformation')

        except Exception as e:
            raise NbaException(e, sys)

    def get_players_game_stats(self, game_id):
        """
        For the specified game_id, get the stats of all players on both teams playing in that
        """

        home_id, away_id = home_away_id(self.games_df, game_id)
        training_cols = ["PLAYER_ID", "TEAM_ID", "GAME_ID","SEASON_YEAR", "GAME_DATE"] + self.data_transformation_config.important_player_stats

        # players in this game
        game_players_stats = self.players_df[self.players_df["GAME_ID"] == game_id]
        home_players_game_stats = game_players_stats[game_players_stats["TEAM_ID"] == home_id][training_cols].copy()
        away_players_game_stats = game_players_stats[game_players_stats["TEAM_ID"] == away_id][training_cols].copy()

        home_players_game_stats['HOME_ID'] =  away_players_game_stats['HOME_ID'] = home_id
        
        away_players_game_stats['AWAY_ID'] =  home_players_game_stats['AWAY_ID'] = away_id

        away_players_game_stats["GAME_ID"] = home_players_game_stats["GAME_ID"] = game_id

        return add_prefix(home_players_game_stats, "GAME", self.data_transformation_config.important_player_stats), add_prefix(away_players_game_stats, "GAME",  self.data_transformation_config.important_player_stats)


    def get_player_alltime_stats_pre_gameid(self, game_id):
        """
        For the specified game_id, for the two teams playing, get the all time stats of all players on both teams
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

        # player games before this game
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
        home_players_alltime_stats = players_season_stats[
            players_season_stats["PLAYER_ID"].isin(home_players_game_stats["PLAYER_ID"])
        ].copy()
        away_players_alltime_stats = players_season_stats[
            players_season_stats["PLAYER_ID"].isin(away_players_game_stats["PLAYER_ID"])
        ].copy()

        home_players_alltime_stats['HOME_ID'] = away_players_alltime_stats['HOME_ID'] = home_id
        
        away_players_alltime_stats['AWAY_ID'] = home_players_alltime_stats['AWAY_ID'] = away_id

        away_players_alltime_stats["GAME_ID"] = home_players_alltime_stats["GAME_ID"] = game_id

        return add_prefix(home_players_alltime_stats, "ALL", self.data_transformation_config.important_player_stats), add_prefix(away_players_alltime_stats, "ALL", self.data_transformation_config.important_player_stats)


    def get_player_season_stats_pre_gameid(self, game_id):
        """
        For the specified game_id, for the two teams playing, get the season stats of all players on both teams
        up to (but not including) the game date.  

        Args:
            - game_id (int): The game ID of the specific game.
        Returns:
            - tuple: (home_team_player_stats_dataframe, away_team_player_stats_dataframe) 

        """
        home_id, away_id = home_away_id(self.games_df, game_id) # get the home and away ids of this game 
        training_cols = ["PLAYER_ID", "TEAM_ID"] + self.data_transformation_config.important_player_stats # column name to keep 

        home_players_game_stats, away_players_game_stats = self.get_players_game_stats(game_id) # get the player stats for this game

        current_game_df = self.games_df.loc[self.games_df["GAME_ID"] == game_id, :]
        game_date = current_game_df.iloc[0]["GAME_DATE"]
        season_id = current_game_df.iloc[0]["SEASON_YEAR"]


        # player games before this game for this season
        mask = (
            (self.players_df["GAME_DATE"] < game_date)
            & (self.players_df["SEASON_YEAR"] == season_id)
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

        home_players_season_stats["GAME_ID"] = away_players_season_stats["GAME_ID"] = game_id

        
        return add_prefix(home_players_season_stats, "SEASON", self.data_transformation_config.important_player_stats), add_prefix(away_players_season_stats, "SEASON", self.data_transformation_config.important_player_stats)

    
    def merge_stats_for_gameid(self, game_id):
        """
        Merging all time, season and game stats for all players
        """
        try:
            home_game_stats, away_game_stats = self.get_players_game_stats(game_id=game_id)
            home_season_stats, away_season_stats = self.get_player_season_stats_pre_gameid(game_id=game_id)
            home_alltime_stats, away_alltime_stats = self.get_player_alltime_stats_pre_gameid(game_id=game_id)

            home_merged = home_game_stats.merge(home_season_stats, on=["PLAYER_ID", "TEAM_ID", "HOME_ID", "AWAY_ID", "GAME_ID"], how="left")\
                                         .merge(home_alltime_stats, on=["PLAYER_ID", "TEAM_ID", "HOME_ID", "AWAY_ID", "GAME_ID"], how="left")
            
            away_merged = away_game_stats.merge(away_season_stats, on=["PLAYER_ID", "TEAM_ID", "HOME_ID", "AWAY_ID", "GAME_ID"], how="left")\
                                         .merge(away_alltime_stats, on=["PLAYER_ID", "TEAM_ID", "HOME_ID", "AWAY_ID", "GAME_ID"], how="left")

            return home_merged, away_merged

        except Exception as e:
            raise NbaException(e, sys)


    def validate_gameid(self, game_id):
        """
        Validate for a given game_id that both teams are valid NBA teams.
        """

        df_teams = pd.DataFrame(teams.get_teams())
        mat = self.games_df.loc[self.games_df['GAME_ID'] == game_id, 'MATCHUP'].iloc[0]

        if '@' in mat:
            teams_abb = mat.split(' @ ')
            if teams_abb[0] not in df_teams['abbreviation'].values or teams_abb[1] not in df_teams['abbreviation'].values:
                return False
            
        else:
            teams_abb = mat.split(' vs. ')
            if teams_abb[0] not in df_teams['abbreviation'].values or teams_abb[1] not in df_teams['abbreviation'].values:
                return False
            
        return True


    def concatenate_gameids_data(self):
        """
        concatenate on x axis (=0) the stats from all players
        """

        try:
            final_list_of_dfs = [] 
            all_game_ids = self.games_df['GAME_ID'].unique().tolist() # get all game ids 
            logging.info(f'CREATING MERGED DATA FOR {len(all_game_ids)} GAMES') 

            for gameid in tqdm(all_game_ids):
                if not self.validate_gameid(gameid): # Double check if a team do not belong in the nba
                    logging.info(f'Skipping game id {gameid} as one of the teams is not an NBA team')
                    continue

                home_merged, away_merged = self.merge_stats_for_gameid(game_id=gameid) # return two dataframes with game, season and alltime historical data of the player until the data of the game id 
                final_list_of_dfs.append(home_merged) 
                final_list_of_dfs.append(away_merged)
            
            final_df = pd.concat(final_list_of_dfs, axis=0)
            final_df['GAME_DATE'] = pd.to_datetime(final_df['GAME_DATE']).dt.date

            return final_df.fillna(0)

        except Exception as e:
            raise NbaException(e, sys)


    def transform_odd_data(self):
        try:
            logging.info(f'Transforming odds data')
            # keep only plauyoffs and regular season games
            self.odds_df['prob_home'], self.odds_df['prob_away'] = zip(*self.odds_df.apply(lambda row: moneyline_to_probability(row),axis =1))
            # map formats 
            self.odds_df['home'] = self.odds_df['home'].map(self.data_transformation_config.mapping_dict)
            self.odds_df['away'] = self.odds_df['away'].map(self.data_transformation_config.mapping_dict)

            abb_to_id = dict(zip(self.teams_df["abbreviation"], self.teams_df["id"]))

            self.odds_df['HOME_ID'] =  self.odds_df['home'].map(abb_to_id)
            self.odds_df['AWAY_ID'] =  self.odds_df['away'].map(abb_to_id)
            self.odds_df = self.odds_df.rename(columns={"date": "GAME_DATE"})
            self.odds_df = self.odds_df[self.data_transformation_config.important_odds_stats]
            self.odds_df['GAME_DATE'] = pd.to_datetime(self.odds_df['GAME_DATE']).dt.date
            
        except Exception as e:
            raise NbaException(e, sys)


    def initialize_data_transformation(self):
        try:
            logging.info(f'Initializing data transformation')

            logging.info(f'Concatenate all gameids player data')
            concatenated_df = self.concatenate_gameids_data()
            

            logging.info(f'Transform Odd data')
            self.transform_odd_data()


            logging.info(f'Add the odd data to the concatenated_df')
            final_df = concatenated_df.merge(
                self.odds_df,
                on=["GAME_DATE", "HOME_ID", "AWAY_ID"],  # list of columns
                how="inner"                     # or "left", "right", "outer"
            )

            if os.path.exists(self.data_transformation_config.transformed_file_path):
                if self.data_transformation_config.force_rebuild_csv:
                    logging.info(f'Overwriting existing transformed')    
                    final_df.to_csv(self.data_transformation_config.transformed_file_path, mode='w', index=False, header=True)
            else:
                final_df.to_csv(self.data_transformation_config.transformed_file_path, mode='w', index=False, header=True)
            logging.info(f'Data transformation completed. Transformed file path: {self.data_transformation_config.transformed_file_path}')

        except Exception as e:
            raise NbaException(e, sys)
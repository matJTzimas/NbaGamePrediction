from nba.entity.config_entity import DataIngestionConfig
import sys 
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import boxscore
from nba_api.stats.endpoints import TeamGameLogs ,playergamelogs
from nba.exception.exception import NbaException
from nba.logging.logger import logging
import time
import random
import pandas as pd 
import os
from nba.entity.artifact_entity import DataIngestionArtifact
import kagglehub
from kagglehub import KaggleDatasetAdapter as KDA

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f'Entered the data ingestion initialization')
            self.data_ingestion_config = data_ingestion_config
            self.store_option = self.data_ingestion_config.store_option

        except Exception as e:
            raise NbaException(e, sys)

    def create_csv_from_api(self):

        team_game_logs_list = [] 
        player_game_logs_list = []

        try:
            #### games csv
            if not os.path.exists(self.data_ingestion_config.games_file_path):
                logging.info(f'Selecting all games for all seasons')
                for season in self.data_ingestion_config.seasons:
                    logging.info(f'Selecting all games for the season: {season}')
                    team_game_logs_list.append(TeamGameLogs(season_nullable=season,league_id_nullable='00').get_data_frames()[0])
                    time.sleep(random.uniform(0.5, 1.5))
                games_df = pd.concat(team_game_logs_list, axis=0)
                # only regular season and playoff games
                games_df["GameID_str"] = games_df["GAME_ID"].astype(str).str.zfill(10)
                games_df = games_df[ 
                    (games_df["GameID_str"].str.startswith("002")) |
                    (games_df["GameID_str"].str.startswith("004"))
                ]

                games_df.to_csv(self.data_ingestion_config.games_file_path, index=False)
            else:
                logging.info(f'Games file already exists at path: {self.data_ingestion_config.games_file_path}')    

            #### players csv 
            if not os.path.exists(self.data_ingestion_config.players_file_path):
                for season in self.data_ingestion_config.seasons:
                    logging.info(f'Selecting all players for the season: {season}')
                    player_game_logs_list.append(playergamelogs.PlayerGameLogs(season_nullable=season,league_id_nullable='00',timeout=60).get_data_frames()[0])
                    time.sleep(random.uniform(0.5, 1.5))
                players_df = pd.concat(player_game_logs_list, axis=0)
                players_df["GameID_str"] = players_df["GAME_ID"].astype(str).str.zfill(10)
                players_df = players_df[ 
                    (players_df["GameID_str"].str.startswith("002")) |
                    (players_df["GameID_str"].str.startswith("004"))
                ]
                players_df.to_csv(self.data_ingestion_config.players_file_path, index=False)
            else:
                logging.info(f'Players file already exists at path: {self.data_ingestion_config.players_file_path}')

            #### Odds csv
            if not os.path.exists(self.data_ingestion_config.odds_file_path):
                df = kagglehub.dataset_load(
                    KDA.PANDAS,
                    "cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024",  # dataset handle
                    "nba_2008-2025.csv",                                                # file inside the dataset
                )
                df['SEASON_YEAR'] = (df['season'] -1).astype(str) + '-' + (df['season']).astype(str).str[-2:]
                df = df[df['playoffs'] != df['regular']]
                df = df[df['SEASON_YEAR'].isin(self.data_ingestion_config.seasons)]

                df.to_csv(self.data_ingestion_config.odds_file_path, index=False)
            else:
                logging.info(f'Odds file already exists at path: {self.data_ingestion_config.odds_file_path}')


            logging.info(f'Exited the data ingestion')

            return self.data_ingestion_config.players_file_path, self.data_ingestion_config.games_file_path, self.data_ingestion_config.odds_file_path

        except Exception as e:
            raise NbaException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info(f'Initiated the data ingestion')
            if self.store_option == 'csv':
                logging.info(f'CSV file option selected')
                csv_players_path, csv_games_path, csv_odds_path = self.create_csv_from_api()
            else: 
                logging.info(f'No valid store option selected, currently only csv is supported')

            data_ingestion_artifact = DataIngestionArtifact(
                raw_players_path=csv_players_path,
                raw_games_path=csv_games_path, 
                raw_odds_path=csv_odds_path
            )
            
            return data_ingestion_artifact 
            
        except Exception as e:
            raise NbaException(e, sys)
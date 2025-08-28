import os 
import sys
import numpy as np
import pandas as pd

TRAINING_PIPELINE:str = "NbaGamePrediction"
ARTIFACT_DIR: str = "Artifacts"
DATA_FOLDER:str = "data"
STORE_OPTION:str = "csv"

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "TZIMAS"

# Where the api data is stored
DATA_INGESTION_GAMES_FILE_NAME: str = "games.csv"
DATA_INGESTION_PLAYERS_FILE_NAME: str = "players.csv"

DATA_INGESTION_DIR_NAME: str = "raw"
DATA_INGESTION_SEASONS: list = ['2018-19','2019-20','2020-21','2021-22','2022-23','2023-24']

# DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
# DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DIR_NAME: str = "validation"

DATA_TRANSFORMATION_FILE_NAME: str = "transformation.csv"
DATA_TRANSFORMATION_DIR_NAME: str = "transformed"
DATA_TRANSFORMATION_FORCE_REBUILD_CSV: bool = True

IMPORTANT_PLAYER_STATS: list = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
       'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']



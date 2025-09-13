import os 
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

TRAINING_PIPELINE:str = "NbaGamePrediction"
ARTIFACT_DIR: str = "Artifacts"
DATA_FOLDER:str = "data"
STORE_OPTION:str = "csv"

DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "TZIMAS"

# Where the api data is stored
DATA_INGESTION_GAMES_FILE_NAME: str = "games.csv"
DATA_INGESTION_PLAYERS_FILE_NAME: str = "players.csv"
DATA_INGESTION_ODDS_FILE_NAME: str = "odds.csv"

DATA_INGESTION_DIR_NAME: str = "raw"
DATA_INGESTION_SEASONS: list = ['2015-16','2016-17','2017-18','2018-19','2019-20','2020-21','2021-22','2022-23','2023-24']
# DATA_INGESTION_SEASONS: list = ['2019-20','2020-21','2021-22']

# DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
# DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DIR_NAME: str = "validation"

DATA_TRANSFORMATION_FILE_NAME: str = "transformation.csv"
DATA_TRANSFORMATION_DIR_NAME: str = "transformed"
DATA_TRANSFORMATION_FORCE_REBUILD_CSV: bool = False

IMPORTANT_PLAYER_STATS: list = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS']

IMPORTANT_ODDS_STATS: list = [ 'GAME_DATE', 'regular', 'HOME_ID', 'AWAY_ID', 'score_away', 'score_home', 'prob_away', 'prob_home']

MAPPING_DICT = {
    "cle": "CLE",
    "gs": "GSW",
    "ind": "IND",
    "wsh": "WAS",
    "orl": "ORL",
    "det": "DET",
    "bos": "BOS",
    "mem": "MEM",
    "dal": "DAL",
    "utah": "UTA",
    "sa": "SAS",
    "sac": "SAC",
    "phx": "PHX",
    "tor": "TOR",
    "okc": "OKC",
    "lal": "LAL",
    "phi": "PHI",
    "cha": "CHA",
    "mil": "MIL",
    "bkn": "BKN",
    "min": "MIN",
    "no": "NOP",
    "hou": "HOU",
    "ny": "NYK",
    "mia": "MIA",
    "chi": "CHI",
    "den": "DEN",
    "lac": "LAC",
    "por": "POR",
    "atl": "ATL"
}

TEST_SIZE: float = 0.2
SPLIT_FILE_NAME: str = "split.yaml"
RANDOM_STATE: int = 42
TRAINING_STATS_CATEGORIES: list = ['SEASON', 'ALL']

####### MLP ########
MLP_PLAYERS_ENCODER_HIDDEN_NUM_LAYERS_RANGE: list = [2, 3, 4]
MLP_PLAYERS_ENCODER_HIDDEN_SIZE: list = [64, 64, 8]
# originally is MLP_HEAD_LIST = [MLP_PLAYERS_ENCODER_HIDDEN_LIST[-1], 32, ...]
# BUT DUE TO AUTOMATION ISSUES WE HAVE TO PUT THE FIRST LAYER EQUAL TO THE LAST LAYER OF THE ENCODER * 2 (FOR BOTH TEAMS)
MLP_HEAD_LIST: list = [64, 16]
MLP_PLAYERS_ACTIVATION: nn.Module = nn.ReLU()
MLP_LEARNING_RATE_RANGE: list = [1e-5, 1e-4, 1e-3]
MLP_DROPOUT: float = 0.1
MLP_BATCH_SIZE_RANGE: list = [16, 32, 64]
MLP_NUM_EPOCHS: int = 2
MLP_IMPUTER: StandardScaler = StandardScaler()
MLP_FEATURE_SCALER_FILE_NAME: str = "mlp_feature_scaler.pkl"
MLP_TARGET_SCALER_FILE_NAME: str = "mlp_target_scaler.pkl"
######################

INFERENCE_STATS: list = [f'{cat}_{stat}' for cat in TRAINING_STATS_CATEGORIES for stat in IMPORTANT_PLAYER_STATS]
INFERENCE_DIR_NAME: str = "inference"
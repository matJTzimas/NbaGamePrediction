import os
import sys
from nba.constants import training
from datetime import datetime
from nba.exception.exception import NbaException



class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir = os.path.join(
            training.DATA_FOLDER, training.DATA_INGESTION_DIR_NAME
        )
        os.makedirs(self.data_ingestion_dir, exist_ok=True)
        self.games_file_path = os.path.join(
            self.data_ingestion_dir, training.DATA_INGESTION_GAMES_FILE_NAME
        )
        self.players_file_path = os.path.join(
            self.data_ingestion_dir, training.DATA_INGESTION_PLAYERS_FILE_NAME
        )
        self.seasons = training.DATA_INGESTION_SEASONS
        self.store_option = training.STORE_OPTION

class DataValidationConfig:
    def __init__(self):
        self.data_validation_dir = os.path.join(
            training.DATA_FOLDER, training.DATA_VALIDATION_DIR_NAME
        )
        os.makedirs(self.data_validation_dir, exist_ok=True)
        self.report_file_path = os.path.join(
            self.data_validation_dir, training.DATA_VALIDATION_REPORT_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_dir = os.path.join(
            training.DATA_FOLDER, training.DATA_TRANSFORMATION_DIR_NAME
        )
        os.makedirs(self.data_transformation_dir, exist_ok=True)

        self.transformed_file_path = os.path.join(
            self.data_transformation_dir, training.DATA_TRANSFORMATION_FILE_NAME
        )

        self.important_player_stats = training.IMPORTANT_PLAYER_STATS
        self.force_rebuild_csv = training.DATA_TRANSFORMATION_FORCE_REBUILD_CSV

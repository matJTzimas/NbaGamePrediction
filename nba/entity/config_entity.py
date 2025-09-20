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
        self.odds_file_path = os.path.join(
            self.data_ingestion_dir, training.DATA_INGESTION_ODDS_FILE_NAME
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
        self.important_odds_stats = training.IMPORTANT_ODDS_STATS
        self.mapping_dict = training.MAPPING_DICT
        self.test_size = training.TEST_SIZE

        self.split_file_path = os.path.join(
            self.data_transformation_dir, training.SPLIT_FILE_NAME
        )

class GeneralModelConfig:
    def __init__(self):
        self.split_file_name = training.SPLIT_FILE_NAME
        self.random_state = training.RANDOM_STATE
        self.training_stats = training.IMPORTANT_PLAYER_STATS
        self.training_stats_categories = training.TRAINING_STATS_CATEGORIES

class MLPConfig:
    def __init__(self):
        self.general_model_config = GeneralModelConfig()
        self.mlp_players_encoder_hidden_num_layers_range = training.MLP_PLAYERS_ENCODER_HIDDEN_NUM_LAYERS_RANGE
        self.mlp_players_encoder_hidden_size = training.MLP_PLAYERS_ENCODER_HIDDEN_SIZE
        self.mlp_learning_rate_range = training.MLP_LEARNING_RATE_RANGE
        self.mlp_batch_size_range = training.MLP_BATCH_SIZE_RANGE
        self.mlp_num_epochs = training.MLP_NUM_EPOCHS
        self.mlp_dropout = training.MLP_DROPOUT
        self.mlp_players_activation = training.MLP_PLAYERS_ACTIVATION
        self.imputer = training.MLP_IMPUTER
        self.mlp_head_list = training.MLP_HEAD_LIST
        self.mlp_dir = os.path.join(training.ARTIFACT_DIR, "MLP")
        os.makedirs(self.mlp_dir, exist_ok=True)

        self.model_name = "mlp_model.pth"
        self.model_path = os.path.join(self.mlp_dir, self.model_name)

        self.mlp_imputer = training.MLP_IMPUTER

        self.feature_scaler_path = os.path.join(self.mlp_dir, training.MLP_FEATURE_SCALER_FILE_NAME)
        self.target_scaler_path = os.path.join(self.mlp_dir, training.MLP_TARGET_SCALER_FILE_NAME)

class InferenceConfig:
    def __init__(self, model_name: str):
        self.inference_dir = os.path.join(training.ARTIFACT_DIR, training.INFERENCE_DIR_NAME)
        os.makedirs(self.inference_dir, exist_ok=True)
        self.important_player_stats = training.IMPORTANT_PLAYER_STATS
        self.daily_csv_file = os.path.join(self.inference_dir, f"predictions.csv")

        if model_name == "mlp":
            model_config = MLPConfig()


        self.model_path = model_config.model_path
        self.feature_scaler_path = model_config.feature_scaler_path
        self.inference_stats = training.INFERENCE_STATS






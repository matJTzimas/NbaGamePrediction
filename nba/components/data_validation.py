from nba.entity.config_entity import DataValidationConfig
from nba.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
import pandas as pd
import logging
import sys
from nba.exception.exception import NbaException
import yaml
import os 
from nba.utils.main_utils import write_yaml_file


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def _analyze_and_report(self, df: pd.DataFrame, file_label: str):
        # Count rows with missing values
        rows_with_missing = df.isnull().any(axis=1).sum()
        logging.info(f"{file_label}: Rows with missing values: {rows_with_missing}")

        # Drop rows with missing values
        df_clean = df.dropna()
        logging.info(f"{file_label}: Shape after dropping missing rows: {df_clean.shape}")

        # Collect column types
        columns_info = {col: str(dtype) for col, dtype in df_clean.dtypes.items()}
        return {
            "rows_with_missing": int(rows_with_missing),
            "final_shape": list(df_clean.shape),
            "columns": columns_info
        }

    def generate_report_yaml(self, players_df: pd.DataFrame, games_df: pd.DataFrame, path: str):
        report = {}
        report["players"] = self._analyze_and_report(players_df, "players")
        report["games"] = self._analyze_and_report(games_df, "games")

        write_yaml_file(file_path=path, content=report,replace=True)

        logging.info(f"Validation report saved to {path}")

    def initiate_data_validation(self):
        try:
            # Read the two dataframes
            players_df = pd.read_csv(self.data_ingestion_artifact.raw_players_path)
            games_df = pd.read_csv(self.data_ingestion_artifact.raw_games_path)

            logging.info("Data validation -> report.yaml.")
            # Analyze, clean, and report
            report_path = self.data_validation_config.report_file_path if hasattr(self.data_validation_config, "report_file_path") else "report.yaml"
            self.generate_report_yaml(players_df, games_df, report_path)

            # Clean dataframes (drop missing rows)
            players_clean = players_df.dropna()
            games_clean = games_df.dropna()
            logging.info(f'Data validation -> save new csv files.')


            # Save cleaned dataframes if they differ from original
            if players_clean.equals(players_df):
                logging.info("No missing values found in players data. Cleaned and original DataFrames are identical.")
                validated_players_path = self.data_ingestion_artifact.raw_players_path

            else:
                logging.info("Missing values were found and removed in players data. Cleaned and original DataFrames differ.")
                validated_players_path = os.path.join(self.data_validation_config.data_validation_dir, os.path.basename(self.data_ingestion_artifact.raw_players_path))
                players_clean.to_csv(validated_players_path, index=False)

            if games_clean.equals(games_df):
                logging.info("No missing values found in games data. Cleaned and original DataFrames are identical.")
                validated_games_path = self.data_ingestion_artifact.raw_games_path
            else:
                logging.info("Missing values were found and removed in games data. Cleaned and original DataFrames differ.")
                validated_games_path = os.path.join(self.data_validation_config.data_validation_dir, os.path.basename(self.data_ingestion_artifact.raw_games_path))
                games_clean.to_csv(validated_games_path, index=False)

            data_validation_artifact = DataValidationArtifact(
                validated_players_path=validated_players_path,
                validated_games_path=validated_games_path,
                report_file_path=report_path
            )

            return data_validation_artifact
        except Exception as e:
            raise NbaException(e, sys)
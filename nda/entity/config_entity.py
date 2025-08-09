import os
import sys
from nba.constatnts import training
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        self.pipeline_name: str = training.TRAINING_PIPELINE
        self.artifcact_name: str = training.ARTIFACT_DIR
        self.artifact_dir: str = os.path.join(self.artifcact_name, timestamp.strftime("%m_%d_%Y_%H_%M_%S"))

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILENAME
        )


if __name__ == "__main__": 
    training_pipeline_config = TrainingPipelineConfig()
    print(training_pipeline_config.pipeline_name)
    print(training_pipeline_config.artifact_dir)
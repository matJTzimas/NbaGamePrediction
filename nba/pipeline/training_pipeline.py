from nba.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from nba.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
import sys 
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from nba.components.data_ingestion import DataIngestion
from nba.components.data_validation import DataValidation
from nba.components.data_tranformation import DataTransformation
import os

class TrainingPipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        try:

            logging.info(f'Data ingestion started.')
            data_ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f'Data ingestion completed successfully.')
            
            logging.info(f'Data validation.')
            data_validation_config = DataValidationConfig()
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f'Data validation completed successfully.')

            logging.info(f'Data transformation.')
            data_transformation_config = DataTransformationConfig()
            data_data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_data_transformation.initialize_data_transformation()
            logging.info(f'Data transformation completed successfully.')

        except Exception as e:
            raise NbaException(e, sys)
        
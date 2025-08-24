from nba.entity.config_entity import DataIngestionConfig
import sys 
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from nba.components.data_ingestion import DataIngestion


if __name__ == "__main__": 
    try:
        data_ingestion_config = DataIngestionConfig()
        logging.info(f'Raw data folder {data_ingestion_config.data_ingestion_dir} created')
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
        logging.info(f'Data ingestion completed successfully.')
        
    except Exception as e:
        raise NbaException(e, sys)
        
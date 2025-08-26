
import sys 
from nba.pipeline.training_pipeline import TrainingPipeline
from nba.exception.exception import NbaException
from nba.logging.logger import logging

if __name__ == "__main__": 
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        logging.info(f'Training pipeline completed successfully.')
        
    except Exception as e:
        raise NbaException(e, sys)
        
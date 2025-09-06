import os 
import sys
import pandas as pd
from nba.entity.config_entity import *
from nba.entity.artifact_entity import *
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from tqdm import tqdm
import numpy as np
import yaml
from nba.models.datasets.dataset import NBADataset
import torch 
from torch.utils.data import Dataset, DataLoader


class ModelTrainer:
    def __init__(self, mlp_config: MLPConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.mlp_config = mlp_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NbaException(e, sys)


    def initiate_model_trainer(self):
        try:
            logging.info("Starting model training")
            train_dataset = NBADataset(mlp_config=self.mlp_config, data_transformation_artifact=self.data_transformation_artifact, split="train")

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            # Print a random batch to test functionality
            for batch in train_loader:
                print("Random batch:", batch['home_features'].size(), batch['away_features'].size())
                break  # Only print the first batch

            return None
        except Exception as e:
            raise NbaException(e, sys)

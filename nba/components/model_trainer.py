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
import torch.nn as nn
import torch.optim as optim
from nba.models.architectures.mlp import MLPNba
from nba.utils.ml_utils import train_one_epoch_mlp, evaluate_mlp
import optuna, mlflow, mlflow.pytorch
import dagshub
from dotenv import load_dotenv
load_dotenv()  # this reads .env and puts vars into os.environ
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


class ModelTrainer:
    def __init__(self, mlp_config: MLPConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.mlp_config = mlp_config
            self.data_transformation_artifact = data_transformation_artifact
            self.global_best_val_acc = 0.0
            self.global_best_model_state = None
        except Exception as e:
            raise NbaException(e, sys)


    def nba_mlp_objective(self, trial):
            logging.info("Starting NBA MLP training with mlflow and optuna")

            lr         = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [32, 64])
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)


            logging.info(f"Trial {trial.number}: lr={lr}, batch_size={batch_size}")
            with mlflow.start_run(nested=True, run_name=f"mlp_nba_{trial.number}_{datetime.now().strftime('%Y%m%d')}"):

                mlflow.log_params({
                        "lr": lr,
                        "batch_size": batch_size
                })

                train_dataset = NBADataset(mlp_config=self.mlp_config, data_transformation_artifact=self.data_transformation_artifact, split="train")
                val_dataset = NBADataset(mlp_config=self.mlp_config, data_transformation_artifact=self.data_transformation_artifact, split="test")

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

                model = MLPNba(player_input_dim=len(train_dataset.get_train_cols()), mlp_config=self.mlp_config)
                self.model = torch.compile(model).to(device)

                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
                best_val_acc = 0.0
                no_improvement_count = 0
                patience = 5
                min_delta = 0.001

                for epoch in range(1, self.mlp_config.mlp_num_epochs+1):

                    train_loss = train_one_epoch_mlp(self.model, train_loader, optimizer, device)
                    val_acc_stats = evaluate_mlp(self.model, val_loader, device)
                    val_acc = val_acc_stats['overall']
                    print(val_acc_stats)
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metrics(val_acc_stats, step=epoch)                       
                    # Report to Optuna & prune if needed
                    trial.report(val_acc, step=epoch)
                    if trial.should_prune():
                        mlflow.log_metric("pruned", 1)
                        raise optuna.TrialPruned()

                    logging.info(f"Trial {trial.number} Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

                    if val_acc > best_val_acc+min_delta:
                        best_val_acc = val_acc
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # After the loop ends (trial done):
                    if val_acc > self.global_best_val_acc:
                        self.global_best_val_acc = val_acc
                        self.global_best_model_state = self.model.state_dict()
                        self.best_val_stats = val_acc_stats
                    
                    if no_improvement_count >= patience:
                        logging.info(f"Early stopping at epoch {epoch} for trial {trial.number}")
                        break

                mlflow.log_metric("best_val_acc", best_val_acc)
                return best_val_acc


    def initiate_nbamlp_trainer(self):
        try:
            mlflow.set_experiment("NBA_MLP_Experiment")
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
            with mlflow.start_run(run_name="mlp_nba_main_run"):
                study = optuna.create_study(direction="maximize", pruner=pruner)
                study.optimize(self.nba_mlp_objective, n_trials=9, show_progress_bar=True)

                mlflow.log_metric("best_trial_acc_value", study.best_value)
                mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

                logging.info(f"Best accuracy: {study.best_value}")
            self.model.load_state_dict(self.global_best_model_state)
            torch.save(self.model, self.mlp_config.model_path)
            mlflow.log_artifact(self.mlp_config.model_path)
            mlflow.pytorch.log_model(self.model, artifact_path="mlp_nba_model", registered_model_name="testing_model")

            return None
        except Exception as e:
            raise NbaException(e, sys)

    def initiate_model_trainer(self):
        try:    

            logging.info("Initiating model trainer")
            return self.initiate_nbamlp_trainer()
        except Exception as e:
            raise NbaException(e, sys) 
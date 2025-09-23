import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nba.entity.config_entity import MLPConfig
from nba.utils.main_utils import *
from nba.entity.artifact_entity import DataTransformationArtifact
from sklearn.preprocessing import StandardScaler
from nba.utils.main_utils import Storage

class NBADataset(Dataset):
    def __init__(self, mlp_config: MLPConfig, data_transformation_artifact: DataTransformationArtifact, split: str = "train"):

        """
        Args:
        """

        self.split = split
        self.data_path = data_transformation_artifact.transformed_data_path
        self.mlp_config = mlp_config
        self.full_data = pd.read_csv(self.data_path, header=0)
        self.split_yaml = read_yaml_file(data_transformation_artifact.split_yaml_path)
        self.current_split_ids = list(self.split_yaml[self.split])
        self.data = self.full_data[self.full_data['GAME_ID'].isin(self.current_split_ids)].reset_index(drop=True)

        self.model_features = [f'{stat_category}_{col}' for col in self.mlp_config.general_model_config.training_stats 
                             for stat_category in self.mlp_config.general_model_config.training_stats_categories]
        self.auxiliary_target_features = [f'GAME_{col}' for col in self.mlp_config.general_model_config.training_stats]

        # we add a dnf player entry with all 0 stats before feeding it to the imputer
        zero_row = {col: 0 for col in self.data.columns}
        zero_row['GAME_ID'] = -1
        self.data = pd.concat([self.data, pd.DataFrame([zero_row])], ignore_index=True)

        self.storage = Storage(cloud_option=self.mlp_config.cloud_option)

        if self.split == "train":
            self.imputer_model = StandardScaler()
            self.imputer_target = StandardScaler()


            model_feature_scaler = self.imputer_model.fit(self.data.loc[:, self.model_features])
            target_feature_scaler = self.imputer_target.fit(self.data.loc[:, self.auxiliary_target_features])

            self.data[self.model_features] = pd.DataFrame(
                model_feature_scaler.transform(self.data[self.model_features]),
                columns=self.model_features,
                index=self.data.index
            )
            self.data[self.auxiliary_target_features] = pd.DataFrame(
                target_feature_scaler.transform(self.data[self.auxiliary_target_features]),
                columns=self.auxiliary_target_features,
                index=self.data.index
            )

            self.storage.save_object(file_path=self.mlp_config.feature_scaler_path, obj=model_feature_scaler)
            self.storage.save_object(file_path=self.mlp_config.target_scaler_path, obj=target_feature_scaler)
            
        else:
            model_feature_scaler = self.storage.load_object(file_path=self.mlp_config.feature_scaler_path)
            target_feature_scaler = self.storage.load_object(file_path=self.mlp_config.target_scaler_path)

            self.data[self.model_features] = model_feature_scaler.transform(self.data[self.model_features])
            self.data[self.auxiliary_target_features] = target_feature_scaler.transform(self.data[self.auxiliary_target_features])


        
    def __len__(self):
        return len(self.current_split_ids)

    def get_train_cols(self):
        return self.model_features

    def sort_and_fill(self, features: pd.DataFrame, by_col='ALL_PTS'):
        dnf_row = self.data[self.data['GAME_ID'] == -1].loc[:, features.columns]
        num_players = features.shape[0]
        features = features.sort_values(by=by_col, ascending=False).reset_index(drop=True)

        if num_players < 12:
            # If less than 9 players, fill with dnf_row until there are 9 rows
            num_missing = 12 - num_players
            dnf_rows = pd.concat([dnf_row] * num_missing, ignore_index=True)
            features = pd.concat([features, dnf_rows], ignore_index=True)
        # If more than 12, keep only top 12
        features = features.iloc[:12].reset_index(drop=True)
        return features


    def __getitem__(self, idx):

        current_game_id = self.current_split_ids[idx]
        current_game_data = self.data[self.data['GAME_ID'] == current_game_id].reset_index(drop=True)
        
        home_team_id = current_game_data.loc[0, 'HOME_ID']
        away_team_id = current_game_data.loc[0, 'AWAY_ID']

        home_features = self.sort_and_fill(current_game_data[current_game_data['TEAM_ID'] == home_team_id][self.model_features])
        away_features = self.sort_and_fill(current_game_data[current_game_data['TEAM_ID'] == away_team_id][self.model_features])

        home_axiliary_target = self.sort_and_fill(current_game_data[current_game_data['TEAM_ID'] == home_team_id][self.auxiliary_target_features], by_col='GAME_PTS')
        away_axiliary_target = self.sort_and_fill(current_game_data[current_game_data['TEAM_ID'] == away_team_id][self.auxiliary_target_features], by_col='GAME_PTS')

        game_info = {
            "score_home": torch.tensor(current_game_data.loc[0, 'score_home'], dtype=torch.float32),
            "score_away": torch.tensor(current_game_data.loc[0, 'score_away'], dtype=torch.float32),
            "prob_home": torch.tensor(current_game_data.loc[0, 'prob_home'], dtype=torch.float32),
            "prob_away": torch.tensor(current_game_data.loc[0, 'prob_away'], dtype=torch.float32),
            "home_features": torch.tensor(home_features.values, dtype=torch.float32),
            "away_features": torch.tensor(away_features.values, dtype=torch.float32),
            "home_auxiliary_target": torch.tensor(home_axiliary_target.values, dtype=torch.float32),
            "away_auxiliary_target": torch.tensor(away_axiliary_target.values, dtype=torch.float32)
        }

        return game_info


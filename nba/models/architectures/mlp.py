import torch
import torch.nn as nn
from nba.entity.config_entity import MLPConfig


class MLPNba(nn.Module):
    def __init__(self, player_input_dim, mlp_config: MLPConfig):
        super(MLPNba, self).__init__()

        self.player_input_dim = player_input_dim
        self.mlp_config = mlp_config

        layers = []
        for l in range(len(self.mlp_config.mlp_players_encoder_hidden_size)):
            if l == 0: 
                layers.append(nn.Linear(self.player_input_dim, self.mlp_config.mlp_players_encoder_hidden_size[l]))
                layers.append(self.mlp_config.mlp_players_activation)
                layers.append(nn.Dropout(self.mlp_config.mlp_dropout))
            else: 
                layers.append(nn.Linear(self.mlp_config.mlp_players_encoder_hidden_size[l-1], self.mlp_config.mlp_players_encoder_hidden_size[l]))
                layers.append(self.mlp_config.mlp_players_activation)
                layers.append(nn.Dropout(self.mlp_config.mlp_dropout))

        self.player_encoder = nn.Sequential(*layers)
        self.intermediate_dim = self.mlp_config.mlp_players_encoder_hidden_size[-1] * 2 * 12  # Because we concatenate home and away features

        head_layers = []
        for l in range(len(self.mlp_config.mlp_head_list)):
            if l == 0:
                head_layers.append(nn.Linear(self.intermediate_dim, self.mlp_config.mlp_head_list[l]))
                head_layers.append(self.mlp_config.mlp_players_activation)
                head_layers.append(nn.Dropout(self.mlp_config.mlp_dropout))
            else:
                head_layers.append(nn.Linear(self.mlp_config.mlp_head_list[l-1], self.mlp_config.mlp_head_list[l]))
                head_layers.append(self.mlp_config.mlp_players_activation)
                head_layers.append(nn.Dropout(self.mlp_config.mlp_dropout))
        else:
            head_layers.append(nn.Linear(self.mlp_config.mlp_head_list[-1], 1))
            head_layers.append(nn.Sigmoid())

        self.fc_head = nn.Sequential(*head_layers)

    def forward(self, home_features, away_features): 
        # Encode player features
        home_encoded = self.player_encoder(home_features)
        away_encoded = self.player_encoder(away_features)


        # # Aggregate player features (e.g., by averaging)
        home_agg = home_encoded.view(home_encoded.size(0), -1)
        away_agg = away_encoded.view(away_encoded.size(0), -1)

        # # Concatenate aggregated features
        combined = torch.cat((home_agg, away_agg), dim=1)

        # # Fully connected layers
        prob_win_home = self.fc_head(combined)

        return prob_win_home
from nba.entity.config_entity import InferenceConfig
import pandas as pd
import sys
import os 
from nba.components.model_inference import ModelInference

if __name__ == "__main__":
    model_inference = ModelInference()
    today_games = model_inference.inference_today_games()
    print(today_games)
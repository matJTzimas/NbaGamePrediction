import os
import sys

import pandas as pd

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scheduleleaguev2

from nba.logging.logger import logging
from nba.exception.exception import NbaException
from nba.utils.transformation_utils import add_prefix, home_away_id
from nba.entity.config_entity import InferenceConfig
from nba.utils.main_utils import load_object


class ModelInference:
    def __init__(self, inference_config: InferenceConfig):
        self.model = None
        self.inference_config = inference_config

        scheduled_games = scheduleleaguev2.ScheduleLeagueV2(
                league_id=LEAGUE,
                season="2025",
                timeout=30
            ).get_data_frames()[0].loc[scheduled_games['gameLabel']=='',:]

        self.scheduled_games = scheduled_games[['gameDate', 'gameId', 'homeTeam_teamId', 'homeTeam_teamTricode', 'awayTeam_teamId', 'awayTeam_teamTricode']]



    

    def inference_for_gamedate(self, game_date: str):
        return None
    def inference_for_gameid(self, game_id: str):
        return None
        
    @staticmethod 
    def get_roster(team_id):
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        roster_df = roster.get_data_frames()[0]  # the first DF = roster
        roster_df = roster_df.loc[:, ['PLAYER_ID', 'TeamID' ]]
        return roster_df
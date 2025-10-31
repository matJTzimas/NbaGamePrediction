import os
import sys
import time
import pandas as pd

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scheduleleaguev2
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import boxscoresummaryv2

from nba_api.stats.endpoints import TeamGameLogs
from nba.logging.logger import logging
from nba.exception.exception import NbaException
from nba.utils.transformation_utils import add_prefix, home_away_id
from nba.entity.config_entity import InferenceConfig
from nba.utils.main_utils import load_object
from dotenv import load_dotenv
load_dotenv()  # this reads .env and puts vars into os.environ
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/matJTzimas/NbaGamePrediction.mlflow")
import torch
import torch.nn as nn
from nba.utils.main_utils import Storage
# nba/components/model_inference.py
import time, random
from requests.exceptions import ReadTimeout, ConnectionError
from nba_api.stats.endpoints import scheduleleaguev2

def fetch_schedule(season="2025", max_attempts=6, per_req_timeout=20):
    delay = 2
    for attempt in range(1, max_attempts + 1):
        try:
            resp = scheduleleaguev2.ScheduleLeagueV2(
                league_id="00", season=season, timeout=per_req_timeout
            )
            return resp.get_data_frames()[0]
        except (ReadTimeout, ConnectionError) as e:
            if attempt == max_attempts:
                raise
            sleep_s = delay + random.uniform(0, 1)
            print(f"[schedule] attempt {attempt} failed ({type(e).__name__}); retrying in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            delay = min(delay * 2, 60)


def fetch_schedule_cdn():
    urls = [
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json",
        "https://data.nba.com/data/10s/prod/v1/calendar.json",
    ]
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com"})
    for u in urls:
        try:
            r = s.get(u, timeout=15); r.raise_for_status()
            data = r.json()
            games = data["leagueSchedule"]["gameDates"]
            rows = []
            for day in games:
                for g in day.get("games", []):
                    rows.append({
                        "gameId": g.get("gameId"),
                        "gameDate": day["gameDate"],
                        "home": g["homeTeam"]["teamTricode"],
                        "away": g["awayTeam"]["teamTricode"],
                        "status": g.get("gameStatus"),
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[cdn fallback] {u} failed: {e}")
    raise RuntimeError("All schedule sources failed")


class ModelInference:
    def __init__(self):
        self.model = None
        self.inference_config = InferenceConfig(model_name="mlp")

        self.storage = Storage(cloud_option=self.inference_config.cloud_option)

        # scheduled_games = scheduleleaguev2.ScheduleLeagueV2(league_id="00", season="2024", timeout=240).get_data_frames()[0]

        try: 
            scheduled_games = fetch_schedule(season="2025", max_attempts=6, per_req_timeout=30)
        except Exception:
            scheduled_games = fetch_schedule_cdn()

        
        scheduled_games = scheduled_games.loc[scheduled_games['gameLabel']=='',:]
        self.scheduled_games = scheduled_games[['gameDate', 'gameId', 'homeTeam_teamId', 'homeTeam_teamTricode', 'awayTeam_teamId', 'awayTeam_teamTricode']]
        self.scheduled_games.loc[:, 'gameDate'] = pd.to_datetime(self.scheduled_games['gameDate']).dt.date

        self.inference_stats_cols = self.inference_config.inference_stats
        
        self.imputer_scaler = load_object(file_path=self.inference_config.feature_scaler_path)

        model_uri = "models:/testing_model/5"
        self.model = mlflow.pytorch.load_model(model_uri=model_uri)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device).eval()
        logging.info(f"Loaded model for inference.")

        self.teams_df = pd.DataFrame(teams.get_teams())
        self.start_date = '2025-10-20'

    # def games_today(self):
    #     """
    #         return a [N,2] list of tuples with [home_team_id, away_team_id]
    #     """
    #     try:
    #         next_day = (pd.to_datetime('today')).date()
    #         today_games = self.scheduled_games.loc[self.scheduled_games['gameDate'] == next_day]
    #         # today_games = self.scheduled_games.loc[self.scheduled_games['gameDate'] == pd.to_datetime('2025-10-21').date()] # --- for testing ---
    #         if today_games.empty:
    #             logging.info("No games scheduled for today.")
    #             return []
    #         home_away_list = []
    #         for i in range(len(today_games)):
    #             game_id = int(today_games.iloc[i]['gameId'])
    #             home_team_id = int(today_games.iloc[i]['homeTeam_teamId'])
    #             away_team_id = int(today_games.iloc[i]['awayTeam_teamId'])
    #             game_date = today_games.iloc[i]['gameDate']
    #             home_away_list.append((home_team_id, away_team_id, game_id, game_date))
    #         return home_away_list
    #     except Exception as e:
    #         raise NbaException(e, sys)   

    def games_today(self):
        """
            return a [N,2] list of tuples with [home_team_id, away_team_id]
        """
        try:
            preds = self.storage.read_csv()
            today = (pd.to_datetime('today')).date()

            stored_game_ids = preds['GAME_ID'].astype(int).tolist() if preds is not None and not preds.empty else []
            # determine last prediction date and build list of calendar dates to check (exclusive of last_date, inclusive of today)
            if preds is not None and not preds.empty:
                last_date_ts = pd.to_datetime(preds['GAME_DATE']).max()
                last_date = last_date_ts.date()
            else:
                last_date = pd.to_datetime(self.start_date).date()

            start_check = last_date + pd.Timedelta(days=1)
            logging.info(f"Checking for games from {start_check} to {today}")
            if start_check > today:
                logging.info("No new dates between last predictions and today.")
                return []
        

            dates_to_check = pd.date_range(start=self.start_date, end=today, freq='D').date.tolist()
            logging.info(f"Checking the following dates for games: {dates_to_check}")
            # use the list in a for loop to collect all games between last_date and today
            home_away_list = []
            for check_date in dates_to_check:
                logging.info(f"Checking for games on {check_date}")
                day_games = self.scheduled_games.loc[self.scheduled_games['gameDate'] == check_date]
                logging.info(f"Found {len(day_games)} games.")
                if day_games.empty:
                    continue
                for i in range(len(day_games)):
                    if day_games.iloc[i]['gameId'] in stored_game_ids:
                        continue
                    game_id = int(day_games.iloc[i]['gameId'])
                    home_team_id = int(day_games.iloc[i]['homeTeam_teamId'])
                    away_team_id = int(day_games.iloc[i]['awayTeam_teamId'])
                    game_date = day_games.iloc[i]['gameDate']
                    home_away_list.append((home_team_id, away_team_id, game_id, game_date))

            if not home_away_list:
                logging.info("No games scheduled for the dates checked.")
                return []

            # return the aggregated list of games for downstream inference
            return home_away_list
        except Exception as e:
            raise NbaException(e, sys) 


    def inference_today_games(self):

        for game_info in self.games_today():
            winners = [] 

            home_team_id, away_team_id, game_id, game_date = game_info
            
            logging.info(f"Inference: Getting rosters for game {game_id} between {home_team_id} and {away_team_id}")
            home_roster = self.get_roster(home_team_id)
            away_roster = self.get_roster(away_team_id)

            home_roster = self.drop_inactive_players_roster(game_id, home_roster)
            away_roster = self.drop_inactive_players_roster(game_id, away_roster)

            home_stats = self.safe_players_stats(home_roster, self.inference_config.important_player_stats)
            away_stats = self.safe_players_stats(away_roster, self.inference_config.important_player_stats)

            home_stats = self.sort_and_pad_roster(home_stats)
            away_stats = self.sort_and_pad_roster(away_stats)

            home_stats = self.imputer_scaler.transform(home_stats[self.imputer_scaler.feature_names_in_])
            away_stats = self.imputer_scaler.transform(away_stats[self.imputer_scaler.feature_names_in_])

            logging.info(f"Completed fetching and processing rosters for game {game_id} between {home_team_id} and {away_team_id}")
            home_stats = torch.tensor(home_stats, dtype=torch.float32).unsqueeze(0).to(self.device)
            away_stats = torch.tensor(away_stats, dtype=torch.float32).unsqueeze(0).to(self.device)

            home_prob = self.model(home_stats, away_stats)

            if home_prob.item() < 0.5:
                winners.append([game_id, game_date, home_team_id, away_team_id, home_prob.item(), "AWAY"])
            else:
                winners.append([game_id, game_date, home_team_id, away_team_id, home_prob.item(), "HOME"])

            self.save_daily_predictions(winners)
            self.update_actuals()
        else:
            logging.info("No games today to run inference for.")
            return []

        return winners

    def update_actuals(self):

        preds = self.storage.read_csv()
        if preds is None or preds.empty:
            logging.info("No predictions to update actuals for.")
            return

        game_logs = TeamGameLogs(season_nullable="2025-26",league_id_nullable='00',timeout=120).get_data_frames()[0]
        game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE']).dt.date

        game_logs = game_logs[game_logs['MATCHUP'].str.contains(" vs. ", na=False)].reset_index(drop=True)
        game_logs['GAME_ID'] = game_logs['GAME_ID'].astype(int)

        for  i in range(len(preds)):
            has_result = True if preds.loc[i, "ACTUAL"] != "-" else False
            if has_result:
                print('Skipping, already has result')
                continue
            game_id = preds.loc[i,'GAME_ID']
            current_log = game_logs[game_logs['GAME_ID'] == game_id]

            if not current_log.empty:
                home_result = current_log['WL'].item()
                if home_result == 'W':
                    preds.loc[i,'ACTUAL'] = "HOME"
                elif home_result == 'L':
                    preds.loc[i,'ACTUAL'] = "AWAY"

            if preds.loc[i, 'ACTUAL'] == preds.loc[i,'WINNER PRED']:
                preds.loc[i,'RESULT'] = True
            else:
                preds.loc[i,'RESULT'] = False

        # preds.to_csv(self.inference_config.daily_csv_file, index=False)
        self.storage.to_csv(preds)

    def sort_and_pad_roster(self, roster_df):
        """
        Sorts the roster by ALL_PTS descending, keeps top 12, and pads with zero rows if less than 12.
        Returns the new roster DataFrame.
        """
        sorted_roster = roster_df.sort_values(by="ALL_PTS", ascending=False).reset_index(drop=True)
        num_players = len(sorted_roster)
        if num_players < 12:
            # Create zero rows with same columns
            zero_rows = pd.DataFrame(0, index=range(12 - num_players), columns=sorted_roster.columns)
            sorted_roster = pd.concat([sorted_roster, zero_rows], ignore_index=True)
        else:
            sorted_roster = sorted_roster.iloc[:12].reset_index(drop=True)
        return sorted_roster

    def save_daily_predictions(self, winners):
        """
        Saves the daily predictions to a CSV file.
        """
        try:
            if not winners:
                logging.info("No games today to save predictions for.")
                return

            df_winners = pd.DataFrame(winners, columns=['GAME_ID','GAME_DATE', 'HOME_ID', 'AWAY_ID', 'PROB_HOME_WIN', 'WINNER PRED'])
            df_winners['GAME_ID'] = df_winners['GAME_ID'].astype(int)
            df_winners['PROB_AWAY_WIN'] = 1 - df_winners['PROB_HOME_WIN']
            df_winners['HOME_ABBR'] = df_winners['HOME_ID'].map(self.teams_df.set_index('id')['abbreviation'])
            df_winners['AWAY_ABBR'] = df_winners['AWAY_ID'].map(self.teams_df.set_index('id')['abbreviation'])
            df_winners['ACTUAL'] = ["-" for _ in range(len(df_winners))]
            df_winners['RESULT'] = ["-" for _ in range(len(df_winners))]
            df_winners = df_winners[['GAME_ID','GAME_DATE', 'HOME_ID', 'HOME_ABBR', 'AWAY_ID', 'AWAY_ABBR', 'PROB_HOME_WIN', 'PROB_AWAY_WIN', 'WINNER PRED', 'ACTUAL', 'RESULT']]

            # if os.path.exists(self.inference_config.daily_csv_file) and os.path.getsize(self.inference_config.daily_csv_file) > 0:
            #     existing_df = pd.read_csv(self.inference_config.daily_csv_file)
            #     df_winners = pd.concat([existing_df, df_winners], ignore_index=True)

            existing_df = self.storage.read_csv()
            if existing_df is not None:
                df_winners = pd.concat([existing_df, df_winners], ignore_index=True)

            

            self.storage.to_csv(df_winners)


        except Exception as e:
            raise NbaException(e, sys)

    def safe_players_stats(self, roster, important_stats):
        players_stats = []
        for player_id in roster['PLAYER_ID']:
            try:
                stats = self.get_player_stats(player_id, important_stats)
                players_stats.append(stats)
            except Exception as e:
                logging.warning(f"Could not fetch stats for player {player_id}: {e}")
        if players_stats:
            return pd.concat(players_stats, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no stats were fetched

    @staticmethod
    def drop_inactive_players_roster(game_id, roster):
        """
        Returns a DataFrame of active players for the given game_id.
        """
        try:
            bs = boxscoresummaryv2.BoxScoreSummaryV2(game_id=str(game_id).zfill(10)).inactive_players.get_data_frame()
            if bs.empty:
                logging.info("No inactive players today.")
                return roster

            roster_df = bs.loc[:, ['PLAYER_ID', 'TEAM_ID']]
            roster = roster[~roster['PLAYER_ID'].isin(roster_df['PLAYER_ID'])]
            return roster
        except Exception as e:
            raise NbaException(e, sys)


    @staticmethod 
    def get_roster(team_id):
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        roster_df = roster.get_data_frames()[0]  # the first DF = roster
        roster_df = roster_df.loc[:, ['PLAYER_ID', 'TeamID' ]]
        return roster_df

    @staticmethod
    def get_player_stats(player_id: int, important_stats: list, timeout: int = 240):
        """
        Returns a dict of DataFrames for the player's career:
        """

        # Gentle pause helps avoid throttling if calling repeatedly
        time.sleep(1)
        pcs = playercareerstats.PlayerCareerStats(
            player_id=player_id,
            per_mode36="PerGame",   # or 'Per36', 'Totals'
            timeout=timeout
        )
        dfs = pcs.get_data_frames()[0]

        tables = {
            t["name"]: pd.DataFrame(t["rowSet"], columns=t["headers"])
            for t in pcs.get_dict()['resultSets']
        }
        # Common tables of interest
        out = {
            "season_totals_regular": tables.get("SeasonTotalsRegularSeason"),
            "career_totals_regular": tables.get("CareerTotalsRegularSeason")
        }

        season = out["season_totals_regular"].iloc[[-1], :].reset_index(drop=True)
        career = out["career_totals_regular"][important_stats].reset_index(drop=True)
        season = add_prefix(season, "SEASON", important_stats)
        career = add_prefix(career, "ALL", important_stats)

        final_player = pd.concat([season, career], axis=1)

        return final_player


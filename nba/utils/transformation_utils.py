import pandas as pd
from nba_api.stats.static import teams
import polars as pl
import numpy as np

def home_away_id(games_df, game_id):
    """
    Returns the home and away team IDs based on the game matchup string.

    Parameters:
    - games_df (pd.DataFrame): DataFrame containing game details with a 'MATCHUP' column.
    - game_id (int or str): The unique GAME_ID to find the matchup.

    Returns:
    - tuple: (home_team_id, away_team_id)
    """
    df_teams = pd.DataFrame(teams.get_teams())
    mat = games_df.loc[games_df['GAME_ID'] == game_id, 'MATCHUP'].iloc[0]


    if '@' in mat:
        teams_abb = mat.split(' @ ')
        home_team_id = df_teams.loc[df_teams['abbreviation'] == teams_abb[1], 'id'].iloc[0]
        away_team_id = df_teams.loc[df_teams['abbreviation'] == teams_abb[0], 'id'].iloc[0]
    else:
        teams_abb = mat.split(' vs. ')
        home_team_id = df_teams.loc[df_teams['abbreviation'] == teams_abb[0], 'id'].iloc[0]
        away_team_id = df_teams.loc[df_teams['abbreviation'] == teams_abb[1], 'id'].iloc[0]

    return home_team_id, away_team_id



def add_prefix(df: pd.DataFrame, prefix: str, included_columns: list = []) -> pd.DataFrame:
    try:
        columns = [col for col in df.columns if col in included_columns]
        df.rename(columns={col: f"{prefix}_{col}" for col in columns}, inplace=True)
        return df
    except Exception as e:
        raise NbaException(e, sys)

def moneyline_to_probability(row):
    moneyline_home = row['moneyline_home']
    moneyline_away = row['moneyline_away']

    def convert_moneyline_to_probability(moneyline):
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return -moneyline / (-moneyline + 100)

    if pd.isna(moneyline_home) or pd.isna(moneyline_away):
        prob_home, prob_away = game_score_to_prob(np.array([row['score_home'],row['score_away']]))
        return prob_home, prob_away

    prob_home = convert_moneyline_to_probability(moneyline_home)
    prob_away = convert_moneyline_to_probability(moneyline_away)
    total_prob = prob_home + prob_away

    return prob_home / total_prob, prob_away / total_prob

    

def game_score_to_prob(x):
    """Compute softmax values for each sets of scores in x."""
    x = x/20
    e_x = np.exp(x - np.max(x))
    res = e_x / e_x.sum()
    return (res[0].item(),res[1].item())
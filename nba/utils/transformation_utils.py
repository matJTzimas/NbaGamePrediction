import pandas as pd
from nba_api.stats.static import teams

def home_away_id(games_df, game_id):
    """"
    Returns the home and away team IDs based on the game matchup string.

    Parameters:
    - games_df (pd.DataFrame): DataFrame containing game details with a 'MATCHUP' column.
    - game_id (int or str): The unique GAME_ID to find the matchup.

    Returns:
    - tuple: (home_team_id, away_team_id)
    """
    df_teams = pd.DataFrame(teams.get_teams())
    mat = games_df[games_df['GAME_ID']==game_id]['MATCHUP'].iloc[0]
    if '@' in mat:
        teams_abb = mat.split(' @ ')
        home_team_id = df_teams[df_teams['abbreviation']==teams_abb[1]]['id'].item()
        away_team_id = df_teams[df_teams['abbreviation']==teams_abb[0]]['id'].item()
    else : 
        teams_abb = mat.split('vs.')
        home_team_id = df_teams[df_teams['abbreviation']==teams_abb[0]]['id'].item()
        away_team_id = df_teams[df_teams['abbreviation']==teams_abb[1]]['id'].item()

    return home_team_id, away_team_id


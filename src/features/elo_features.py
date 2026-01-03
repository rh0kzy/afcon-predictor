import pandas as pd
import numpy as np

def calculate_elo(df):
    """
    Calculates dynamic Elo ratings for each team.
    """
    # Initialize Elo ratings
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    elo_ratings = {team: 1500 for team in teams}
    
    home_elo_before = []
    away_elo_before = []
    elo_diff = []
    
    # Sort by date to ensure chronological calculation
    df = df.sort_values('date')
    
    # Base K-factor
    BASE_K = 10
    
    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get current ratings
        r_home = elo_ratings[home_team]
        r_away = elo_ratings[away_team]
        
        home_elo_before.append(r_home)
        away_elo_before.append(r_away)
        elo_diff.append(r_home - r_away)
        
        # Calculate expected scores
        e_home = 1 / (1 + 10**((r_away - r_home) / 400))
        e_away = 1 - e_home
        
        # Actual scores
        if row['home_score'] > row['away_score']:
            s_home, s_away = 1, 0
        elif row['home_score'] < row['away_score']:
            s_home, s_away = 0, 1
        else:
            s_home, s_away = 0.5, 0.5
            
        # K-factor adjusted by tournament weight
        # tournament_weight is already in the df from context_features
        k = BASE_K * row.get('tournament_weight', 2)
        
        # Update ratings
        elo_ratings[home_team] = r_home + k * (s_home - e_home)
        elo_ratings[away_team] = r_away + k * (s_away - e_away)
        
    df['home_elo'] = home_elo_before
    df['away_elo'] = away_elo_before
    df['elo_diff'] = elo_diff
    
    return df

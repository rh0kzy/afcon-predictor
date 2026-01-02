import pandas as pd

def calculate_context_features(df):
    # Home advantage flag (1 if home team is playing in their own country)
    df['is_home_adv'] = (df['country'] == df['home_team']).astype(int)
    
    # Neutral venue flag
    df['is_neutral'] = df['neutral'].astype(int)
    
    # Tournament importance weighting
    tournament_weights = {
        'FIFA World Cup': 10,
        'African Cup of Nations': 8,
        'FIFA World Cup qualification': 7,
        'African Cup of Nations qualification': 6,
        'Confederations Cup': 5,
        'Arab Cup': 4,
        'Gold Cup': 4,
        'COSAFA Cup': 3,
        'CECAFA Cup': 3,
        'Friendly': 1
    }
    
    df['tournament_weight'] = df['tournament'].map(tournament_weights).fillna(2)
    
    return df

import pandas as pd

def calculate_context_features(df):
    # Home advantage flag (1 if home team is playing in their own country)
    df['is_home_adv'] = (df['country'] == df['home_team']).astype(int)
    
    # Neutral venue flag
    df['is_neutral'] = df['neutral'].astype(int)
    
    return df

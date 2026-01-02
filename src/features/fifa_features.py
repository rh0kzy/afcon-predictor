import pandas as pd

def calculate_fifa_features(df):
    # Rank difference
    df['rank_diff'] = df['home_rank'] - df['away_rank']
    
    # Point difference
    df['point_diff'] = df['home_points'] - df['away_points']
    
    return df

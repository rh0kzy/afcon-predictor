import pandas as pd

def calculate_fifa_features(df):
    # Rank difference
    df['rank_diff'] = df['home_rank'] - df['away_rank']
    
    # Point difference
    df['point_diff'] = df['home_points'] - df['away_points']
    
    # Rank change (momentum)
    df['home_rank_momentum'] = df['home_rank_change']
    df['away_rank_momentum'] = df['away_rank_change']
    
    return df

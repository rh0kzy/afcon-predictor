import pandas as pd
import numpy as np
import os

def calculate_squad_features(df, squad_values_path="data/raw/squad_values.csv"):
    """
    Integrates market value and squad quality data into the features table.
    """
    if not os.path.exists(squad_values_path):
        print(f"Warning: {squad_values_path} not found. Skipping squad features.")
        df['home_squad_value'] = 50.0 # Default median
        df['away_squad_value'] = 50.0
        df['value_diff'] = 0.0
        df['value_ratio'] = 1.0
        return df

    squad_df = pd.read_csv(squad_values_path)
    value_map = dict(zip(squad_df['team'], squad_df['market_value_mln']))
    quality_map = dict(zip(squad_df['team'], squad_df['top_five_league_players']))

    # Map values
    df['home_squad_value'] = df['home_team'].map(value_map).fillna(10.0) # Fill unknown with low value
    df['away_squad_value'] = df['away_team'].map(value_map).fillna(10.0)
    
    df['home_squad_quality'] = df['home_team'].map(quality_map).fillna(0)
    df['away_squad_quality'] = df['away_team'].map(quality_map).fillna(0)

    # Derived features
    # Use log to handle the wide variance in market values (Morocco vs Sudan)
    df['log_home_value'] = np.log1p(df['home_squad_value'])
    df['log_away_value'] = np.log1p(df['away_squad_value'])
    
    df['value_diff'] = df['home_squad_value'] - df['away_squad_value']
    df['value_ratio'] = df['home_squad_value'] / (df['away_squad_value'] + 1e-6)
    
    df['quality_diff'] = df['home_squad_quality'] - df['away_squad_quality']

    return df

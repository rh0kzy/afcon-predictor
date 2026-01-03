import pandas as pd
import numpy as np
from src.utils.constants import ROLLING_WINDOW

def calculate_form(df):
    # Create a long format to calculate rolling stats per team
    home = df[['date', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'}
    )
    home['is_home'] = 1
    
    away = df[['date', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'}
    )
    away['is_home'] = 0
    
    team_stats = pd.concat([home, away]).sort_values(['team', 'date'])
    
    # Calculate points
    team_stats['points'] = np.where(team_stats['goals_for'] > team_stats['goals_against'], 3,
                                   np.where(team_stats['goals_for'] == team_stats['goals_against'], 1, 0))
    
    # Rolling averages
    team_stats['rolling_points'] = team_stats.groupby('team')['points'].transform(lambda x: x.shift().rolling(ROLLING_WINDOW, min_periods=1).mean())
    
    # Weighted rolling points (Time Decay)
    def weighted_mean(x):
        if len(x) == 0: return np.nan
        weights = np.arange(1, len(x) + 1)
        return np.sum(x * weights) / np.sum(weights)
    
    team_stats['weighted_points'] = team_stats.groupby('team')['points'].transform(
        lambda x: x.shift().rolling(ROLLING_WINDOW, min_periods=1).apply(weighted_mean, raw=True)
    )
    
    team_stats['rolling_goals_for'] = team_stats.groupby('team')['goals_for'].transform(lambda x: x.shift().rolling(ROLLING_WINDOW, min_periods=1).mean())
    team_stats['rolling_goals_against'] = team_stats.groupby('team')['goals_against'].transform(lambda x: x.shift().rolling(ROLLING_WINDOW, min_periods=1).mean())
    team_stats['rolling_goal_diff'] = team_stats['rolling_goals_for'] - team_stats['rolling_goals_against']
    
    # Merge back to original df
    df = df.merge(team_stats[team_stats['is_home'] == 1][['date', 'team', 'rolling_points', 'weighted_points', 'rolling_goals_for', 'rolling_goals_against', 'rolling_goal_diff']], 
                  left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').drop(columns='team')
    df = df.rename(columns={'rolling_points': 'home_form', 'weighted_points': 'home_weighted_form', 'rolling_goals_for': 'home_avg_goals_for', 'rolling_goals_against': 'home_avg_goals_against', 'rolling_goal_diff': 'home_goal_diff_form'})
    
    df = df.merge(team_stats[team_stats['is_home'] == 0][['date', 'team', 'rolling_points', 'weighted_points', 'rolling_goals_for', 'rolling_goals_against', 'rolling_goal_diff']], 
                  left_on=['date', 'away_team'], right_on=['date', 'team'], how='left').drop(columns='team')
    df = df.rename(columns={'rolling_points': 'away_form', 'weighted_points': 'away_weighted_form', 'rolling_goals_for': 'away_avg_goals_for', 'rolling_goals_against': 'away_avg_goals_against', 'rolling_goal_diff': 'away_goal_diff_form'})
    
    return df

import pandas as pd
from src.config import MATCHES_CLEANED, FIFA_CLEANED, FEATURES_TABLE
from src.utils.team_name_map import normalize_team_name

def merge_fifa_rankings():
    matches = pd.read_csv(MATCHES_CLEANED)
    fifa = pd.read_csv(FIFA_CLEANED)
    
    matches['date'] = pd.to_datetime(matches['date'])
    fifa['rank_date'] = pd.to_datetime(fifa['rank_date'])
    
    # Sort for merge_asof
    matches = matches.sort_values('date')
    fifa = fifa.sort_values('rank_date')
    
    # Merge for home team
    matches = pd.merge_asof(
        matches, 
        fifa[['rank_date', 'country_full', 'rank', 'total_points', 'rank_change']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='home_team', 
        right_by='country_full',
        direction='backward'
    ).rename(columns={'rank': 'home_rank', 'total_points': 'home_points', 'rank_change': 'home_rank_change'}).drop(columns=['rank_date', 'country_full'])
    
    # Merge for away team
    matches = pd.merge_asof(
        matches, 
        fifa[['rank_date', 'country_full', 'rank', 'total_points', 'rank_change']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='away_team', 
        right_by='country_full',
        direction='backward'
    ).rename(columns={'rank': 'away_rank', 'total_points': 'away_points', 'rank_change': 'away_rank_change'}).drop(columns=['rank_date', 'country_full'])
    
    # Note: We'll save this as a temporary step or just return it for the next pipeline stage
    return matches

if __name__ == "__main__":
    merge_fifa_rankings()

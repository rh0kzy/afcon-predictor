import pandas as pd
from src.config import MATCHES_CLEANED, FIFA_RANKING_RAW, FIFA_CLEANED
from src.utils.team_name_map import normalize_team_name

def merge_fifa_rankings():
    matches = pd.read_csv(MATCHES_CLEANED)
    fifa = pd.read_csv(FIFA_RANKING_RAW)
    
    matches['date'] = pd.to_datetime(matches['date'])
    fifa['rank_date'] = pd.to_datetime(fifa['rank_date'])
    fifa['country_full'] = fifa['country_full'].apply(normalize_team_name)
    
    # Sort for merge_asof
    matches = matches.sort_values('date')
    fifa = fifa.sort_values('rank_date')
    
    # Merge for home team
    matches = pd.merge_asof(
        matches, 
        fifa[['rank_date', 'country_full', 'rank', 'total_points']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='home_team', 
        right_by='country_full',
        direction='backward'
    ).rename(columns={'rank': 'home_rank', 'total_points': 'home_points'}).drop(columns=['rank_date', 'country_full'])
    
    # Merge for away team
    matches = pd.merge_asof(
        matches, 
        fifa[['rank_date', 'country_full', 'rank', 'total_points']], 
        left_on='date', 
        right_on='rank_date', 
        left_by='away_team', 
        right_by='country_full',
        direction='backward'
    ).rename(columns={'rank': 'away_rank', 'total_points': 'away_points'}).drop(columns=['rank_date', 'country_full'])
    
    matches.to_csv(FIFA_CLEANED, index=False)
    print(f"Merged FIFA data saved to {FIFA_CLEANED}")
    return matches

if __name__ == "__main__":
    merge_fifa_rankings()

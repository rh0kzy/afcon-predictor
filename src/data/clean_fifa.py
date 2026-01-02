import pandas as pd
from src.config import FIFA_RANKING_RAW, FIFA_CLEANED
from src.utils.team_name_map import normalize_team_name

def clean_fifa_rankings():
    df = pd.read_csv(FIFA_RANKING_RAW)
    
    # Parse rank_date to datetime
    df['rank_date'] = pd.to_datetime(df['rank_date'])
    
    # Normalize country names
    df['country_full'] = df['country_full'].apply(normalize_team_name)
    
    # Remove duplicates (same country, same date)
    df = df.drop_duplicates(subset=['rank_date', 'country_full'])
    
    # Sort by rank_date
    df = df.sort_values('rank_date')
    
    # Keep relevant columns only
    relevant_cols = ['rank_date', 'country_full', 'rank', 'total_points', 'previous_points', 'rank_change', 'confederation']
    df = df[relevant_cols]
    
    # Save output
    df.to_csv(FIFA_CLEANED, index=False)
    print(f"Cleaned FIFA rankings saved to {FIFA_CLEANED}")
    return df

if __name__ == "__main__":
    clean_fifa_rankings()

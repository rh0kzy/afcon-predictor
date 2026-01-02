import pandas as pd
from src.config import MATCHES_RAW, MATCHES_CLEANED
from src.utils.constants import CAF_TEAMS
from src.utils.team_name_map import normalize_team_name

def clean_matches():
    df = pd.read_csv(MATCHES_RAW)
    
    # Normalize team names
    df['home_team'] = df['home_team'].apply(normalize_team_name)
    df['away_team'] = df['away_team'].apply(normalize_team_name)
    
    # Filter for CAF teams (at least one team must be in CAF)
    df = df[df['home_team'].isin(CAF_TEAMS) | df['away_team'].isin(CAF_TEAMS)]
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop rows with missing scores
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Save cleaned data
    df.to_csv(MATCHES_CLEANED, index=False)
    print(f"Cleaned matches saved to {MATCHES_CLEANED}")
    return df

if __name__ == "__main__":
    clean_matches()

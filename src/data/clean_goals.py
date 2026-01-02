import pandas as pd
from src.config import GOALS_RAW, PROCESSED_DATA_DIR

def clean_goals():
    df = pd.read_csv(GOALS_RAW)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate goals per team per match
    goals_per_match = df.groupby(['date', 'home_team', 'away_team', 'team']).size().reset_index(name='goals_scored')
    
    # Save cleaned data
    output_path = PROCESSED_DATA_DIR / "goals_cleaned.csv"
    goals_per_match.to_csv(output_path, index=False)
    print(f"Cleaned goals saved to {output_path}")
    return goals_per_match

if __name__ == "__main__":
    clean_goals()

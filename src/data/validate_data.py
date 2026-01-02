import pandas as pd
from src.config import MATCHES_CLEANED, FIFA_CLEANED, GOALS_RAW
from src.utils.constants import CAF_TEAMS

def validate_data():
    print("--- Starting Data Validation ---")
    
    matches = pd.read_csv(MATCHES_CLEANED)
    fifa = pd.read_csv(FIFA_CLEANED)
    
    # 1. Verify team names match across datasets
    match_teams = set(matches['home_team']).union(set(matches['away_team']))
    fifa_teams = set(fifa['country_full'])
    
    missing_in_fifa = match_teams - fifa_teams
    print(f"Teams in matches but missing in FIFA rankings: {len(missing_in_fifa)}")
    if len(missing_in_fifa) > 0:
        print(f"Example missing teams: {list(missing_in_fifa)[:5]}")
    
    # 2. Verify date ranges overlap
    matches['date'] = pd.to_datetime(matches['date'])
    fifa['rank_date'] = pd.to_datetime(fifa['rank_date'])
    
    print(f"Matches date range: {matches['date'].min()} to {matches['date'].max()}")
    print(f"FIFA rankings date range: {fifa['rank_date'].min()} to {fifa['rank_date'].max()}")
    
    # 3. Check number of matches per year
    matches_per_year = matches['date'].dt.year.value_counts().sort_index()
    print("\nMatches per year (last 5 years):")
    print(matches_per_year.tail(5))
    
    # 4. Check for potential leakage (matches before FIFA rankings started)
    earliest_fifa = fifa['rank_date'].min()
    matches_before_fifa = matches[matches['date'] < earliest_fifa]
    print(f"\nMatches before earliest FIFA ranking ({earliest_fifa}): {len(matches_before_fifa)}")
    
    # 5. Check number of CAF teams covered
    caf_teams_in_data = match_teams.intersection(set(CAF_TEAMS))
    print(f"CAF teams covered in match data: {len(caf_teams_in_data)}/{len(CAF_TEAMS)}")
    
    print("--- Data Validation Completed ---")

if __name__ == "__main__":
    validate_data()

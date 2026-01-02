import pandas as pd
from src.config import MATCHES_RAW, FIFA_RANKING_RAW, GOALS_RAW

def load_raw_matches():
    return pd.read_csv(MATCHES_RAW)

def load_raw_fifa_rankings():
    return pd.read_csv(FIFA_RANKING_RAW)

def load_raw_goals():
    return pd.read_csv(GOALS_RAW)

if __name__ == "__main__":
    matches = load_raw_matches()
    print(f"Loaded {len(matches)} matches.")
    fifa = load_raw_fifa_rankings()
    print(f"Loaded {len(fifa)} FIFA rankings.")
    goals = load_raw_goals()
    print(f"Loaded {len(goals)} goal records.")

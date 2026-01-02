import pandas as pd
import numpy as np
import pickle
from src.config import XGB_MODEL_PATH, FEATURES_TABLE
from src.features.form_features import calculate_form
from src.features.h2h_features import calculate_h2h
from src.features.fifa_features import calculate_fifa_features
from src.features.context_features import calculate_context_features

def get_team_features(team, historical_df):
    # Get the latest features for a team
    team_latest = historical_df[(historical_df['home_team'] == team) | (historical_df['away_team'] == team)].sort_values('date').tail(1)
    if len(team_latest) == 0:
        return None
    
    if team_latest.iloc[0]['home_team'] == team:
        return {
            'rank': team_latest.iloc[0]['home_rank'],
            'points': team_latest.iloc[0]['home_points'],
            'form': team_latest.iloc[0]['home_form'],
            'goal_diff_form': team_latest.iloc[0]['home_goal_diff_form'],
            'rank_momentum': team_latest.iloc[0]['home_rank_momentum']
        }
    else:
        return {
            'rank': team_latest.iloc[0]['away_rank'],
            'points': team_latest.iloc[0]['away_points'],
            'form': team_latest.iloc[0]['away_form'],
            'goal_diff_form': team_latest.iloc[0]['away_goal_diff_form'],
            'rank_momentum': team_latest.iloc[0]['away_rank_momentum']
        }

def get_match_probs_fast(home_team, away_team, model, historical_df):
    home_feats = get_team_features(home_team, historical_df)
    away_feats = get_team_features(away_team, historical_df)
    
    if not home_feats or not away_feats:
        return [0.33, 0.34, 0.33]
    
    # Construct feature vector
    match_feats = {
        'home_rank': home_feats['rank'],
        'away_rank': away_feats['rank'],
        'home_points': home_feats['points'],
        'away_points': away_feats['points'],
        'home_form': home_feats['form'],
        'away_form': away_feats['form'],
        'home_goal_diff_form': home_feats['goal_diff_form'],
        'away_goal_diff_form': away_feats['goal_diff_form'],
        'rank_diff': home_feats['rank'] - away_feats['rank'],
        'point_diff': home_feats['points'] - away_feats['points'],
        'home_rank_momentum': home_feats['rank_momentum'],
        'away_rank_momentum': away_feats['rank_momentum'],
        'h2h_win_rate': 0.5, # Simplified for simulation
        'h2h_game_count': 0,
        'is_home_adv': 0,
        'is_neutral': 1
    }
    
    X = pd.DataFrame([match_feats])
    probs = model.predict_proba(X)[0]
    return probs

def simulate_match(home_team, away_team, model, historical_df):
    probs = get_match_probs_fast(home_team, away_team, model, historical_df)
    outcome = np.random.choice(['home', 'draw', 'away'], p=probs)
    
    if outcome == 'home':
        return home_team
    elif outcome == 'away':
        return away_team
    else:
        return np.random.choice([home_team, away_team])

def simulate_tournament():
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    historical_df = pd.read_csv(FEATURES_TABLE)
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    
    teams = ["Morocco", "Senegal", "Ivory Coast", "Nigeria", "Egypt", "Algeria", "Cameroon", "Tunisia"]
    
    n_simulations = 100
    win_counts = {team: 0 for team in teams}
    
    print(f"Simulating tournament {n_simulations} times...")
    
    for _ in range(n_simulations):
        # Quarter-finals
        qf1 = simulate_match(teams[0], teams[7], model, historical_df)
        qf2 = simulate_match(teams[1], teams[6], model, historical_df)
        qf3 = simulate_match(teams[2], teams[5], model, historical_df)
        qf4 = simulate_match(teams[3], teams[4], model, historical_df)
        
        # Semi-finals
        sf1 = simulate_match(qf1, qf4, model, historical_df)
        sf2 = simulate_match(qf2, qf3, model, historical_df)
        
        # Final
        winner = simulate_match(sf1, sf2, model, historical_df)
        win_counts[winner] += 1
        
    print("\nTournament Win Probabilities (based on 100 simulations):")
    for team, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{team}: {count/n_simulations:.2%}")

if __name__ == "__main__":
    simulate_tournament()

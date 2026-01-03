import pandas as pd
import numpy as np
import pickle
import os
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
            'weighted_form': team_latest.iloc[0]['home_weighted_form'],
            'goal_diff_form': team_latest.iloc[0]['home_goal_diff_form'],
            'rank_momentum': team_latest.iloc[0]['home_rank_momentum'],
            'elo': team_latest.iloc[0]['home_elo'],
            'travel_dist': team_latest.iloc[0]['home_travel_dist'],
            'squad_value': team_latest.iloc[0]['home_squad_value'],
            'squad_quality': team_latest.iloc[0]['home_squad_quality'],
            'log_value': team_latest.iloc[0]['log_home_value']
        }
    else:
        return {
            'rank': team_latest.iloc[0]['away_rank'],
            'points': team_latest.iloc[0]['away_points'],
            'form': team_latest.iloc[0]['away_form'],
            'weighted_form': team_latest.iloc[0]['away_weighted_form'],
            'goal_diff_form': team_latest.iloc[0]['away_goal_diff_form'],
            'rank_momentum': team_latest.iloc[0]['away_rank_momentum'],
            'elo': team_latest.iloc[0]['away_elo'],
            'travel_dist': team_latest.iloc[0]['away_travel_dist'],
            'squad_value': team_latest.iloc[0]['away_squad_value'],
            'squad_quality': team_latest.iloc[0]['away_squad_quality'],
            'log_value': team_latest.iloc[0]['log_away_value']
        }

def get_match_probs_fast(home_team, away_team, model, team_features):
    home_feats = team_features.get(home_team)
    away_feats = team_features.get(away_team)
    
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
        'home_weighted_form': home_feats['weighted_form'],
        'away_weighted_form': away_feats['weighted_form'],
        'home_goal_diff_form': home_feats['goal_diff_form'],
        'away_goal_diff_form': away_feats['goal_diff_form'],
        'rank_diff': home_feats['rank'] - away_feats['rank'],
        'point_diff': home_feats['points'] - away_feats['points'],
        'home_rank_momentum': home_feats['rank_momentum'],
        'away_rank_momentum': away_feats['rank_momentum'],
        'h2h_win_rate': 0.5, # Simplified for simulation
        'h2h_game_count': 0,
        'is_home_adv': 0,
        'is_neutral': 1,
        'tournament_weight': 8, # AFCON weight
        'home_elo': home_feats['elo'],
        'away_elo': away_feats['elo'],
        'elo_diff': home_feats['elo'] - away_feats['elo'],
        'home_travel_dist': home_feats['travel_dist'],
        'away_travel_dist': away_feats['travel_dist'],
        'home_squad_value': home_feats['squad_value'],
        'away_squad_value': away_feats['squad_value'],
        'home_squad_quality': home_feats['squad_quality'],
        'away_squad_quality': away_feats['squad_quality'],
        'log_home_value': home_feats['log_value'],
        'log_away_value': away_feats['log_value'],
        'value_diff': home_feats['squad_value'] - away_feats['squad_value'],
        'value_ratio': home_feats['squad_value'] / (away_feats['squad_value'] + 1e-6),
        'quality_diff': home_feats['squad_quality'] - away_feats['squad_quality']
    }
    
    X = pd.DataFrame([match_feats])
    probs = model.predict_proba(X)[0]
    return probs

def simulate_match(home_team, away_team, model, team_features):
    probs = get_match_probs_fast(home_team, away_team, model, team_features)
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
    
    # Round of 16 Matchups
    r16_matches = [
        ("Senegal", "Sudan"),
        ("Mali", "Tunisia"),
        ("Morocco", "Tanzania"),
        ("South Africa", "Cameroon"),
        ("Egypt", "Benin"),
        ("Nigeria", "Mozambique"),
        ("Algeria", "DR Congo"),
        ("Burkina Faso", "Ivory Coast")
    ]
    
    all_teams = list(set([t for m in r16_matches for t in m]))
    
    # Pre-calculate latest features for all teams
    print("Pre-calculating team features...")
    team_features = {}
    for team in all_teams:
        feats = get_team_features(team, historical_df)
        if feats:
            team_features[team] = feats
    
    # Pre-calculate all possible match probabilities to speed up simulation
    print("Pre-calculating match probabilities...")
    match_probs = {}
    for t1 in all_teams:
        for t2 in all_teams:
            if t1 == t2: continue
            match_probs[(t1, t2)] = get_match_probs_fast(t1, t2, model, team_features)

    def simulate_match_fast(t1, t2):
        probs = match_probs.get((t1, t2), [0.33, 0.34, 0.33])
        outcome = np.random.choice(['home', 'draw', 'away'], p=probs)
        if outcome == 'home': return t1
        if outcome == 'away': return t2
        return np.random.choice([t1, t2])

    n_simulations = 10000
    win_counts = {team: 0 for team in all_teams}
    final_counts = {team: 0 for team in all_teams}
    sf_counts = {team: 0 for team in all_teams}
    qf_counts = {team: 0 for team in all_teams}
    
    print(f"Simulating tournament {n_simulations} times...")
    
    for _ in range(n_simulations):
        # Round of 16
        r16_winners = [simulate_match_fast(m[0], m[1]) for m in r16_matches]
        for w in r16_winners: qf_counts[w] += 1
        
        # Quarter-finals
        qf_winners = [
            simulate_match_fast(r16_winners[0], r16_winners[1]),
            simulate_match_fast(r16_winners[2], r16_winners[3]),
            simulate_match_fast(r16_winners[4], r16_winners[5]),
            simulate_match_fast(r16_winners[6], r16_winners[7])
        ]
        for w in qf_winners: sf_counts[w] += 1
        
        # Semi-finals
        sf_winners = [
            simulate_match_fast(qf_winners[0], qf_winners[1]),
            simulate_match_fast(qf_winners[2], qf_winners[3])
        ]
        for w in sf_winners: final_counts[w] += 1
        
        # Final
        winner = simulate_match_fast(sf_winners[0], sf_winners[1])
        win_counts[winner] += 1
    
    # Results
    results = pd.DataFrame({
        'Team': all_teams,
        'QF Prob': [qf_counts[t]/n_simulations for t in all_teams],
        'SF Prob': [sf_counts[t]/n_simulations for t in all_teams],
        'Final Prob': [final_counts[t]/n_simulations for t in all_teams],
        'Winner Prob': [win_counts[t]/n_simulations for t in all_teams]
    }).sort_values('Winner Prob', ascending=False)
    
    print("\nTournament Simulation Results:")
    print(results.to_string(index=False))
    
    # Save results for dashboard
    os.makedirs("data/processed", exist_ok=True)
    results.to_csv("data/processed/simulation_results.csv", index=False)
    print(f"\nResults saved to data/processed/simulation_results.csv")
    
    return results

if __name__ == "__main__":
    simulate_tournament()

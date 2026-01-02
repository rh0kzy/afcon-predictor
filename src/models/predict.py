import pandas as pd
import pickle
from src.config import XGB_MODEL_PATH

def predict_match(home_team, away_team, features_dict):
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # features_dict should contain the necessary features for the prediction
    X = pd.DataFrame([features_dict])
    
    probs = model.predict_proba(X)[0]
    
    print(f"Prediction for {home_team} vs {away_team}:")
    print(f"Home Win: {probs[0]:.2%}")
    print(f"Draw: {probs[1]:.2%}")
    print(f"Away Win: {probs[2]:.2%}")
    
    return probs

if __name__ == "__main__":
    # Example usage
    example_features = {
        'home_rank': 30, 'away_rank': 50, 
        'home_points': 1500, 'away_points': 1400,
        'home_form': 2.0, 'away_form': 1.5,
        'rank_diff': -20, 'point_diff': 100
    }
    predict_match("Nigeria", "Ghana", example_features)

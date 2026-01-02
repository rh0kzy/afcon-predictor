import pandas as pd
import pickle
from src.config import XGB_MODEL_PATH, FEATURES_TABLE, EXTERNAL_DATA_DIR
from src.features.form_features import calculate_form
from src.features.h2h_features import calculate_h2h
from src.features.fifa_features import calculate_fifa_features
from src.features.context_features import calculate_context_features

def predict_afcon_2025():
    # Load model
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load fixtures
    fixtures = pd.read_csv(EXTERNAL_DATA_DIR / "afcon_2025_fixtures.csv")
    fixtures['date'] = pd.to_datetime(fixtures['date'])
    
    # Load historical data to calculate features
    historical_df = pd.read_csv(FEATURES_TABLE)
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    
    # Combine fixtures with historical data to calculate rolling features
    # We only need the most recent matches for each team
    combined_df = pd.concat([historical_df, fixtures], ignore_index=True).sort_values('date')
    
    # Re-calculate features (this is a bit redundant but ensures consistency)
    # In a real app, you'd just take the last known state for each team
    combined_df = calculate_form(combined_df)
    combined_df = calculate_h2h(combined_df)
    combined_df = calculate_fifa_features(combined_df)
    combined_df = calculate_context_features(combined_df)
    
    # Filter back to just the fixtures
    prediction_df = combined_df[combined_df['home_score'].isna()].copy()
    
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight'
    ]
    
    # Ensure all features exist and fill NaNs
    for col in features:
        if col not in prediction_df.columns:
            prediction_df[col] = 0.0
        else:
            # If there are duplicate columns, take the first one
            if isinstance(prediction_df[col], pd.DataFrame):
                prediction_df[col] = prediction_df[col].iloc[:, 0]
            prediction_df[col] = prediction_df[col].fillna(0.0)
    
    # Select only the first occurrence of each feature to avoid duplicates
    X = prediction_df.loc[:, ~prediction_df.columns.duplicated()][features]
    probs = model.predict_proba(X)
    
    results = prediction_df[['date', 'home_team', 'away_team']].copy()
    results['Home Win Prob'] = probs[:, 0]
    results['Draw Prob'] = probs[:, 1]
    results['Away Win Prob'] = probs[:, 2]
    
    print("\nAFCON 2025 Predictions:")
    print(results[['home_team', 'away_team', 'Home Win Prob', 'Draw Prob', 'Away Win Prob']])
    
    results.to_csv(EXTERNAL_DATA_DIR / "afcon_2025_predictions.csv", index=False)
    print(f"\nPredictions saved to {EXTERNAL_DATA_DIR / 'afcon_2025_predictions.csv'}")

if __name__ == "__main__":
    predict_afcon_2025()

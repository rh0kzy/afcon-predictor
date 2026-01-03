import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES_TABLE, BASELINE_MODEL_PATH, SCALER_PATH

def train_baseline():
    df = pd.read_csv(FEATURES_TABLE)
    
    # Define target: 0 for Home Win, 1 for Draw, 2 for Away Win
    df['target'] = 1 # Draw
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0 # Home Win
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2 # Away Win
    
    # Select features
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_weighted_form', 'away_weighted_form',
        'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight',
        'home_elo', 'away_elo', 'elo_diff', 'home_travel_dist', 'away_travel_dist'
    ]
    
    # Drop rows with NaN in features
    df = df.dropna(subset=features)
    
    # Time-based split (train < 2024, test >= 2024)
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year < 2024]
    test_df = df[df['date'].dt.year >= 2024]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Training Baseline (Logistic Regression) on {len(X_train)} matches, testing on {len(X_test)} matches.")
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(
        C=0.01,
        solver='saga',
        multi_class='multinomial', 
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    with open(BASELINE_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Baseline model trained and saved to {BASELINE_MODEL_PATH}")
    return model, X_test, y_test

if __name__ == "__main__":
    train_baseline()

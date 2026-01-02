import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from src.config import FEATURES_TABLE, XGB_MODEL_PATH

def train_model():
    df = pd.read_csv(FEATURES_TABLE)
    
    # Define target: 0 for Home Win, 1 for Draw, 2 for Away Win
    df['target'] = 1 # Draw
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0 # Home Win
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2 # Away Win
    
    # Select features
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight'
    ]
    
    # Drop rows with NaN in features (e.g., early matches with no FIFA rank)
    df = df.dropna(subset=features)
    
    # Time-based split (train < 2024, test >= 2024)
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year < 2024]
    test_df = df[df['date'].dt.year >= 2024]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches.")
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.01, 
        max_depth=3, 
        subsample=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    # Save model
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {XGB_MODEL_PATH}")
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()

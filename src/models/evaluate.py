import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from src.config import XGB_MODEL_PATH, FEATURES_TABLE

def evaluate_model():
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(FEATURES_TABLE)
    df['target'] = 1
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2
    
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral'
    ]
    
    df = df.dropna(subset=features)
    
    X = df[features]
    y = df['target']
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    print(f"\nLog Loss: {log_loss(y, y_prob):.4f}")
    
if __name__ == "__main__":
    evaluate_model()

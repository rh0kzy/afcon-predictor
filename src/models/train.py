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
    features = ['home_rank', 'away_rank', 'home_points', 'away_points', 
                'home_form', 'away_form', 'rank_diff', 'point_diff']
    
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {XGB_MODEL_PATH}")
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()

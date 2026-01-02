import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from src.config import XGB_MODEL_PATH, FEATURES_TABLE

def evaluate_model():
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(FEATURES_TABLE)
    # (Simplified: just evaluating on the whole set for now, usually you'd use a test set)
    df['target'] = 1
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2
    
    features = ['home_rank', 'away_rank', 'home_points', 'away_points', 
                'home_form', 'away_form', 'rank_diff', 'point_diff']
    
    X = df[features]
    y = df['target']
    
    y_pred = model.predict(X)
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
if __name__ == "__main__":
    evaluate_model()

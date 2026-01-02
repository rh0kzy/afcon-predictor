import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, log_loss, accuracy_score
from src.config import XGB_MODEL_PATH, BASELINE_MODEL_PATH, FEATURES_TABLE, SCALER_PATH

def evaluate_model(model_path, model_name, is_baseline=False):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(FEATURES_TABLE)
    df['target'] = 1
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2
    
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight'
    ]
    
    df = df.dropna(subset=features)
    
    # Time-based split (test >= 2024)
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'].dt.year >= 2024]
    
    X = test_df[features]
    y = test_df['target']
    
    if is_baseline:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    return accuracy_score(y, y_pred), log_loss(y, y_prob)

def compare_models():
    results = {}
    
    if XGB_MODEL_PATH.exists():
        acc, loss = evaluate_model(XGB_MODEL_PATH, "XGBoost")
        results['XGBoost'] = {'Accuracy': acc, 'Log Loss': loss}
        
    if BASELINE_MODEL_PATH.exists():
        acc, loss = evaluate_model(BASELINE_MODEL_PATH, "Logistic Regression (Baseline)", is_baseline=True)
        results['Baseline'] = {'Accuracy': acc, 'Log Loss': loss}
    
    if len(results) > 1:
        print("\n--- Model Comparison ---")
        comparison_df = pd.DataFrame(results).T
        print(comparison_df)

if __name__ == "__main__":
    compare_models()

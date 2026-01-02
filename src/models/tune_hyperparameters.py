import pandas as pd
import pickle
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES_TABLE, XGB_MODEL_PATH, BASELINE_MODEL_PATH, SCALER_PATH

def tune_hyperparameters():
    df = pd.read_csv(FEATURES_TABLE)
    
    # Define target
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
    
    # Time-based split (train < 2024)
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'].dt.year < 2024]
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("--- Tuning XGBoost ---")
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    
    print(f"Best XGBoost Params: {xgb_grid.best_params_}")
    print(f"Best XGBoost Score: {xgb_grid.best_score_:.4f}")
    
    # Save best XGBoost model
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_grid.best_estimator_, f)
        
    print("--- Tuning Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga']
    }
    
    lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    
    print(f"Best Logistic Regression Params: {lr_grid.best_params_}")
    print(f"Best Logistic Regression Score: {lr_grid.best_score_:.4f}")
    
    # Save best Logistic Regression model and scaler
    with open(BASELINE_MODEL_PATH, 'wb') as f:
        pickle.dump(lr_grid.best_estimator_, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    return xgb_grid.best_params_, lr_grid.best_params_

if __name__ == "__main__":
    tune_hyperparameters()

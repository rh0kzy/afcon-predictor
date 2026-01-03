import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from src.config import XGB_MODEL_PATH, FEATURES_TABLE, FIGURES_DIR

def explain_model():
    # Load model
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(FEATURES_TABLE)
    
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_weighted_form', 'away_weighted_form',
        'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight',
        'home_elo', 'away_elo', 'elo_diff', 'home_travel_dist', 'away_travel_dist'
    ]
    
    df = df.dropna(subset=features)
    X = df[features]
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    
    # Save plot
    plot_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(plot_path)
    print(f"SHAP summary plot saved to {plot_path}")
    
    # For multi-class, shap_values is a list of arrays. 
    # Let's save summary plots for each class if needed, 
    # but the default summary_plot handles multi-class by showing feature importance across all classes.

if __name__ == "__main__":
    explain_model()

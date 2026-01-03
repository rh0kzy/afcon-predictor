import pandas as pd
import pickle
import numpy as np
from src.config import XGB_MODEL_PATH, FEATURES_TABLE

def backtest_strategy(threshold=0.6, bet_size=10):
    """
    Simulates a betting strategy based on model probabilities.
    Since we don't have real odds, we'll simulate 'fair' odds with a bookmaker margin.
    """
    with open(XGB_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(FEATURES_TABLE)
    df['target'] = 1
    df.loc[df['home_score'] > df['away_score'], 'target'] = 0
    df.loc[df['home_score'] < df['away_score'], 'target'] = 2
    
    features = [
        'home_rank', 'away_rank', 'home_points', 'away_points', 
        'home_form', 'away_form', 'home_weighted_form', 'away_weighted_form',
        'home_goal_diff_form', 'away_goal_diff_form',
        'rank_diff', 'point_diff', 'home_rank_momentum', 'away_rank_momentum',
        'h2h_win_rate', 'h2h_game_count', 'is_home_adv', 'is_neutral', 'tournament_weight',
        'home_elo', 'away_elo', 'elo_diff', 'home_travel_dist', 'away_travel_dist'
    ]
    
    df = df.dropna(subset=features)
    
    # Test on 2024-2025 data
    df['date'] = pd.to_datetime(df['date'])
    test_df = df[df['date'].dt.year >= 2024].copy()
    
    X = test_df[features]
    y = test_df['target']
    
    y_prob = model.predict_proba(X)
    
    # Simulate odds: Fair odds = 1/prob. Bookie odds = Fair odds * 0.95 (5% margin)
    # In a real scenario, we would use actual historical odds.
    # Here we'll just see if the model's 'high confidence' bets are accurate.
    
    test_df['max_prob'] = np.max(y_prob, axis=1)
    test_df['pred_outcome'] = np.argmax(y_prob, axis=1)
    
    # Filter for bets above threshold
    bets = test_df[test_df['max_prob'] >= threshold].copy()
    
    if len(bets) == 0:
        print(f"No bets found with probability >= {threshold}")
        return
    
    bets['is_correct'] = (bets['pred_outcome'] == bets['target'])
    
    # Calculate profit
    # If correct, we win: bet_size * (1/max_prob * 0.95 - 1)
    # If wrong, we lose: bet_size
    
    bets['profit'] = np.where(
        bets['is_correct'],
        bet_size * ( (1 / bets['max_prob']) * 0.95 - 1 ),
        -bet_size
    )
    
    total_profit = bets['profit'].sum()
    roi = (total_profit / (len(bets) * bet_size)) * 100
    
    print(f"--- Backtesting Results (Threshold: {threshold}) ---")
    print(f"Total Bets: {len(bets)}")
    print(f"Accuracy on Bets: {bets['is_correct'].mean():.2%}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")
    
    return total_profit

if __name__ == "__main__":
    backtest_strategy(threshold=0.5)
    backtest_strategy(threshold=0.6)
    backtest_strategy(threshold=0.7)

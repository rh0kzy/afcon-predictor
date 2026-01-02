import pandas as pd

def calculate_h2h(df):
    # Sort by date to ensure we only use past games
    df = df.sort_values('date')
    
    def get_h2h_stats(row, data):
        # Filter for past games between these two teams
        past_games = data[(data['date'] < row['date']) & 
                          (((data['home_team'] == row['home_team']) & (data['away_team'] == row['away_team'])) |
                           ((data['home_team'] == row['away_team']) & (data['away_team'] == row['home_team'])))]
        
        if len(past_games) == 0:
            return 0.5, 0 # Default to 0.5 win rate if no history
        
        # Calculate home team's win rate against this specific opponent
        home_wins = len(past_games[((past_games['home_team'] == row['home_team']) & (past_games['home_score'] > past_games['away_score'])) |
                                   ((past_games['away_team'] == row['home_team']) & (past_games['away_score'] > past_games['home_score']))])
        
        draws = len(past_games[past_games['home_score'] == past_games['away_score']])
        
        # Win rate: (wins + 0.5 * draws) / total
        win_rate = (home_wins + 0.5 * draws) / len(past_games)
        
        return win_rate, len(past_games)

    # Apply the function
    h2h_results = df.apply(lambda row: get_h2h_stats(row, df), axis=1, result_type='expand')
    df['h2h_win_rate'] = h2h_results[0]
    df['h2h_game_count'] = h2h_results[1]
    
    return df

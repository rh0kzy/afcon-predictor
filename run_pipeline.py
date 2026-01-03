import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.data.clean_matches import clean_matches
from src.data.clean_fifa import clean_fifa_rankings
from src.data.clean_goals import clean_goals
from src.data.merge_fifa import merge_fifa_rankings
from src.features.form_features import calculate_form
from src.features.h2h_features import calculate_h2h
from src.features.fifa_features import calculate_fifa_features
from src.features.context_features import calculate_context_features
from src.features.elo_features import calculate_elo
from src.features.travel_features import calculate_travel_distance
from src.features.squad_features import calculate_squad_features
from src.models.train import train_model
from src.models.train_baseline import train_baseline
from src.models.evaluate import compare_models
from src.config import FEATURES_TABLE

def run_pipeline():
    print("--- Starting Pipeline ---")
    
    print("1. Cleaning data...")
    clean_matches()
    clean_fifa_rankings()
    clean_goals()
    
    print("2. Merging datasets...")
    df = merge_fifa_rankings()
    
    print("3. Engineering features...")
    df = calculate_form(df)
    df = calculate_h2h(df)
    df = calculate_fifa_features(df)
    df = calculate_context_features(df)
    df = calculate_elo(df)
    df = calculate_travel_distance(df)
    df = calculate_squad_features(df)
    
    # Save features
    df.to_csv(FEATURES_TABLE, index=False)
    print(f"Features saved to {FEATURES_TABLE}")
    
    print("4. Training models...")
    train_baseline()
    train_model()
    
    print("5. Evaluating and comparing models...")
    compare_models()
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()

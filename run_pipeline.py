import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.data.clean_matches import clean_matches
from src.data.merge_fifa import merge_fifa_rankings
from src.features.form_features import calculate_form
from src.features.fifa_features import calculate_fifa_features
from src.models.train import train_model
from src.config import FIFA_CLEANED, FEATURES_TABLE

def run_pipeline():
    print("--- Starting Pipeline ---")
    
    print("1. Cleaning matches...")
    clean_matches()
    
    print("2. Merging FIFA rankings...")
    df = merge_fifa_rankings()
    
    print("3. Engineering features...")
    df = calculate_form(df)
    df = calculate_fifa_features(df)
    
    # Save features
    df.to_csv(FEATURES_TABLE, index=False)
    print(f"Features saved to {FEATURES_TABLE}")
    
    print("4. Training model...")
    train_model()
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()

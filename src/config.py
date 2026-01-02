import os
from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# File Paths
MATCHES_RAW = RAW_DATA_DIR / "matches.csv"
FIFA_RANKING_RAW = RAW_DATA_DIR / "fifa_ranking.csv"
GOALS_RAW = RAW_DATA_DIR / "goals.csv"

MATCHES_CLEANED = PROCESSED_DATA_DIR / "matches_cleaned.csv"
FIFA_CLEANED = PROCESSED_DATA_DIR / "fifa_cleaned.csv"
FEATURES_TABLE = PROCESSED_DATA_DIR / "features.csv"

# Model Paths
MODEL_DIR = ROOT_DIR / "models"
XGB_MODEL_PATH = MODEL_DIR / "xgb_v1.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Output Paths
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure directories exist
for path in [PROCESSED_DATA_DIR, MODEL_DIR, FIGURES_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# AFCON 2025 Predictor 🏆

A professional-grade sports modeling system for predicting the Africa Cup of Nations 2025.

## Features
- **Advanced Feature Engineering**: Elo ratings, weighted form (time decay), travel fatigue factors, and FIFA rankings.
- **Machine Learning**: Optimized XGBoost and Logistic Regression models.
- **Explainability**: SHAP values to understand model decisions.
- **Evaluation**: Ranked Probability Score (RPS) and betting backtest (ROI analysis).
- **Simulation**: 10,000-run Monte Carlo simulation for tournament outcomes.
- **Interactive Dashboard**: Streamlit app for real-time predictions and visualization.

## Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Install dependencies: `pip install -e .`

## Usage
### Run the full pipeline:
```bash
./run_pipeline.bat
```

### Launch the dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

## Project Structure
- `src/data`: Data cleaning and ingestion.
- `src/features`: Feature engineering (Elo, Travel, Form).
- `src/models`: Training, evaluation, and simulation.
- `src/visualization`: Streamlit dashboard.
- `data/`: Raw and processed data.
- `models/`: Saved model artifacts.
- `outputs/`: Figures and evaluation reports.

## Model Performance
The current XGBoost model achieves high accuracy on high-confidence predictions (~77% for prob > 0.7).
- **Primary Metric**: Ranked Probability Score (RPS).
- **Key Features**: Elo Difference, FIFA Rank Difference, Weighted Form.


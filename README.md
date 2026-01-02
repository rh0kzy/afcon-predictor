# AFCON Predictor

A machine learning project to predict the outcomes of African Cup of Nations (AFCON) matches.

## Project Structure

- `data/`: Raw and processed datasets.
- `notebooks/`: Jupyter notebooks for exploration and experimentation.
- `src/`: Source code for data processing, feature engineering, and modeling.
- `models/`: Saved model files.
- `outputs/`: Generated figures and reports.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python run_pipeline.py
   ```

## Features

- **FIFA Rankings integration**: Uses historical FIFA rankings to gauge team strength.
- **Rolling form features**: Calculates points and goal differences over the last 5 matches.
- **Head-to-head statistics**: Incorporates historical performance between specific team pairs.
- **Contextual features**: Includes home advantage and neutral venue flags.
- **XGBoost classification model**: Predicts Home Win, Draw, or Away Win probabilities.

## Usage

### 1. Data Pipeline
Run the full pipeline to clean data, engineer features, and train the model:
```bash
python run_pipeline.py
```

### 2. Predictions
Predict outcomes for AFCON 2025 fixtures:
```bash
$env:PYTHONPATH = "."; python src/models/predict_afcon.py
```

### 3. Tournament Simulation
Simulate the knockout stage to estimate title probabilities:
```bash
$env:PYTHONPATH = "."; python src/models/simulate_tournament.py
```

### 4. API
Start the FastAPI server for real-time predictions:
```bash
$env:PYTHONPATH = "."; python api/main.py
```

## Model Performance
The current XGBoost model achieves approximately **67% accuracy** on the test set (matches from 2024 onwards).
- **Log Loss**: ~0.78
- **Key Features**: FIFA Rank Difference, Team Form, Home Advantage.

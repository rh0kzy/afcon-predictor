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

- FIFA Rankings integration.
- Rolling form features.
- Head-to-head statistics.
- XGBoost classification model.

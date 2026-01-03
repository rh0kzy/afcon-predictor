@echo off
echo Starting AFCON 2025 Predictor Pipeline...

echo [1/5] Cleaning and Preprocessing Data...
.venv\Scripts\python.exe src/data/clean_data.py

echo [2/5] Engineering Features (Elo, Travel, Form)...
.venv\Scripts\python.exe src/features/build_features.py

echo [3/5] Training Model and Evaluating...
.venv\Scripts\python.exe src/models/train_model.py

echo [4/5] Generating Explainability and Backtest...
.venv\Scripts\python.exe src/models/explain_model.py
.venv\Scripts\python.exe src/models/backtest.py

echo [5/5] Running 10,000 Tournament Simulations...
.venv\Scripts\python.exe src/models/simulate_tournament.py

echo Pipeline Complete! Run 'streamlit run src/visualization/dashboard.py' to view results.

# AFCON 2025 Match Predictor — Task List

## PHASE 0 — Project Setup
**Goal:** Create a clean and reproducible ML project structure.
- [x] Create project root folder: `afcon-predictor`
- [x] Create folders: `data/raw`, `data/processed`, `src`, `models`, `notebooks`
- [x] Create `requirements.txt`
- [x] Create `.gitignore`
- [x] Create `README.md`
- [ ] (Optional) Initialize a Git repository

## PHASE 1 — Raw Data Ingestion
**Goal:** Store original datasets safely (never modify them).
- [x] Copy `matches.csv` into `data/raw`
- [x] Copy `fifa_ranking.csv` into `data/raw`
- [x] Copy `goals.csv` into `data/raw`
- [ ] Verify CSV encoding and delimiter
- [ ] Verify column names and date formats

## PHASE 2 — Data Cleaning (No Features Yet)
**Goal:** Produce clean, normalized datasets using Python scripts only.
### Matches Data
- [x] Parse date column to datetime
- [x] Drop rows with missing scores
- [x] Filter matches (recommended: year ≥ 1992)
- [x] Sort matches chronologically
- [x] Normalize team names
- [x] Save output as `data/processed/matches_cleaned.csv`
### FIFA Ranking Data
- [x] Parse `rank_date` to datetime
- [x] Normalize country names
- [ ] Remove duplicates
- [x] Sort by `rank_date`
- [x] Keep relevant columns only
- [x] Save output as `data/processed/fifa_cleaned.csv`
### Goals Data
- [ ] Verify date alignment
- [ ] Aggregate goals per team per match if needed
- [ ] Remove unnecessary historical noise
- [ ] Save output as `data/processed/goals_cleaned.csv`

## PHASE 3 — Data Validation
**Goal:** Ensure data consistency before feature engineering.
- [ ] Verify team names match across all datasets
- [ ] Verify date ranges overlap correctly
- [ ] Ensure no future data leakage
- [ ] Check number of matches per year
- [ ] Check number of teams

## PHASE 4 — Feature Engineering
**Goal:** Build predictive features.
### Target Variable
- [x] Create match result label: Home win, Draw, Away win
### Team Form Features
- [x] Points last 5 matches
- [x] Goals scored last 5 matches
- [x] Goals conceded last 5 matches
- [ ] Goal difference last 5 matches
### Head-to-Head Features
- [ ] Historical win rate (home vs away)
- [ ] Safe handling when no history exists
### FIFA Features
- [x] Home FIFA rank (latest before match)
- [x] Away FIFA rank
- [x] FIFA rank difference
- [x] Home FIFA points
- [x] Away FIFA points
- [x] FIFA points difference
- [ ] Rank change (momentum)
### Context Features
- [ ] Home advantage flag
- [ ] Neutral venue flag
- [ ] Tournament importance weighting (optional)
- [x] Save final feature table as `data/processed/features.csv`

## PHASE 5 — Train / Test Split
**Goal:** Prepare data for modeling safely.
- [ ] Perform time-based split (train < 2024, test ≥ 2024)
- [x] Separate features (X) and target (y)
- [x] Verify no data leakage

## PHASE 6 — Modeling
**Goal:** Train predictive models.
- [ ] Train Logistic Regression (baseline)
- [x] Train XGBoost or LightGBM (main model)
- [ ] Tune basic hyperparameters
- [ ] Select best model
- [x] Save trained model to `models/` directory

## PHASE 7 — Evaluation
**Goal:** Measure realistic performance.
- [x] Compute accuracy
- [ ] Compute log loss
- [x] Generate confusion matrix
- [x] Evaluate Win / Draw / Loss performance
- [ ] Compare against baseline

## PHASE 8 — AFCON 2025 Prediction
**Goal:** Predict real tournament matches.
- [ ] Load official AFCON 2025 fixtures
- [ ] Generate features using latest available data
- [ ] Predict probabilities for each match
- [ ] Export predictions to CSV

## PHASE 9 — Tournament Simulation (Optional)
**Goal:** Simulate AFCON 2025 outcomes.
- [ ] Simulate group stage using Monte Carlo
- [ ] Estimate qualification probabilities
- [ ] Simulate knockout rounds
- [ ] Estimate title probabilities

## PHASE 10 — Deployment (Optional)
**Goal:** Make predictions usable.
- [x] Create single-match prediction script
- [ ] Build FastAPI endpoint
- [ ] Define input/output schema
- [ ] Test locally

## PHASE 11 — Documentation
**Goal:** Make the project understandable and reusable.
- [x] Update README with pipeline explanation
- [ ] Document features used
- [ ] Document model assumptions and limitations

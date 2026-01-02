from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from src.config import XGB_MODEL_PATH

app = FastAPI(title="AFCON 2025 Predictor API")

# Load model at startup
with open(XGB_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

class MatchInput(BaseModel):
    home_rank: int
    away_rank: int
    home_points: float
    away_points: float
    home_form: float
    away_form: float
    home_goal_diff_form: float
    away_goal_diff_form: float
    rank_diff: int
    point_diff: float
    home_rank_momentum: int
    away_rank_momentum: int
    h2h_win_rate: float
    h2h_game_count: int
    is_home_adv: int
    is_neutral: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the AFCON 2025 Predictor API"}

@app.post("/predict")
def predict(match: MatchInput):
    df = pd.DataFrame([match.dict()])
    probs = model.predict_proba(df)[0]
    return {
        "home_win_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "away_win_prob": float(probs[2])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

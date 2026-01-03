import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from PIL import Image

# Paths
MODEL_PATH = "models/xgboost_model.pkl"
FEATURES_TABLE = "data/processed/match_features.csv"
SHAP_PLOT_PATH = "outputs/figures/shap_summary.png"

st.set_page_config(page_title="AFCON 2025 Predictor", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    if os.path.exists(FEATURES_TABLE):
        df = pd.read_csv(FEATURES_TABLE)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return None

def get_team_features(team_name, df):
    team_df = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].sort_values('date', ascending=False)
    if team_df.empty:
        return None
    
    latest = team_df.iloc[0]
    if latest['home_team'] == team_name:
        return {
            'rank': latest['home_rank'],
            'points': latest['home_points'],
            'form': latest['home_form'],
            'weighted_form': latest['home_weighted_form'],
            'goal_diff_form': latest['home_goal_diff_form'],
            'rank_momentum': latest['home_rank_momentum'],
            'elo': latest['home_elo'],
            'travel_dist': latest['home_travel_dist']
        }
    else:
        return {
            'rank': latest['away_rank'],
            'points': latest['away_points'],
            'form': latest['away_form'],
            'weighted_form': latest['away_weighted_form'],
            'goal_diff_form': latest['away_goal_diff_form'],
            'rank_momentum': latest['away_rank_momentum'],
            'elo': latest['away_elo'],
            'travel_dist': latest['away_travel_dist']
        }

def main():
    st.title("🏆 AFCON 2025 Predictor - Pro Dashboard")
    st.markdown("---")

    model = load_model()
    df = load_data()

    if model is None or df is None:
        st.error("Model or data not found. Please run the training pipeline first.")
        return

    # Sidebar
    st.sidebar.header("Match Predictor")
    all_teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))
    
    home_team = st.sidebar.selectbox("Home Team", all_teams, index=all_teams.index("Morocco") if "Morocco" in all_teams else 0)
    away_team = st.sidebar.selectbox("Away Team", all_teams, index=all_teams.index("Senegal") if "Senegal" in all_teams else 1)

    if st.sidebar.button("Predict Match"):
        h_feats = get_team_features(home_team, df)
        a_feats = get_team_features(away_team, df)
        
        if h_feats and a_feats:
            match_feats = {
                'home_rank': h_feats['rank'],
                'away_rank': a_feats['rank'],
                'home_points': h_feats['points'],
                'away_points': a_feats['points'],
                'home_form': h_feats['form'],
                'away_form': a_feats['form'],
                'home_weighted_form': h_feats['weighted_form'],
                'away_weighted_form': a_feats['weighted_form'],
                'home_goal_diff_form': h_feats['goal_diff_form'],
                'away_goal_diff_form': a_feats['goal_diff_form'],
                'rank_diff': h_feats['rank'] - a_feats['rank'],
                'point_diff': h_feats['points'] - a_feats['points'],
                'home_rank_momentum': h_feats['rank_momentum'],
                'away_rank_momentum': a_feats['rank_momentum'],
                'h2h_win_rate': 0.5,
                'h2h_game_count': 0,
                'is_home_adv': 0,
                'is_neutral': 1,
                'tournament_weight': 8,
                'home_elo': h_feats['elo'],
                'away_elo': a_feats['elo'],
                'elo_diff': h_feats['elo'] - a_feats['elo'],
                'home_travel_dist': h_feats['travel_dist'],
                'away_travel_dist': a_feats['travel_dist']
            }
            
            X = pd.DataFrame([match_feats])
            probs = model.predict_proba(X)[0]
            
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{home_team} Win", f"{probs[0]*100:.1f}%")
            col2.metric("Draw", f"{probs[1]*100:.1f}%")
            col3.metric(f"{away_team} Win", f"{probs[2]*100:.1f}%")
            
            # Probability Bar Chart
            prob_df = pd.DataFrame({
                'Outcome': [home_team, 'Draw', away_team],
                'Probability': probs
            })
            fig = px.bar(prob_df, x='Outcome', y='Probability', color='Outcome', 
                         color_discrete_map={home_team: 'green', 'Draw': 'gray', away_team: 'blue'})
            st.plotly_chart(fig)

    # Main Content
    tab1, tab2, tab3 = st.tabs(["Tournament Simulation", "Model Explainability", "Team Stats"])

    with tab1:
        st.header("Monte Carlo Simulation (10,000 Runs)")
        SIM_RESULTS_PATH = "data/processed/simulation_results.csv"
        
        if os.path.exists(SIM_RESULTS_PATH):
            sim_df = pd.read_csv(SIM_RESULTS_PATH)
            
            fig_sim = px.bar(sim_df, x='Winner Prob', y='Team', orientation='h', 
                             title="Probability of Winning AFCON 2025",
                             color='Winner Prob', color_continuous_scale='Viridis')
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.subheader("Detailed Probabilities")
            st.dataframe(sim_df.style.format({
                'QF Prob': '{:.2%}',
                'SF Prob': '{:.2%}',
                'Final Prob': '{:.2%}',
                'Winner Prob': '{:.2%}'
            }))
        else:
            st.warning("Simulation results not found. Run src/models/simulate_tournament.py first.")

    with tab2:
        st.header("SHAP Feature Importance")
        if os.path.exists(SHAP_PLOT_PATH):
            image = Image.open(SHAP_PLOT_PATH)
            st.image(image, caption="SHAP Summary Plot - What drives the model?")
        else:
            st.warning("SHAP plot not found. Run src/models/explain_model.py to generate it.")

    with tab3:
        st.header("Team Comparison")
        t1 = st.selectbox("Select Team 1", all_teams, index=0)
        t2 = st.selectbox("Select Team 2", all_teams, index=1)
        
        f1 = get_team_features(t1, df)
        f2 = get_team_features(t2, df)
        
        if f1 and f2:
            comp_df = pd.DataFrame([f1, f2], index=[t1, t2]).T
            st.table(comp_df)

if __name__ == "__main__":
    main()

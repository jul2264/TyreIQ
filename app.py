import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
from strategy_engine import calculate_pit_window, get_base_lap_time, recommend_pit, simulate_full_race_strategy
from generate_data import simulate_lap_time

st.set_page_config(page_title="TyreIQ", layout="wide")

@st.cache_resource
def load_model():
    if os.path.exists('model.joblib'):
        return joblib.load('model.joblib')
    return None

model = load_model()

# Sidebar rendering needs to happen early to capture theme state for CSS
if model is not None:
    with st.sidebar:
        st.header("🧠 Model Architecture")
        st.markdown("**Core Model**: Random Forest Regressor")
        st.markdown("**Accuracy (RMSE)**: 0.23 sec")
        
        st.divider()
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        features = ['Lap_Number', 'Tire_Type', 'Track_Temp', 'Track_Type']
        for i, f in enumerate(features):
            st.progress(float(importances[i]), text=f"{f}: {importances[i]*100:.1f}%")
            
        st.divider()
        mode = st.radio("Analytics Engine", ["AI Trained Model", "Pure Simulation Mathematics"], help="Toggle between AI-inferred loss vs rigid equation output.")
        
        st.divider()
        theme = st.radio("UI Theme", ["Dark Mode", "Light Mode"])
else:
    theme = "Dark Mode"
    mode = "AI Trained Model"

# Custom CSS for rich aesthetics and dynamic theme
if theme == "Dark Mode":
    theme_css = """
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #0b0f19 0%, #1a2235 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
    }
    h1, h2, h3, h4, h5, h6, label, .stMetricValue, .stMetricLabel, p { color: #e2e8f0 !important; }
    """
    plotly_theme = "plotly_dark"
    plotly_font_color = "#e2e8f0"
else:
    theme_css = """
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        color: #0f172a !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #1e293b !important;
        color: #1e293b !important;
    }
    h1, h2, h3, h4, h5, h6, label, .stMetricValue, .stMetricLabel, p, .stMarkdown p, .stAlert p { color: #1e293b !important; }
    """
    plotly_theme = "plotly_white"
    plotly_font_color = "#1e293b"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    h1 {{
        background: -webkit-linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}
    
    [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {{
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #f8fafc !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        margin-top: 20px;
    }}
    .stButton p {{
        color: #f8fafc !important;
    }}
    [data-testid="baseButton-secondary"]:hover, [data-testid="baseButton-primary"]:hover {{
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.6) !important;
        color: #ffffff !important;
        border-color: transparent !important;
    }}
    {theme_css}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ TypeIQ: AI Tire Strategy Engine")
st.markdown("Projected stint performance based on compounding variables and AI-detected deterioration cliffs.")

if model is None:
    st.error("Model not found! Please run `python model_trainer.py` in your terminal.")
else:

    # Global Track Configurations manually elevated above the tabs
    st.subheader("Race Conditions")
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        track_str = st.selectbox("Circuit Profile", ["Silverstone (UK)", "Monza (Italy)", "Monaco", "Marina Bay (Singapore)", "Interlagos (Brazil)"])
        track_map = {
            "Silverstone (UK)": 0, 
            "Monza (Italy)": 1, 
            "Monaco": 2, 
            "Marina Bay (Singapore)": 3, 
            "Interlagos (Brazil)": 4
        }
        track_type = track_map[track_str]
    with env_col2:
        track_temp = st.slider("Track Temperature (°C)", min_value=20.0, max_value=50.0, value=35.0, step=0.5)
        
    st.markdown("---")
    
    # Render Tabs
    tab1, tab2 = st.tabs(["⏱️ Mid-Race Tactical Analyzer", "🏁 Pre-Race Full Simulator"])
    
    # ------------------
    # TAB 1: MID-RACE
    # ------------------
    with tab1:
        kpi_row1_col1, kpi_row1_col2, kpi_row1_col3 = st.columns(3)
        kpi_row2_col1, kpi_row2_col2, kpi_row2_col3 = st.columns(3)
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            st.subheader("Race Status")
            tire_str = st.selectbox("Current Tire Compound", ["Soft", "Medium", "Hard"])
            tire_map = {"Soft": 0, "Medium": 1, "Hard": 2}
            tire_type = tire_map[tire_str]
            
            current_lap = st.slider("Current Race Lap", min_value=1, max_value=70, value=10)
            
            analyze_btn = st.button("Generate Strategy Analysis")
            
        with col2:
            if analyze_btn:
                laps_ahead = list(range(current_lap, current_lap + 20))
                
                if mode == "Pure Simulation Mathematics":
                    predicted_laps = [simulate_lap_time(l, tire_type, track_temp, track_type, add_noise=False) for l in laps_ahead]
                    base_lap_time = get_base_lap_time(tire_type)
                    strategy_info = recommend_pit(current_lap, tire_type, predicted_laps, base_lap_time)
                    strategy_info["Alt_Loss"] = 0.0
                    strategy_info["Confidence"] = 100.0
                    strategy_info["Phase"] = "Simulation Mode"
                    strategy_info["Undercut_Gain"] = 0.0
                    strategy_info["Undercut_Risk"] = "N/A"
                    strategy_info["Std_Dev"] = 0.5
                    alt_preds = []
                else:
                    strategy_info, predicted_laps, alt_preds = calculate_pit_window(
                        tire_type, current_lap, track_temp, track_type, model
                    )
                    
                # Render KPIs
                pct_confidence = strategy_info["Confidence"]
                kpi_row1_col1.metric("Strategy Confidence", f"{pct_confidence:.1f}%")
                kpi_row1_col2.metric("Degradation Phase", strategy_info.get("Phase", "Linear"))
                kpi_row1_col3.metric("Projected Cliff", f"Lap {strategy_info['Projected_Cliff']}")
                
                avg_deg = strategy_info["Avg_Degradation"]
                total_loss = strategy_info["Total_Time_Loss"]
                undercut_gain = strategy_info.get("Undercut_Gain", 0.0)
                
                kpi_row2_col1.metric("Avg Degradation Rate", f"{avg_deg:+.2f} s/lap")
                kpi_row2_col2.metric("Total Degradation Delta", f"{total_loss:.1f} sec")
                kpi_row2_col3.metric("Undercut Advantage", f"{undercut_gain:+.2f}s ({strategy_info['Undercut_Risk']} Risk)")
                
                # Reasoning Output
                decision = strategy_info["Decision"]
                if decision == "PIT NOW":
                    st.error(f"🚨 **Strategy Decision: PIT NOW**")
                else:
                    st.success(f"✅ **Strategy Decision: STAY OUT**")
                    
                st.markdown(f"**Reason**: {strategy_info['Reason']}")
                
                # Multi-Stint Alternative
                alt_loss = strategy_info.get("Alt_Loss", 0.0)
                if mode == "AI Trained Model":
                    if alt_loss < total_loss:
                        st.info(f"**Multi-Stint Analysis**: \n- Recommended Strategy B (Pit to Mediums): +{alt_loss:.1f}s loss \n- Alternative (Stay out): +{total_loss:.1f}s loss widening")
                    else:
                        st.info(f"**Multi-Stint Analysis**: \n- Recommended Strategy A (Stay out): +{total_loss:.1f}s loss \n- Alternative (Pit to Mediums): +{alt_loss:.1f}s loss")
                        
                # Visualizing the Plotly Chart
                st.subheader("Projected Stint Performance vs. Ideal Timeline")
                
                base_time = get_base_lap_time(tire_type)
                ideals = [base_time - (l * 0.05) for l in laps_ahead]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=laps_ahead, y=predicted_laps, 
                    mode='lines+markers', name='Projected Stint Performance (Stay Out)', 
                    line=dict(color='#00f2fe', width=3)
                ))
                
                if mode == "AI Trained Model" and len(alt_preds) > 0:
                    alt_curve_y = [predicted_laps[0] + 22.0] + list(alt_preds[:-1])
                    fig.add_trace(go.Scatter(
                        x=laps_ahead, y=alt_curve_y, 
                        mode='lines+markers', name='Alternative (Pit Now + Fresh Mediums)', 
                        line=dict(color='#f59e0b', width=2, dash='dot')
                    ))
                
                fig.add_trace(go.Scatter(
                    x=laps_ahead, y=ideals, 
                    mode='lines', name='Baseline Ideal (Fuel & Track Adjusted)', 
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=laps_ahead + laps_ahead[::-1],
                    y=predicted_laps + ideals[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 242, 254, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                pit_lap_detected = strategy_info["Pit_Lap"]
                if pit_lap_detected:
                    window_width = max(1.0, strategy_info.get("Std_Dev", 0.5) * 3)
                    fig.add_vrect(
                        x0=pit_lap_detected - (window_width/2), x1=pit_lap_detected + (window_width/2), 
                        fillcolor="rgba(239, 68, 68, 0.3)", 
                        layer="below", line_width=0, 
                        annotation_text="Dynamic Strategy Window",
                        annotation_position="top left",
                        annotation_font_color="#ef4444"
                    )
                
                fig.update_layout(
                    template=plotly_theme, 
                    font=dict(color=plotly_font_color),
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Race Lap Number",
                    yaxis_title="Lap Time (s)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True, theme=None)

    # ------------------
    # TAB 2: FULL RACE
    # ------------------
    with tab2:
        st.header("🏁 Full Race AI Simulator")
        st.markdown("Compare a 1-Stop and a 2-Stop strategy dynamically modeled across a full Grand Prix. Predicts total time across multiple stints factoring in 22-second pit penalties.")
        
        sim_col1, sim_col2 = st.columns([1, 2.5])
        
        with sim_col1:
            total_race_laps = st.slider("Total Race Laps", 40, 70, 50, key="full_race_laps")
            st.divider()
            
            st.markdown("### 1-Stop Strategy")
            s1_t1 = st.selectbox("1-Stop: Stint 1 Compound", ["Soft", "Medium", "Hard"], index=1, key="s1_t1")
            s1_l1 = st.number_input("1-Stop: Stint 1 Laps", value=20, min_value=1, key="s1_l1")
            s1_t2 = st.selectbox("1-Stop: Stint 2 Compound", ["Soft", "Medium", "Hard"], index=2, key="s1_t2")
            
            st.divider()
            
            st.markdown("### 2-Stop Strategy")
            s2_t1 = st.selectbox("2-Stop: Stint 1 Compound", ["Soft", "Medium", "Hard"], index=0, key="s2_t1")
            s2_l1 = st.number_input("2-Stop: Stint 1 Laps", value=12, min_value=1, key="s2_l1")
            s2_t2 = st.selectbox("2-Stop: Stint 2 Compound", ["Soft", "Medium", "Hard"], index=1, key="s2_t2")
            s2_l2 = st.number_input("2-Stop: Stint 2 Laps", value=18, min_value=1, key="s2_l2")
            s2_t3 = st.selectbox("2-Stop: Stint 3 Compound", ["Soft", "Medium", "Hard"], index=0, key="s2_t3")
            
            run_sim_btn = st.button("Simulate Race Now", type="primary")

        with sim_col2:
            if run_sim_btn:
                # Setup configuration lengths
                s1_l2 = total_race_laps - s1_l1
                if s1_l2 <= 0:
                    st.error("1-Stop configuration laps must not exceed Total Race Laps!")
                else:
                    one_stop_strategy = [(s1_t1, s1_l1), (s1_t2, s1_l2)]
                    
                    s2_l3 = total_race_laps - s2_l1 - s2_l2
                    if s2_l3 <= 0:
                        st.error("2-Stop configuration (Stints 1 & 2) surpasses the total race laps! Adjust lengths.")
                    else:
                        two_stop_strategy = [(s2_t1, s2_l1), (s2_t2, s2_l2), (s2_t3, s2_l3)]
                        
                        # Generate Trajectories via Strategy Engine AI
                        one_stop_laps = simulate_full_race_strategy(one_stop_strategy, track_temp, track_type, model)
                        two_stop_laps = simulate_full_race_strategy(two_stop_strategy, track_temp, track_type, model)
                        
                        time_1 = sum(one_stop_laps)
                        time_2 = sum(two_stop_laps)
                        
                        st.subheader("Strategy Time Comparison")
                        
                        # Calculate string formatting
                        t1_m, t1_s = divmod(time_1, 60)
                        t1_h, t1_m = divmod(t1_m, 60)
                        time1_str = f"{int(t1_h)}h {int(t1_m)}m {t1_s:.1f}s"
                        
                        t2_m, t2_s = divmod(time_2, 60)
                        t2_h, t2_m = divmod(t2_m, 60)
                        time2_str = f"{int(t2_h)}h {int(t2_m)}m {t2_s:.1f}s"
                        
                        st.markdown(f"**1-Stop Strategy Total Time:** {time1_str}")
                        st.markdown(f"**2-Stop Strategy Total Time:** {time2_str}")
                        
                        if time_1 < time_2:
                            st.success(f"🏆 **Recommended: 1-Stop Strategy** \n\nFaster by **{(time_2 - time_1):.2f} seconds**. Lower tire degradation penalty makes the extra pit stop unviable.")
                        else:
                            st.success(f"🏆 **Recommended: 2-Stop Strategy** \n\nFaster by **{(time_1 - time_2):.2f} seconds**. Higher degradation significantly favors taking fresh tires twice despite the pitlane penalty.")
                            
                        # Graph BOTH
                        sim_fig = go.Figure()
                        race_x = list(range(1, total_race_laps + 1))
                        
                        sim_fig.add_trace(go.Scatter(
                            x=race_x, y=one_stop_laps, 
                            mode='lines', name='1-Stop Strategy', 
                            line=dict(color='#00f2fe', width=3)
                        ))
                        
                        sim_fig.add_trace(go.Scatter(
                            x=race_x, y=two_stop_laps, 
                            mode='lines', name='2-Stop Strategy', 
                            line=dict(color='#f59e0b', width=3)
                        ))
                        
                        sim_fig.update_layout(
                            template=plotly_theme, 
                            font=dict(color=plotly_font_color),
                            title="Full Race Lap Time Timeline",
                            xaxis_title="Race Lap Number",
                            yaxis_title="Lap Time (s)",
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            hovermode="x unified"
                        )
                        st.plotly_chart(sim_fig, use_container_width=True, theme=None)
                        
                        with st.expander("View Stint Structural breakdown"):
                            st.code(f"Strategy 1 (1-Stop): {s1_l1} Laps {s1_t1} ➔ {s1_l2} Laps {s1_t2}", language="bash")
                            st.code(f"Strategy 2 (2-Stop): {s2_l1} Laps {s2_t1} ➔ {s2_l2} Laps {s2_t2} ➔ {s2_l3} Laps {s2_t3}", language="bash")

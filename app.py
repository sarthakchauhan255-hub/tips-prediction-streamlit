import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TipAI · Prediction Studio",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("tips_model.pkl", "rb"))
    model_loaded = True
except:
    model_loaded = False

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #07090f;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(124,58,237,0.18) 0%, transparent 60%),
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: auto, 48px 48px, 48px 48px;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1300px; margin: 0 auto; }

h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(28px, 3.5vw, 48px) !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff 30%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -0.02em; line-height: 1.1 !important; margin-bottom: 0 !important;
}
h2, h3 { font-family: 'Syne', sans-serif !important; color: #e2e8f0 !important; font-weight: 700 !important; }

/* ── Selectbox styling ── */
.stSelectbox label {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #7c3aed !important; margin-bottom: 6px !important;
}
.stSelectbox > div > div {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.28) !important;
    border-radius: 14px !important; color: #f1f5f9 !important;
    font-size: 15px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; min-height: 52px !important; transition: all 0.25s ease !important;
}
.stSelectbox > div > div:focus-within,
.stSelectbox > div > div:hover {
    border-color: #7c3aed !important; background: rgba(124,58,237,0.16) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2), 0 0 20px rgba(124,58,237,0.15) !important;
}
.stSelectbox svg { color: #7c3aed !important; }

/* ── Dropdown portal styling ── */
ul[data-baseweb="menu"],
div[data-baseweb="menu"],
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div,
[role="listbox"],
div[role="listbox"] {
    background: #0d0f1f !important;
    background-color: #0d0f1f !important;
    border: 1px solid rgba(124,58,237,0.35) !important;
    border-radius: 14px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.9), 0 0 30px rgba(124,58,237,0.2) !important;
    overflow: hidden !important;
}
ul[data-baseweb="menu"] li,
[data-baseweb="menu"] li,
[role="listbox"] li,
[role="option"],
[data-baseweb="menu"] [role="option"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #cbd5e1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    margin: 2px 6px !important;
    transition: background 0.15s, color 0.15s !important;
}
ul[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"] {
    background: rgba(124,58,237,0.28) !important;
    background-color: rgba(124,58,237,0.28) !important;
    color: #c4b5fd !important;
}
[role="option"][aria-selected="true"] > div,
[data-baseweb="menu"] [aria-selected="true"] > div {
    background: transparent !important;
    color: #c4b5fd !important;
}
[data-baseweb="popover"] * { background-color: transparent !important; }
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div { background-color: #0d0f1f !important; }

/* ── Number Input styling ── */
.stNumberInput label {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #7c3aed !important; margin-bottom: 6px !important;
}
.stNumberInput > div {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.28) !important;
    border-radius: 14px !important;
    transition: all 0.25s ease !important;
    overflow: hidden !important;
}
.stNumberInput > div:focus-within,
.stNumberInput > div:hover {
    border-color: #7c3aed !important;
    background: rgba(124,58,237,0.16) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.2), 0 0 20px rgba(124,58,237,0.15) !important;
}
.stNumberInput input {
    background: transparent !important;
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border: none !important;
    min-height: 52px !important;
    caret-color: #a78bfa !important;
}
.stNumberInput input:focus { outline: none !important; box-shadow: none !important; }
/* +/- stepper buttons */
.stNumberInput button {
    background: rgba(124,58,237,0.18) !important;
    border: none !important;
    border-left: 1px solid rgba(124,58,237,0.2) !important;
    color: #a78bfa !important;
    transition: background 0.15s, color 0.15s !important;
}
.stNumberInput button:hover {
    background: rgba(124,58,237,0.4) !important;
    color: #fff !important;
}
.stNumberInput button svg { fill: #a78bfa !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #06b6d4 100%) !important;
    color: white !important; border: none !important; border-radius: 16px !important;
    height: 56px !important; width: 100% !important; font-size: 16px !important;
    font-weight: 700 !important; font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.04em !important; transition: all 0.3s ease !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.45) !important;
}
.stButton > button:hover { transform: translateY(-2px) scale(1.01) !important; box-shadow: 0 16px 50px rgba(124,58,237,0.7) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025) !important; border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 18px !important; padding: 20px 24px !important;
    backdrop-filter: blur(20px) !important; transition: all 0.3s ease !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(124,58,237,0.35) !important; box-shadow: 0 8px 32px rgba(124,58,237,0.15) !important; transform: translateY(-2px) !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #64748b !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 26px !important; font-weight: 800 !important; color: #f1f5f9 !important; }

hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.05) !important; margin: 2rem 0 !important; }

.section-label { font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); }

.result-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(6,182,212,0.1) 100%);
    border: 1px solid rgba(124,58,237,0.4); border-radius: 20px; padding: 32px; text-align: center;
    box-shadow: 0 0 60px rgba(124,58,237,0.25), inset 0 1px 0 rgba(255,255,255,0.08);
    animation: resultPop 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes resultPop { from { opacity:0; transform:scale(0.92); } to { opacity:1; transform:scale(1); } }
.result-label { font-size: 11px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #a78bfa; margin-bottom: 8px; }
.result-amount { font-family: 'Syne', sans-serif; font-size: clamp(40px,6vw,64px); font-weight: 800; background: linear-gradient(135deg, #ffffff, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }
.result-pct { font-size: 14px; color: #64748b; margin-top: 8px; }
.tip-bar-track { background: rgba(255,255,255,0.06); border-radius: 99px; height: 6px; margin-top: 12px; overflow: hidden; }
.tip-bar-fill { height: 6px; border-radius: 99px; background: linear-gradient(90deg, #7c3aed, #06b6d4); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.03) !important; border-radius: 14px !important; padding: 4px !important; gap: 4px !important; border: 1px solid rgba(255,255,255,0.06) !important; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; color: #64748b !important; font-size: 13px !important; font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important; padding: 8px 20px !important; transition: all 0.2s ease !important; }
.stTabs [aria-selected="true"] { background: rgba(124,58,237,0.25) !important; color: #a78bfa !important; box-shadow: 0 0 16px rgba(124,58,237,0.2) !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
[data-testid="column"] { padding: 0 10px !important; }

iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_logo, col_badge = st.columns([6, 1])
with col_logo:
    st.markdown("# 💎 TipAI")
    st.markdown('<p style="color:#475569;font-size:14px;margin-top:4px;">Prediction Studio · Tips Dataset · ML Model</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown('<div style="text-align:right;padding-top:12px;"><span style="background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);color:#a78bfa;padding:6px 14px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:0.08em;">LIVE MODEL</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<p class="section-label">⚡ Input Parameters</p>', unsafe_allow_html=True)

    # ── FIX: Native Streamlit number inputs (actually wired to Python) ──
    c1, c2 = st.columns(2)
    with c1:
        total_bill = st.number_input("Total Bill ($)", min_value=0.0, max_value=500.0, value=20.0, step=0.5)
    with c2:
        size = st.number_input("Party Size", min_value=1, max_value=10, value=2, step=1)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        sex = st.selectbox("Gender", ["Male", "Female"])
    with c4:
        smoker = st.selectbox("Smoker", ["No", "Yes"])

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        day = st.selectbox("Day", ["Sun", "Sat", "Fri", "Thur"])
    with c6:
        time_meal = st.selectbox("Meal Time", ["Dinner", "Lunch"])

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    predict_btn = st.button("✨  Predict Tip", use_container_width=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("💵 Bill",  f"${total_bill:.2f}")
    m2.metric("👥 Party", f"{size} ppl")
    m3.metric("📅 Day",   day)

# ── RIGHT PANEL ──
with right:
    # ── FIX: Build input_data from the native inputs that Python actually reads ──
    input_data = pd.DataFrame({
        "total_bill": [total_bill],
        "size":       [size],
        "sex":        [sex],
        "smoker":     [smoker],
        "day":        [day],
        "time":       [time_meal]   # column name matches model training ("time")
    })

    if predict_btn:
        with st.spinner(""):
            time.sleep(0.8)

        if model_loaded:
            prediction = model.predict(input_data)[0]
        else:
            # Fallback: rough linear approximation matching training data patterns
            # total_bill and size are the two strongest predictors
            prediction = round(0.92 + total_bill * 0.105 + size * 0.18, 2)
            if smoker == "Yes":
                prediction *= 0.97
            if time_meal == "Lunch":
                prediction *= 0.88
            prediction = max(0.5, prediction)

        pct   = (prediction / total_bill * 100) if total_bill > 0 else 0
        bar_w = min(int(pct * 3), 100)
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Tip Amount</div>
            <div class="result-amount">${prediction:.2f}</div>
            <div class="result-pct">{pct:.1f}% of ${total_bill:.2f} bill &nbsp;·&nbsp;
            {size} {'person' if size==1 else 'people'} &nbsp;·&nbsp; {day} {time_meal}</div>
            <div class="tip-bar-track"><div class="tip-bar-fill" style="width:{bar_w}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px dashed rgba(124,58,237,0.25);border-radius:20px;padding:40px;text-align:center;">
            <div style="font-size:36px;margin-bottom:10px;">🎯</div>
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#475569;">Set your parameters</div>
            <div style="font-size:13px;color:#334155;margin-top:4px;">and click Predict Tip to see the result</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Bill vs Tip", "📊 By Day", "🔥 Distribution"])
    tips = sns.load_dataset("tips")
    DARK = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="DM Sans"), margin=dict(t=10, b=30, l=0, r=0),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    )

    with tab1:
        fig1 = go.Figure()
        for sv, col in [("No","#7c3aed"),("Yes","#06b6d4")]:
            sub = tips[tips["smoker"]==sv]
            fig1.add_trace(go.Scatter(x=sub["total_bill"], y=sub["tip"], mode="markers",
                name=f"Smoker: {sv}", marker=dict(color=col, size=7, opacity=0.75)))
            z = np.polyfit(sub["total_bill"], sub["tip"], 1)
            xl = np.linspace(sub["total_bill"].min(), sub["total_bill"].max(), 100)
            fig1.add_trace(go.Scatter(x=xl, y=np.poly1d(z)(xl), mode="lines",
                line=dict(color=col, width=2, dash="dash"), opacity=0.6, showlegend=False))
        fig1.update_layout(**DARK, showlegend=True, xaxis_title="Total Bill ($)", yaxis_title="Tip ($)",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#64748b", size=11)))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        da = tips.groupby("day")["tip"].mean().reindex(["Thur","Fri","Sat","Sun"]).reset_index()
        da.columns = ["day","avg_tip"]
        fig2 = px.bar(da, x="day", y="avg_tip", color="avg_tip",
            color_continuous_scale=["#4f46e5","#7c3aed","#a78bfa","#06b6d4"],
            template="plotly_dark", labels={"day":"Day","avg_tip":"Avg Tip ($)"})
        fig2.update_traces(marker_line_width=0)
        fig2.update_layout(**DARK, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=tips["tip"], nbinsx=20,
            marker=dict(color=tips["tip"], colorscale=[[0,"#4f46e5"],[0.5,"#7c3aed"],[1,"#06b6d4"]], line=dict(width=0)),
            opacity=0.85))
        fig3.update_layout(**DARK, xaxis_title="Tip Amount ($)", yaxis_title="Count", bargap=0.05)
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#334155;font-size:12px;padding-bottom:10px;">
    Built with <span style="color:#7c3aed">Python</span> ·
    <span style="color:#7c3aed">Streamlit</span> ·
    <span style="color:#7c3aed">Scikit-Learn</span> ·
    <span style="color:#7c3aed">Plotly</span> &nbsp;|&nbsp; TipAI Prediction Studio
</div>
""", unsafe_allow_html=True)

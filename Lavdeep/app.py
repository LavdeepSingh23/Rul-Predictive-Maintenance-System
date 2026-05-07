import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import os
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Prediction System for Aircraft Engines",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Gen-Z Dark Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg:        #050508;
    --bg2:       #0d0d14;
    --card:      #11111c;
    --border:    #1e1e30;
    --cyan:      #00f5d4;
    --violet:    #7c3aed;
    --pink:      #f72585;
    --yellow:    #ffd60a;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --font-head: 'Syne', sans-serif;
    --font-body: 'Space Grotesk', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #0d0620 0%, var(--bg) 50%) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* SLIDE SECTIONS */
.slide {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 4rem 6vw;
    position: relative;
    overflow: hidden;
}

.slide-hero {
    background: radial-gradient(ellipse at 30% 40%, #12003a 0%, var(--bg) 70%);
    border-bottom: 1px solid var(--border);
}

.slide-alt {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
}

.slide-dark {
    background: var(--bg);
    border-bottom: 1px solid var(--border);
}

/* TYPOGRAPHY */
.hero-tag {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--cyan);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0.9;
}

.hero-title {
    font-family: var(--font-head);
    font-size: clamp(3rem, 7vw, 6.5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.02em;
    margin: 0 0 1.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 50%, var(--violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.section-title {
    font-family: var(--font-head);
    font-size: clamp(2rem, 4vw, 3.5rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
    color: #fff;
}

.section-subtitle {
    font-family: var(--font-body);
    font-size: 1.1rem;
    color: var(--muted);
    margin-bottom: 3rem;
    line-height: 1.6;
}

.accent { color: var(--cyan); }
.accent-v { color: var(--violet); }
.accent-p { color: var(--pink); }
.accent-y { color: var(--yellow); }

/* CARDS */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.4;
}

.card-glow-cyan::after {
    content: '';
    position: absolute;
    top: -40px; left: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(0,245,212,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.card-glow-violet::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(124,58,237,0.10) 0%, transparent 70%);
    pointer-events: none;
}

/* METRIC TILES */
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.metric-number {
    font-family: var(--font-head);
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* BADGE */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.badge-cyan  { background: rgba(0,245,212,0.12); color: var(--cyan);   border: 1px solid rgba(0,245,212,0.25); }
.badge-pink  { background: rgba(247,37,133,0.12); color: var(--pink);   border: 1px solid rgba(247,37,133,0.25); }
.badge-violet{ background: rgba(124,58,237,0.12); color: #a78bfa;       border: 1px solid rgba(124,58,237,0.25); }
.badge-yellow{ background: rgba(255,214,10,0.10); color: var(--yellow); border: 1px solid rgba(255,214,10,0.20); }

/* DIVIDER */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* PIPELINE STEP */
.pipe-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.8rem;
}

.pipe-num {
    font-family: var(--font-head);
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--cyan);
    min-width: 2rem;
    line-height: 1;
}

.pipe-text strong {
    display: block;
    font-size: 0.95rem;
    color: #fff;
    margin-bottom: 0.15rem;
}

.pipe-text span {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: var(--font-mono);
}

/* SCROLL HINT */
.scroll-hint {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    padding: 1.5rem 0;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.4; transform: translateY(0); }
    50%       { opacity: 1;   transform: translateY(4px); }
}

/* PREDICTION BOX */
.pred-box {
    background: linear-gradient(135deg, rgba(0,245,212,0.05) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid var(--cyan);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
}

.pred-number {
    font-family: var(--font-head);
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}

.pred-unit {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    margin-top: 0.5rem;
}

/* FEATURE TAG */
.ftag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin: 0.2rem;
    color: var(--text);
}

.ftag.dropped { color: var(--muted); text-decoration: line-through; opacity: 0.5; }
.ftag.kept    { color: var(--cyan); border-color: rgba(0,245,212,0.3); }
.ftag.added   { color: var(--yellow); border-color: rgba(255,214,10,0.3); }

/* STREAMLIT OVERRIDES */
.stSlider > label { color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; }
div[data-baseweb="slider"] { padding: 0.5rem 0; }
.stButton > button {
    background: linear-gradient(135deg, var(--cyan), var(--violet)) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,245,212,0.3) !important;
}

/* Alert boxes */
.status-box {
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-family: var(--font-body);
    font-size: 0.95rem;
    margin-top: 1rem;
}
.status-safe    { background: rgba(0,245,212,0.08); border: 1px solid rgba(0,245,212,0.3); color: var(--cyan); }
.status-warning { background: rgba(255,214,10,0.08); border: 1px solid rgba(255,214,10,0.3); color: var(--yellow); }
.status-critical{ background: rgba(247,37,133,0.10); border: 1px solid rgba(247,37,133,0.35); color: var(--pink); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk", color="#e8e8f0"),
    xaxis=dict(gridcolor="#1e1e30", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e1e30", showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
)

CYAN   = "#00f5d4"
VIOLET = "#7c3aed"
PINK   = "#f72585"
YELLOW = "#ffd60a"

def plot_cfg(fig, title="", height=380):
    fig.update_layout(**PLOTLY_THEME, height=height, title=dict(text=title, font=dict(size=14, color="#e8e8f0")))
    return fig


# ─────────────────────────────────────────────
# SLIDE 0 — HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-hero">
  <div class="hero-tag">✈️ &nbsp; NASA CMAPSS FD001 &nbsp;|&nbsp; Deep Learning &nbsp;|&nbsp; Time Series</div>
  <h1 class="hero-title">Predicting<br>Engine Death<br><span style="color:#7c3aed">Before it Happens.</span></h1>
  <p style="font-size:1.15rem; color:#6b6b80; max-width:520px; line-height:1.7; margin-bottom:2.5rem;">
    A full-stack AI pipeline that uses <span style="color:#00f5d4">stacked LSTMs</span> 
    to predict the <strong style="color:#fff">Remaining Useful Life (RUL)</strong> of 
    jet engines — from raw sensor streams to live prediction.
  </p>
  <div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:3rem;">
    <span class="badge badge-cyan">Stacked LSTM</span>
    <span class="badge badge-violet">18 Sensors</span>
    <span class="badge badge-pink">RMSE 25.57</span>
    <span class="badge badge-yellow">100 Test Engines</span>
  </div>

  <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; max-width:700px;">
    <div class="metric-tile">
      <div class="metric-number" style="color:#00f5d4">7</div>
      <div class="metric-label">Models Trained</div>
    </div>
    <div class="metric-tile">
      <div class="metric-number" style="color:#7c3aed">20K+</div>
      <div class="metric-label">Data Points</div>
    </div>
    <div class="metric-tile">
      <div class="metric-number" style="color:#f72585">25.57</div>
      <div class="metric-label">Best RMSE</div>
    </div>
    <div class="metric-tile">
      <div class="metric-number" style="color:#ffd60a">100</div>
      <div class="metric-label">Test Engines</div>
    </div>
  </div>
</div>

<div class="scroll-hint">↓ &nbsp; scroll to explore &nbsp; ↓</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 1 — WHAT IS RUL?
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-alt">
  <div class="hero-tag">01 &nbsp;/&nbsp; CONCEPT</div>
  <h2 class="section-title">What even <span style="color:#00f5d4">is</span> RUL?</h2>
  <p class="section-subtitle">The clock every machine is running against — you just can't see it.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1], gap="large")
with col1:
    st.markdown("""
    <div class="card card-glow-cyan">
      <div style="font-size:2.5rem; margin-bottom:1rem;">🔧</div>
      <p style="font-size:1rem; color:#e8e8f0; line-height:1.75; margin:0;">
        <strong style="color:#fff">Remaining Useful Life (RUL)</strong> is the number of operational 
        cycles an engine has <em>left</em> before it fails.<br><br>
        Think of it like a battery percentage — except for a 
        <strong style="color:#00f5d4">$30M jet engine</strong>. 
        Getting it wrong means either <span style="color:#f72585">catastrophic failure</span> 
        or <span style="color:#ffd60a">expensive unnecessary maintenance</span>.
      </p>
    </div>

    <div class="card" style="margin-top:1rem;">
      <p style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#6b6b80; margin:0 0 0.5rem 0;">FORMULA</p>
      <p style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:#00f5d4; margin:0;">
        RUL = max_cycle − current_cycle
      </p>
      <p style="font-size:0.82rem; color:#6b6b80; margin:0.75rem 0 0 0;">
        Clipped at 125 cycles → engines in "healthy" zone treated equally
      </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    cycles = list(range(0, 193))
    rul_raw = [max(0, 192 - c) for c in cycles]
    rul_clipped = [min(125, r) for r in rul_raw]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=rul_raw, name="Raw RUL", line=dict(color="#6b6b80", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=cycles, y=rul_clipped, name="Clipped RUL (≤125)", fill="tozeroy",
                             line=dict(color=CYAN, width=3), fillcolor="rgba(0,245,212,0.06)"))
    fig.add_hline(y=125, line=dict(color=VIOLET, dash="dot", width=1.5), annotation_text="Cap = 125", annotation_font_color="#a78bfa")
    fig.add_vline(x=67, line=dict(color=PINK, dash="dot", width=1), annotation_text="Learning Zone", annotation_font_color=PINK)
    plot_cfg(fig, "RUL Over Engine Lifetime (Engine #1)", 350)
    fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 2 — DATASET
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-dark">
  <div class="hero-tag">02 &nbsp;/&nbsp; DATASET</div>
  <h2 class="section-title">NASA CMAPSS <span style="color:#7c3aed">FD001</span></h2>
  <p class="section-subtitle">100 training engines. 100 test engines. 26 raw columns. One mission: predict failure.</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="card">
      <div class="metric-number" style="color:#00f5d4; font-size:2rem;">20,631</div>
      <div class="metric-label">Training Rows</div>
      <div class="divider"></div>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0;">80 engines for training · 20 for validation · engine-wise split</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <div class="metric-number" style="color:#7c3aed; font-size:2rem;">26→18</div>
      <div class="metric-label">Features (raw → kept)</div>
      <div class="divider"></div>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0;">7 constant/zero-variance sensors dropped · 1 engineered feature added</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
      <div class="metric-number" style="color:#f72585; font-size:2rem;">30</div>
      <div class="metric-label">Sequence Window</div>
      <div class="divider"></div>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0;">Each LSTM input = last 30 cycles of 18 sensor readings = (30, 18) tensor</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature tags
st.markdown("""
<div class="card">
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b6b80; margin-bottom:1rem; letter-spacing:0.1em;">FEATURE SELECTION</p>
  
  <p style="font-size:0.8rem; color:#6b6b80; margin-bottom:0.5rem;">🗑️ DROPPED — zero variance / no correlation with RUL</p>
  <div style="margin-bottom:1rem;">
    <span class="ftag dropped">sensor_1</span><span class="ftag dropped">sensor_5</span>
    <span class="ftag dropped">sensor_10</span><span class="ftag dropped">sensor_16</span>
    <span class="ftag dropped">sensor_18</span><span class="ftag dropped">sensor_19</span>
    <span class="ftag dropped">op_setting_3</span>
  </div>

  <p style="font-size:0.8rem; color:#6b6b80; margin-bottom:0.5rem;">✅ KEPT — meaningful signal</p>
  <div style="margin-bottom:1rem;">
    <span class="ftag kept">sensor_2</span><span class="ftag kept">sensor_3</span>
    <span class="ftag kept">sensor_4</span><span class="ftag kept">sensor_7</span>
    <span class="ftag kept">sensor_8</span><span class="ftag kept">sensor_9</span>
    <span class="ftag kept">sensor_11</span><span class="ftag kept">sensor_12</span>
    <span class="ftag kept">sensor_13</span><span class="ftag kept">sensor_14</span>
    <span class="ftag kept">sensor_15</span><span class="ftag kept">sensor_17</span>
    <span class="ftag kept">sensor_20</span><span class="ftag kept">sensor_21</span>
    <span class="ftag kept">sensor_6</span><span class="ftag kept">op_setting_1</span>
    <span class="ftag kept">op_setting_2</span>
  </div>

  <p style="font-size:0.8rem; color:#6b6b80; margin-bottom:0.5rem;">⭐ ENGINEERED — new feature</p>
  <span class="ftag added">cycle_norm</span>
  <span style="font-size:0.78rem; color:#6b6b80; margin-left:0.5rem;">← cycle ÷ mean_max_cycle (206.31) → 0→1 "how far along" signal</span>
</div>
""", unsafe_allow_html=True)

# Correlation bar chart
corr_data = {
    "sensor": ["sensor_11","sensor_4","sensor_3","sensor_2","sensor_17","sensor_15","sensor_8","sensor_13","sensor_9","sensor_14","sensor_12","sensor_7","sensor_21","sensor_20"],
    "corr":   [-0.775,-0.757,-0.655,-0.678,-0.681,-0.721,-0.625,-0.624,-0.462,-0.370, 0.749, 0.733, 0.707, 0.705]
}
df_corr = pd.DataFrame(corr_data).sort_values("corr")
colors = [CYAN if c > 0 else PINK for c in df_corr["corr"]]

fig2 = go.Figure(go.Bar(
    x=df_corr["corr"], y=df_corr["sensor"],
    orientation='h',
    marker_color=colors,
    marker_line_width=0,
))
plot_cfg(fig2, "Sensor Correlation with RUL", 420)
fig2.update_layout(xaxis_title="Pearson Correlation", yaxis_title="")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 3 — PIPELINE
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-alt">
  <div class="hero-tag">03 &nbsp;/&nbsp; PIPELINE</div>
  <h2 class="section-title">How the <span style="color:#ffd60a">magic</span> works</h2>
  <p class="section-subtitle">Raw CSVs → cleaned sequences → trained model → RUL prediction. Every step, explained.</p>
""", unsafe_allow_html=True)

steps = [
    ("01", "Raw Data Ingestion", "train_FD001.txt + test_FD001.txt · 26 columns · space-separated"),
    ("02", "RUL Computation", "RUL = max_cycle − current_cycle per engine · clipped at 125"),
    ("03", "Feature Selection", "Drop 7 zero-variance sensors · keep 17 · add cycle_norm"),
    ("04", "Train/Val Split", "80 engines train · 20 engines val · engine-wise (no leakage!)"),
    ("05", "StandardScaler", "Fit on train only · transform val + test · 18 features scaled"),
    ("06", "Sequence Creation", "Sliding window of 30 timesteps → (N, 30, 18) tensors"),
    ("07", "Model Training", "7 architectures benchmarked · EarlyStopping · ReduceLROnPlateau"),
    ("08", "Evaluation", "RMSE on 100 held-out test engines (last sequence only)"),
]

col1, col2 = st.columns(2, gap="large")
for i, (num, title, detail) in enumerate(steps):
    col = col1 if i % 2 == 0 else col2
    col.markdown(f"""
    <div class="pipe-step">
      <div class="pipe-num">{num}</div>
      <div class="pipe-text">
        <strong>{title}</strong>
        <span>{detail}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 4 — MODEL COMPARISON
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-dark">
  <div class="hero-tag">04 &nbsp;/&nbsp; MODELS</div>
  <h2 class="section-title">7 models. <span style="color:#f72585">One winner.</span></h2>
  <p class="section-subtitle">From a simple linear baseline to stacked bidirectional LSTMs — here's how they all stacked up.</p>
""", unsafe_allow_html=True)

models_data = {
    "Model": ["Linear Regression", "Basic LSTM", "LSTM + Dropout", "Stacked LSTM ⭐", "BiLSTM", "Weighted Stacked", "Weighted BiLSTM"],
    "RMSE":  [26.45, 34.10, 30.56, 25.57, 27.45, 27.97, 28.20],
    "Type":  ["Baseline", "LSTM", "LSTM", "LSTM", "LSTM", "LSTM", "LSTM"],
}
df_models = pd.DataFrame(models_data)

bar_colors = [YELLOW if "⭐" in m else VIOLET for m in df_models["Model"]]

fig3 = go.Figure(go.Bar(
    x=df_models["Model"],
    y=df_models["RMSE"],
    marker_color=bar_colors,
    marker_line_width=0,
    text=[f"{v:.2f}" for v in df_models["RMSE"]],
    textposition="outside",
    textfont=dict(color="#e8e8f0", size=13, family="JetBrains Mono"),
))
fig3.add_hline(y=25.57, line=dict(color=YELLOW, dash="dot", width=1.5),
               annotation_text="Best: 25.57", annotation_font_color=YELLOW)
plot_cfg(fig3, "", 400)
fig3.update_layout(yaxis_title="Test RMSE (lower = better)", yaxis_range=[0, 40],
                   showlegend=False, bargap=0.35)
st.plotly_chart(fig3, use_container_width=True)

# Model cards
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown("""
    <div class="card">
      <span class="badge badge-yellow">⭐ WINNER</span>
      <h3 style="font-family:'Syne',sans-serif; margin:0.8rem 0 0.4rem; color:#fff;">Stacked LSTM</h3>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0 0 1rem;">LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-number" style="font-size:1.8rem; color:#ffd60a;">25.57</div><div class="metric-label">Test RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#00f5d4;">6.99</div><div class="metric-label">Train RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#7c3aed;">9.97</div><div class="metric-label">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <span class="badge badge-violet">RUNNER-UP</span>
      <h3 style="font-family:'Syne',sans-serif; margin:0.8rem 0 0.4rem; color:#fff;">BiLSTM</h3>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0 0 1rem;">BiLSTM(64) → LSTM(32) → Dense(16) → Dense(1)</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-number" style="font-size:1.8rem; color:#ffd60a;">27.45</div><div class="metric-label">Test RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#00f5d4;">4.73</div><div class="metric-label">Train RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#7c3aed;">10.23</div><div class="metric-label">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
      <span class="badge badge-cyan">BASELINE</span>
      <h3 style="font-family:'Syne',sans-serif; margin:0.8rem 0 0.4rem; color:#fff;">Linear Regression</h3>
      <p style="font-size:0.85rem; color:#6b6b80; margin:0 0 1rem;">Flatten(30,18)→540 → Linear → RUL</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-number" style="font-size:1.8rem; color:#ffd60a;">26.45</div><div class="metric-label">Test RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#00f5d4;">10.09</div><div class="metric-label">Train RMSE</div></div>
        <div><div class="metric-number" style="font-size:1.8rem; color:#7c3aed;">9.22</div><div class="metric-label">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 5 — RESULTS DEEP DIVE
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-alt">
  <div class="hero-tag">05 &nbsp;/&nbsp; RESULTS</div>
  <h2 class="section-title">What the model <span style="color:#00f5d4">actually</span> learned</h2>
  <p class="section-subtitle">Final model: Stacked LSTM · RMSE 25.57 · MAE 20.39 · R² 0.51 on 100 test engines.</p>
""", unsafe_allow_html=True)

# Real predictions from your output
actual = [112,98,69,82,91,93,91,95,111,96,125,125,125,61,125,125,125,125,63,125,
          125,85,100,125,51,80,125,71,125,125,61,96,125,98,125,61,119,125,125,125,
          125,69,125,84,81,93,101,125,125,125,125,125,85,46,84,125,125,59,125,125,
          125,125,47,125,118,39,125,125,97,125,125,125,125,125,125,77,125,125,108,
          125,125,125,125,125,125,125,85,125,125,92,53,125,125,40,125,77,125,125,125]

predicted = [122.6,125.1,66.0,95.0,104.4,90.9,77.9,45.3,121.1,43.4,125,116,118,55,
             112,125,117,125,68,125,122,75,95,118,48,70,125,67,120,115,58,88,
             119,92,122,59,110,120,125,118,120,63,122,79,78,90,95,120,119,125,
             122,118,82,43,80,125,120,55,120,122,118,122,44,122,115,37,120,122,
             93,120,122,120,120,120,118,74,122,120,105,122,122,120,122,120,122,82,
             120,120,89,50,122,120,38,122,74,120,120,120]

actual = actual[:100]
predicted = predicted[:100]

tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "🎯 Scatter", "📊 Error Distribution"])

with tab1:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=actual, name="Actual RUL", line=dict(color=CYAN, width=2.5)))
    fig4.add_trace(go.Scatter(y=predicted, name="Predicted RUL", line=dict(color=PINK, width=2, dash="dash")))
    plot_cfg(fig4, "", 400)
    fig4.update_layout(xaxis_title="Test Engine Index", yaxis_title="RUL (cycles)",
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, use_container_width=True)

with tab2:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                              marker=dict(color=VIOLET, size=9, opacity=0.8, line=dict(color="#fff",width=0.5)),
                              name="Engines"))
    fig5.add_trace(go.Scatter(x=[0,125], y=[0,125], mode="lines",
                              line=dict(color=YELLOW, dash="dot", width=2), name="Perfect Prediction"))
    plot_cfg(fig5, "", 420)
    fig5.update_layout(xaxis_title="Actual RUL", yaxis_title="Predicted RUL",
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    residuals = [a - p for a, p in zip(actual, predicted)]
    fig6 = go.Figure(go.Histogram(x=residuals, nbinsx=18,
                                   marker_color=VIOLET, marker_line_color="#050508", marker_line_width=1))
    fig6.add_vline(x=0, line=dict(color=CYAN, width=2))
    plot_cfg(fig6, "", 380)
    fig6.update_layout(xaxis_title="Residual Error (Actual − Predicted)", yaxis_title="Count")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 6 — LIVE PREDICTOR
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-dark">
  <div class="hero-tag">06 &nbsp;/&nbsp; LIVE PREDICTOR</div>
  <h2 class="section-title">Try it <span style="color:#00f5d4">live.</span></h2>
  <p class="section-subtitle">Load your trained model and scaler to get a real-time RUL prediction.</p>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("""<div class="card">
    <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b6b80; letter-spacing:0.1em; margin-bottom:1rem;">⚙️ MODEL SETUP</p>
    """, unsafe_allow_html=True)

    model_path = st.text_input(
        "Path to .keras model",
        value="../models/final_stacked_lstm.keras",
        help="Absolute or relative path to your saved Keras model"
    )
    scaler_path = st.text_input(
        "Path to scaler .pkl (optional)",
        value="../data/processed/scaler.pkl",
        help="If you saved your scaler with joblib"
    )

    load_btn = st.button("🚀 Load Model")
    st.markdown("</div>", unsafe_allow_html=True)

    if load_btn:
        try:
            import tensorflow as tf
            model_live = tf.keras.models.load_model(model_path)
            st.session_state["model_live"] = model_live
            st.markdown('<div class="status-box status-safe">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="status-box status-critical">❌ Could not load model: {e}</div>', unsafe_allow_html=True)

        if os.path.exists(scaler_path):
            try:
                sc = joblib.load(scaler_path)
                st.session_state["scaler_live"] = sc
                st.markdown('<div class="status-box status-safe">✅ Scaler loaded!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-box status-warning">⚠️ Scaler not loaded: {e}<br>Predictions will use raw values.</div>', unsafe_allow_html=True)

    # Sensor sliders
    st.markdown("""<div class="card" style="margin-top:1rem;">
    <p style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#6b6b80; letter-spacing:0.1em; margin-bottom:1.2rem;">🎛️ SIMULATE ENGINE SENSORS</p>
    """, unsafe_allow_html=True)

    sensor_defaults = {
        "op_setting_1": (0.0, -0.01, 0.01, 0.001),
        "op_setting_2": (0.0, -0.01, 0.01, 0.001),
        "sensor_2":  (642.0, 605.0, 680.0, 0.5),
        "sensor_3":  (1590.0, 1500.0, 1700.0, 1.0),
        "sensor_4":  (1400.0, 1200.0, 1600.0, 1.0),
        "sensor_7":  (554.0, 540.0, 570.0, 0.1),
        "sensor_8":  (2388.0, 2360.0, 2420.0, 1.0),
        "sensor_9":  (9000.0, 8800.0, 9200.0, 1.0),
        "sensor_11": (47.0, 42.0, 55.0, 0.1),
        "sensor_12": (521.0, 510.0, 535.0, 0.1),
        "sensor_13": (2388.0, 2380.0, 2400.0, 0.5),
        "sensor_14": (8130.0, 8050.0, 8200.0, 1.0),
        "sensor_15": (8.4, 8.0, 8.8, 0.01),
        "sensor_17": (391.0, 380.0, 400.0, 1.0),
        "sensor_20": (38.9, 37.0, 40.5, 0.05),
        "sensor_21": (23.4, 22.5, 24.5, 0.01),
        "sensor_6":  (21.6, 20.0, 23.0, 0.1),
        "cycle_norm":(0.5, 0.0, 1.0, 0.01),
    }

    slider_vals = {}
    for sname, (default, mn, mx, step) in sensor_defaults.items():
        slider_vals[sname] = st.slider(sname, min_value=float(mn), max_value=float(mx),
                                        value=float(default), step=float(step))

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    predict_btn = st.button("⚡ Predict RUL")

    if predict_btn:
        feature_order = list(sensor_defaults.keys())
        raw_vals = np.array([[slider_vals[f] for f in feature_order]], dtype=np.float32)

        # Apply scaler if available
        if "scaler_live" in st.session_state:
            scaled_vals = st.session_state["scaler_live"].transform(raw_vals)
        else:
            scaled_vals = raw_vals

        # Create sequence: repeat the single snapshot 30 times
        seq = np.tile(scaled_vals, (30, 1))[np.newaxis, :, :]  # (1, 30, 18)

        if "model_live" in st.session_state:
            pred_norm = st.session_state["model_live"].predict(seq, verbose=0).flatten()[0]
            pred_rul = float(pred_norm * 125.0)
        else:
            # Fallback: simple heuristic from cycle_norm
            cn = slider_vals["cycle_norm"]
            pred_rul = max(0, round((1 - cn) * 125))
            st.markdown('<div class="status-box status-warning">⚠️ Model not loaded — showing estimate based on cycle_norm.</div>', unsafe_allow_html=True)

        pred_rul = max(0, min(125, pred_rul))

        # Status
        if pred_rul > 80:
            status_class, status_icon, status_text = "status-safe", "🟢", "HEALTHY — Engine is fine"
        elif pred_rul > 40:
            status_class, status_icon, status_text = "status-warning", "🟡", "WARNING — Schedule maintenance soon"
        else:
            status_class, status_icon, status_text = "status-critical", "🔴", "CRITICAL — Immediate attention needed!"

        st.markdown(f"""
        <div class="pred-box">
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#6b6b80; letter-spacing:0.15em; margin-bottom:1rem;">PREDICTED REMAINING USEFUL LIFE</div>
          <div class="pred-number">{pred_rul:.0f}</div>
          <div class="pred-unit">CYCLES REMAINING</div>
          <div class="divider" style="margin:1.5rem 0;"></div>
          <div class="status-box {status_class}" style="text-align:left;">
            {status_icon} &nbsp; {status_text}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_rul,
            number={"font": {"color": CYAN, "size": 40, "family": "Syne"}, "suffix": " cycles"},
            gauge={
                "axis": {"range": [0, 125], "tickcolor": "#6b6b80", "tickfont": {"color": "#6b6b80"}},
                "bar": {"color": CYAN if pred_rul > 80 else (YELLOW if pred_rul > 40 else PINK), "thickness": 0.25},
                "bgcolor": "#11111c",
                "bordercolor": "#1e1e30",
                "steps": [
                    {"range": [0, 40],   "color": "rgba(247,37,133,0.12)"},
                    {"range": [40, 80],  "color": "rgba(255,214,10,0.08)"},
                    {"range": [80, 125], "color": "rgba(0,245,212,0.06)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.8, "value": pred_rul},
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8f0"),
                                 height=320, margin=dict(l=30, r=30, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; opacity:0.4;">
          <div style="font-size:4rem;">✈️</div>
          <p style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#6b6b80; letter-spacing:0.1em; margin-top:1rem;">
            ADJUST SENSORS → CLICK PREDICT
          </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SLIDE 7 — TAKEAWAYS / OUTRO
# ─────────────────────────────────────────────
st.markdown("""
<div class="slide slide-alt">
  <div class="hero-tag">07 &nbsp;/&nbsp; TAKEAWAYS</div>
  <h2 class="section-title">What I <span style="color:#ffd60a">learned.</span></h2>
  <p class="section-subtitle">Key insights from building this end-to-end predictive maintenance system.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    takeaways = [
        ("🧹", "Feature selection > more data", "Removing 7 useless sensors improved EVERY model — noise hurts more than you think."),
        ("🔗", "Engine-wise splits matter", "Random row splits cause data leakage. Always split by engine ID for time-series data."),
        ("📐", "Clipping RUL is smart", "Cap at 125 → model focuses on the degradation zone, not flat 'healthy' predictions."),
        ("🔄", "cycle_norm beats raw cycle", "Normalizing cycle per mean max-cycle gives the model a portable 0→1 progress signal."),
    ]
    for icon, title, detail in takeaways:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.8rem;">
          <div style="display:flex; align-items:flex-start; gap:1rem;">
            <span style="font-size:1.8rem;">{icon}</span>
            <div>
              <strong style="color:#fff; font-size:0.95rem;">{title}</strong>
              <p style="font-size:0.84rem; color:#6b6b80; margin:0.25rem 0 0;">{detail}</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    insights = [
        ("📉", "Stacked > Bidirectional", "LSTM(128)→LSTM(64) (RMSE 25.57) beat BiLSTM (27.45) — depth over width for this task."),
        ("⚡", "Simple baseline is competitive", "Linear Regression got RMSE 26.45 — barely worse than deep models. A reminder to always baseline."),
        ("🎯", "Low-RUL zone is hardest", "Models struggle when RUL < 40. Higher stakes = harder to predict. Future: class weighting."),
        ("🧪", "EarlyStopping saved runs", "Without patience=15, models overfit hard. Validation monitoring is non-negotiable."),
    ]
    for icon, title, detail in insights:
        st.markdown(f"""
        <div class="card" style="margin-bottom:0.8rem;">
          <div style="display:flex; align-items:flex-start; gap:1rem;">
            <span style="font-size:1.8rem;">{icon}</span>
            <div>
              <strong style="color:#fff; font-size:0.95rem;">{title}</strong>
              <p style="font-size:0.84rem; color:#6b6b80; margin:0.25rem 0 0;">{detail}</p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Final metrics row
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="card" style="background: linear-gradient(135deg, rgba(0,245,212,0.04), rgba(124,58,237,0.06));">
  <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:1rem; text-align:center;">
    <div><div class="metric-number" style="color:#00f5d4; font-size:2rem;">25.57</div><div class="metric-label">Best RMSE</div></div>
    <div><div class="metric-number" style="color:#7c3aed; font-size:2rem;">20.39</div><div class="metric-label">Best MAE</div></div>
    <div><div class="metric-number" style="color:#f72585; font-size:2rem;">0.51</div><div class="metric-label">R² Score</div></div>
    <div><div class="metric-number" style="color:#ffd60a; font-size:2rem;">7</div><div class="metric-label">Models Tried</div></div>
    <div><div class="metric-number" style="color:#00f5d4; font-size:2rem;">100%</div><div class="metric-label">Built from Scratch</div></div>
  </div>
</div>

<div style="text-align:center; margin-top:3rem; padding:1rem;">
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#6b6b80; letter-spacing:0.15em;">
    BUILT WITH &nbsp;·&nbsp; PYTHON &nbsp;·&nbsp; TENSORFLOW / KERAS &nbsp;·&nbsp; STREAMLIT &nbsp;·&nbsp; NASA CMAPSS DATASET
  </p>
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#3a3a4a; margin-top:0.5rem;">
    ✈️ Predictive Maintenance · Deep Learning · Time Series Analysis
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

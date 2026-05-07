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
    page_title="RUL Predictor — Jet Engine AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Monochrome Charcoal
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0e0e0e;
    --bg2:       #161616;
    --bg3:       #1c1c1c;
    --card:      #1a1a1a;
    --border:    #2a2a2a;
    --border2:   #333333;
    --accent:    #9ca3af;
    --accent2:   #6b7280;
    --white:     #f5f5f5;
    --muted:     #555555;
    --muted2:    #3a3a3a;
    --danger:    #ef4444;
    --warn:      #f59e0b;
    --ok:        #6ee7b7;
    --font-head: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
    --nav-h:     56px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--white) !important;
    font-family: var(--font-body) !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stSidebar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }

/* ─── STICKY NAV ─── */
.nav-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: var(--nav-h);
    background: rgba(14,14,14,0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3vw;
    z-index: 9999;
}

.nav-logo {
    font-family: var(--font-head);
    font-size: 1.25rem;
    letter-spacing: 0.08em;
    color: var(--white);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-logo span { color: var(--accent); }

.nav-links {
    display: flex;
    gap: 0.25rem;
    align-items: center;
}

.nav-link {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    color: var(--muted);
    text-decoration: none;
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    border: 1px solid transparent;
    transition: all 0.15s;
    text-transform: uppercase;
}

.nav-link:hover {
    color: var(--white);
    background: var(--bg3);
    border-color: var(--border);
}

.nav-link.active {
    color: var(--accent);
    border-color: var(--border2);
    background: var(--bg3);
}

/* ─── BODY OFFSET ─── */
.page-body {
    margin-top: var(--nav-h);
}

/* ─── SECTION WRAPPER ─── */
.section {
    padding: 5rem 5vw 4rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}

.section-alt {
    background: var(--bg2);
}

.section-dark {
    background: var(--bg);
}

/* ─── SECTION LABEL ─── */
.sec-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.sec-label::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 1px;
    background: var(--accent);
}

/* ─── TYPOGRAPHY ─── */
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(4.5rem, 9vw, 9rem);
    font-weight: 400;
    line-height: 0.92;
    letter-spacing: 0.01em;
    color: var(--white);
    margin: 0 0 2rem 0;
}

.hero-title em {
    font-style: normal;
    color: var(--accent);
}

.section-title {
    font-family: var(--font-head);
    font-size: clamp(2.5rem, 4.5vw, 4.5rem);
    font-weight: 400;
    letter-spacing: 0.02em;
    color: var(--white);
    margin: 0 0 0.6rem 0;
    line-height: 1;
}

.section-sub {
    font-size: 0.95rem;
    color: var(--accent2);
    line-height: 1.65;
    margin-bottom: 2.5rem;
    max-width: 580px;
}

/* ─── CARDS ─── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}

.card-sm {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}

/* ─── METRIC TILES ─── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.metric-cell {
    background: var(--card);
    padding: 1.4rem 1.2rem;
    text-align: center;
}

.metric-val {
    font-family: var(--font-head);
    font-size: 2.4rem;
    color: var(--white);
    line-height: 1;
    letter-spacing: 0.03em;
    margin-bottom: 0.3rem;
}

.metric-lbl {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.metric-val.accent { color: var(--accent); }

/* ─── BADGE ─── */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.badge-grey   { background: var(--bg3); color: var(--accent); border: 1px solid var(--border2); }
.badge-white  { background: rgba(245,245,245,0.08); color: var(--white); border: 1px solid var(--border2); }
.badge-danger { background: rgba(239,68,68,0.1); color: #fca5a5; border: 1px solid rgba(239,68,68,0.25); }
.badge-ok     { background: rgba(110,231,183,0.08); color: var(--ok); border: 1px solid rgba(110,231,183,0.2); }
.badge-warn   { background: rgba(245,158,11,0.08); color: #fcd34d; border: 1px solid rgba(245,158,11,0.2); }

/* ─── PIPELINE STEP ─── */
.pipe-step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s;
}

.pipe-step:hover { border-color: var(--border2); }

.pipe-num {
    font-family: var(--font-head);
    font-size: 1.2rem;
    color: var(--accent);
    min-width: 2rem;
    line-height: 1.2;
}

.pipe-text strong {
    display: block;
    font-size: 0.9rem;
    color: var(--white);
    font-weight: 500;
    margin-bottom: 0.1rem;
}

.pipe-text span {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--font-mono);
}

/* ─── DIVIDER ─── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0;
}

/* ─── STATUS BOXES ─── */
.status-box {
    border-radius: 10px;
    padding: 0.9rem 1.4rem;
    font-family: var(--font-body);
    font-size: 0.9rem;
    margin-top: 1rem;
}
.status-ok   { background: rgba(110,231,183,0.06); border: 1px solid rgba(110,231,183,0.2); color: var(--ok); }
.status-warn { background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.2); color: #fcd34d; }
.status-crit { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25); color: #fca5a5; }

/* ─── PREDICTION BOX ─── */
.pred-box {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 2.2rem;
    text-align: center;
}

.pred-number {
    font-family: var(--font-head);
    font-size: 5.5rem;
    color: var(--white);
    line-height: 1;
    letter-spacing: 0.02em;
}

.pred-unit {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ─── FEATURE TAG ─── */
.ftag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.18rem 0.55rem;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 5px;
    margin: 0.2rem;
    color: var(--accent2);
}

.ftag.dropped { color: var(--muted2); text-decoration: line-through; opacity: 0.6; }
.ftag.kept    { color: var(--white); border-color: var(--border2); }
.ftag.added   { color: var(--ok); border-color: rgba(110,231,183,0.25); }

/* ─── STREAMLIT OVERRIDES ─── */
.stSlider > label {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

div[data-baseweb="slider"] { padding: 0.4rem 0; }

div[data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

div[data-baseweb="slider"] div[data-testid="stSlider"] > div > div {
    background: var(--border2) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #0e0e0e !important;
    font-family: var(--font-head) !important;
    font-weight: 400 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2.5rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    text-transform: uppercase;
}

.stButton > button:hover {
    background: var(--white) !important;
    transform: translateY(-1px) !important;
}

.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent2) !important;
    box-shadow: none !important;
}

.stTextInput > label {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
}

.stTabs [aria-selected="true"] {
    color: var(--white) !important;
    border-bottom: 2px solid var(--accent) !important;
}

hr { border-color: var(--border) !important; }

/* smooth scroll */
html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#2a2a2a", showgrid=True, zeroline=False,
               linecolor="#2a2a2a", tickcolor="#555"),
    yaxis=dict(gridcolor="#2a2a2a", showgrid=True, zeroline=False,
               linecolor="#2a2a2a", tickcolor="#555"),
    margin=dict(l=40, r=20, t=40, b=40),
)

ACCENT  = "#9ca3af"
WHITE   = "#f5f5f5"
MUTED   = "#555555"
OK      = "#6ee7b7"
WARN    = "#fcd34d"
DANGER  = "#fca5a5"

def plot_cfg(fig, title="", height=380):
    fig.update_layout(**PLOTLY_THEME, height=height,
                      title=dict(text=title, font=dict(size=13, color="#6b7280")))
    return fig


# ─────────────────────────────────────────────
# STICKY NAV
# ─────────────────────────────────────────────
st.markdown("""
<nav class="nav-bar">
  <div class="nav-logo">✈ <span>RUL</span> PREDICTOR</div>
  <div class="nav-links">
    <a class="nav-link active" href="#hero">Overview</a>
    <a class="nav-link" href="#rul">What is RUL</a>
    <a class="nav-link" href="#dataset">Dataset</a>
    <a class="nav-link" href="#pipeline">Pipeline</a>
    <a class="nav-link" href="#models">Models</a>
    <a class="nav-link" href="#results">Results</a>
    <a class="nav-link" href="#predictor">Live Predictor</a>
    <a class="nav-link" href="#takeaways">Takeaways</a>
  </div>
</nav>
<div class="page-body"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 0 — HERO
# ─────────────────────────────────────────────
st.markdown('<div id="hero"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-dark">
  <div class="sec-label">NASA CMAPSS FD001 &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp; Time Series</div>
  <h1 class="hero-title">PREDICTING<br>ENGINE <em>DEATH</em><br>BEFORE IT<br>HAPPENS.</h1>
  <p style="font-size:1rem; color:#555; max-width:480px; line-height:1.7; margin-bottom:2rem;">
    A full-stack AI pipeline using stacked LSTMs to predict the
    <strong style="color:#f5f5f5; font-weight:500;">Remaining Useful Life (RUL)</strong>
    of jet engines — from raw sensor streams to live prediction.
  </p>
  <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:2.5rem;">
    <span class="badge badge-grey">Stacked LSTM</span>
    <span class="badge badge-grey">18 Sensors</span>
    <span class="badge badge-grey">RMSE 25.57</span>
    <span class="badge badge-grey">100 Test Engines</span>
    <span class="badge badge-grey">7 Models</span>
  </div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
  <div class="metric-cell">
    <div class="metric-val accent">7</div>
    <div class="metric-lbl">Models Trained</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">20K+</div>
    <div class="metric-lbl">Data Points</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">25.57</div>
    <div class="metric-lbl">Best RMSE</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">0.51</div>
    <div class="metric-lbl">R² Score</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">100</div>
    <div class="metric-lbl">Test Engines</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — WHAT IS RUL?
# ─────────────────────────────────────────────
st.markdown('<div id="rul"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
  <div class="sec-label">01 &nbsp;/&nbsp; Concept</div>
  <h2 class="section-title">WHAT EVEN IS RUL?</h2>
  <p class="section-sub">The clock every machine is running against — you just can't see it.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1], gap="large")
with col1:
    st.markdown("""
    <div class="card">
      <p style="font-size:0.95rem; color:#aaa; line-height:1.8; margin:0;">
        <strong style="color:#f5f5f5; font-weight:500;">Remaining Useful Life (RUL)</strong> is the number of
        operational cycles an engine has <em>left</em> before failure.<br><br>
        Think of it like a battery percentage — except for a
        <strong style="color:#f5f5f5; font-weight:500;">$30M jet engine</strong>.
        Getting it wrong means either
        <span style="color:#fca5a5;">catastrophic failure</span>
        or <span style="color:#fcd34d;">expensive unnecessary maintenance</span>.
      </p>
    </div>
    <div class="card" style="margin-top:0.75rem;">
      <p style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#555; margin:0 0 0.5rem;">FORMULA</p>
      <p style="font-family:'DM Mono',monospace; font-size:1rem; color:#f5f5f5; margin:0;">
        RUL = max_cycle − current_cycle
      </p>
      <p style="font-size:0.8rem; color:#555; margin:0.6rem 0 0;">
        Clipped at 125 cycles → engines in healthy zone treated equally
      </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    cycles = list(range(0, 193))
    rul_raw = [max(0, 192 - c) for c in cycles]
    rul_clipped = [min(125, r) for r in rul_raw]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cycles, y=rul_raw, name="Raw RUL",
                             line=dict(color="#3a3a3a", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=cycles, y=rul_clipped, name="Clipped RUL (≤125)",
                             fill="tozeroy", line=dict(color=ACCENT, width=2.5),
                             fillcolor="rgba(156,163,175,0.06)"))
    fig.add_hline(y=125, line=dict(color="#555", dash="dot", width=1.5),
                  annotation_text="Cap = 125", annotation_font_color="#555")
    fig.add_vline(x=67, line=dict(color="#444", dash="dot", width=1),
                  annotation_text="Learning Zone", annotation_font_color="#555")
    plot_cfg(fig, "RUL Over Engine Lifetime (Engine #1)", 340)
    fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#555")))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 2 — DATASET
# ─────────────────────────────────────────────
st.markdown('<div id="dataset"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-dark">
  <div class="sec-label">02 &nbsp;/&nbsp; Dataset</div>
  <h2 class="section-title">NASA CMAPSS FD001</h2>
  <p class="section-sub">100 training engines. 100 test engines. 26 raw columns. One mission: predict failure.</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="card">
      <div class="metric-val">20,631</div>
      <div class="metric-lbl" style="margin-bottom:1rem;">Training Rows</div>
      <div class="divider"></div>
      <p style="font-size:0.82rem; color:#555; margin:0;">80 engines train · 20 validation · engine-wise split</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <div class="metric-val">26→18</div>
      <div class="metric-lbl" style="margin-bottom:1rem;">Features (raw → kept)</div>
      <div class="divider"></div>
      <p style="font-size:0.82rem; color:#555; margin:0;">7 constant sensors dropped · 1 engineered feature added</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
      <div class="metric-val">30</div>
      <div class="metric-lbl" style="margin-bottom:1rem;">Sequence Window</div>
      <div class="divider"></div>
      <p style="font-size:0.82rem; color:#555; margin:0;">Each LSTM input = last 30 cycles of 18 sensors = (30, 18) tensor</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
  <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#555; margin-bottom:1rem; letter-spacing:0.1em; text-transform:uppercase;">Feature Selection</p>

  <p style="font-size:0.78rem; color:#555; margin-bottom:0.5rem; font-family:'DM Mono',monospace;">DROPPED — zero variance / no correlation</p>
  <div style="margin-bottom:1rem;">
    <span class="ftag dropped">sensor_1</span><span class="ftag dropped">sensor_5</span>
    <span class="ftag dropped">sensor_10</span><span class="ftag dropped">sensor_16</span>
    <span class="ftag dropped">sensor_18</span><span class="ftag dropped">sensor_19</span>
    <span class="ftag dropped">op_setting_3</span>
  </div>

  <p style="font-size:0.78rem; color:#555; margin-bottom:0.5rem; font-family:'DM Mono',monospace;">KEPT — meaningful signal</p>
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

  <p style="font-size:0.78rem; color:#555; margin-bottom:0.5rem; font-family:'DM Mono',monospace;">ENGINEERED</p>
  <span class="ftag added">cycle_norm</span>
  <span style="font-size:0.75rem; color:#555; margin-left:0.5rem;">← cycle ÷ mean_max_cycle (206.31) → 0→1 progress signal</span>
</div>
""", unsafe_allow_html=True)

# Correlation bar chart
corr_data = {
    "sensor": ["sensor_11","sensor_4","sensor_3","sensor_2","sensor_17","sensor_15","sensor_8",
               "sensor_13","sensor_9","sensor_14","sensor_12","sensor_7","sensor_21","sensor_20"],
    "corr":   [-0.775,-0.757,-0.655,-0.678,-0.681,-0.721,-0.625,-0.624,-0.462,-0.370, 0.749, 0.733, 0.707, 0.705]
}
df_corr = pd.DataFrame(corr_data).sort_values("corr")
bar_colors = [WHITE if c > 0 else ACCENT for c in df_corr["corr"]]

fig2 = go.Figure(go.Bar(
    x=df_corr["corr"], y=df_corr["sensor"],
    orientation='h',
    marker_color=bar_colors,
    marker_line_width=0,
))
plot_cfg(fig2, "Sensor Correlation with RUL", 420)
fig2.update_layout(xaxis_title="Pearson Correlation", yaxis_title="")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 3 — PIPELINE
# ─────────────────────────────────────────────
st.markdown('<div id="pipeline"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
  <div class="sec-label">03 &nbsp;/&nbsp; Pipeline</div>
  <h2 class="section-title">HOW THE MAGIC WORKS</h2>
  <p class="section-sub">Raw CSVs → cleaned sequences → trained model → RUL prediction. Every step, explained.</p>
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
# SECTION 4 — MODEL COMPARISON
# ─────────────────────────────────────────────
st.markdown('<div id="models"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-dark">
  <div class="sec-label">04 &nbsp;/&nbsp; Models</div>
  <h2 class="section-title">7 MODELS. ONE WINNER.</h2>
  <p class="section-sub">From a simple linear baseline to stacked bidirectional LSTMs — here's how they all stacked up.</p>
""", unsafe_allow_html=True)

models_data = {
    "Model": ["Linear Regression", "Basic LSTM", "LSTM + Dropout", "Stacked LSTM ⭐", "BiLSTM", "Weighted Stacked", "Weighted BiLSTM"],
    "RMSE":  [26.45, 34.10, 30.56, 25.57, 27.45, 27.97, 28.20],
}
df_models = pd.DataFrame(models_data)
bar_colors = [WHITE if "⭐" in m else ACCENT for m in df_models["Model"]]

fig3 = go.Figure(go.Bar(
    x=df_models["Model"],
    y=df_models["RMSE"],
    marker_color=bar_colors,
    marker_line_width=0,
    text=[f"{v:.2f}" for v in df_models["RMSE"]],
    textposition="outside",
    textfont=dict(color="#6b7280", size=12, family="DM Mono"),
))
fig3.add_hline(y=25.57, line=dict(color="#555", dash="dot", width=1.5),
               annotation_text="Best: 25.57", annotation_font_color="#555")
plot_cfg(fig3, "", 400)
fig3.update_layout(yaxis_title="Test RMSE (lower = better)", yaxis_range=[0, 40],
                   showlegend=False, bargap=0.35)
st.plotly_chart(fig3, use_container_width=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown("""
    <div class="card">
      <span class="badge badge-white">⭐ Winner</span>
      <h3 style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; margin:0.8rem 0 0.3rem; color:#f5f5f5; letter-spacing:0.04em;">Stacked LSTM</h3>
      <p style="font-size:0.8rem; color:#555; margin:0 0 1rem; font-family:'DM Mono',monospace;">LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-val" style="font-size:1.6rem;">25.57</div><div class="metric-lbl">Test RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">6.99</div><div class="metric-lbl">Train RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">9.97</div><div class="metric-lbl">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
      <span class="badge badge-grey">Runner-up</span>
      <h3 style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; margin:0.8rem 0 0.3rem; color:#f5f5f5; letter-spacing:0.04em;">BiLSTM</h3>
      <p style="font-size:0.8rem; color:#555; margin:0 0 1rem; font-family:'DM Mono',monospace;">BiLSTM(64) → LSTM(32) → Dense(16) → Dense(1)</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-val" style="font-size:1.6rem;">27.45</div><div class="metric-lbl">Test RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">4.73</div><div class="metric-lbl">Train RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">10.23</div><div class="metric-lbl">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
      <span class="badge badge-grey">Baseline</span>
      <h3 style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; margin:0.8rem 0 0.3rem; color:#f5f5f5; letter-spacing:0.04em;">Linear Regression</h3>
      <p style="font-size:0.8rem; color:#555; margin:0 0 1rem; font-family:'DM Mono',monospace;">Flatten(30,18)→540 → Linear → RUL</p>
      <div style="display:flex; justify-content:space-between;">
        <div><div class="metric-val" style="font-size:1.6rem;">26.45</div><div class="metric-lbl">Test RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">10.09</div><div class="metric-lbl">Train RMSE</div></div>
        <div><div class="metric-val" style="font-size:1.6rem;">9.22</div><div class="metric-lbl">Val RMSE</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 5 — RESULTS
# ─────────────────────────────────────────────
st.markdown('<div id="results"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
  <div class="sec-label">05 &nbsp;/&nbsp; Results</div>
  <h2 class="section-title">WHAT THE MODEL LEARNED</h2>
  <p class="section-sub">Final model: Stacked LSTM · RMSE 25.57 · MAE 20.39 · R² 0.51 on 100 test engines.</p>
""", unsafe_allow_html=True)

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

tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Scatter", "Error Distribution"])

with tab1:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=actual, name="Actual RUL",
                              line=dict(color=WHITE, width=2)))
    fig4.add_trace(go.Scatter(y=predicted, name="Predicted RUL",
                              line=dict(color=ACCENT, width=2, dash="dash")))
    plot_cfg(fig4, "", 400)
    fig4.update_layout(xaxis_title="Test Engine Index", yaxis_title="RUL (cycles)",
                       legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    st.plotly_chart(fig4, use_container_width=True)

with tab2:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                              marker=dict(color=ACCENT, size=8, opacity=0.7,
                                          line=dict(color="#2a2a2a", width=0.5)),
                              name="Engines"))
    fig5.add_trace(go.Scatter(x=[0,125], y=[0,125], mode="lines",
                              line=dict(color="#444", dash="dot", width=1.5),
                              name="Perfect Prediction"))
    plot_cfg(fig5, "", 420)
    fig5.update_layout(xaxis_title="Actual RUL", yaxis_title="Predicted RUL",
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5, use_container_width=True)

with tab3:
    residuals = [a - p for a, p in zip(actual, predicted)]
    fig6 = go.Figure(go.Histogram(x=residuals, nbinsx=18,
                                   marker_color=ACCENT,
                                   marker_line_color="#161616", marker_line_width=1))
    fig6.add_vline(x=0, line=dict(color=WHITE, width=1.5))
    plot_cfg(fig6, "", 380)
    fig6.update_layout(xaxis_title="Residual Error (Actual − Predicted)", yaxis_title="Count")
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 6 — LIVE PREDICTOR
# ─────────────────────────────────────────────
st.markdown('<div id="predictor"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-dark">
  <div class="sec-label">06 &nbsp;/&nbsp; Live Predictor</div>
  <h2 class="section-title">TRY IT LIVE.</h2>
  <p class="section-sub">Load your trained model and scaler to get a real-time RUL prediction from sensor values.</p>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("""
    <div class="card">
      <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#555; letter-spacing:0.1em; margin-bottom:1rem; text-transform:uppercase;">Model Setup</p>
    """, unsafe_allow_html=True)

    model_path = st.text_input(
        "Path to .keras model",
        value="../models/final_stacked_lstm.keras",
    )
    scaler_path = st.text_input(
        "Path to scaler .pkl (optional)",
        value="../data/processed/scaler.pkl",
    )

    load_btn = st.button("Load Model")
    st.markdown("</div>", unsafe_allow_html=True)

    if load_btn:
        try:
            import tensorflow as tf
            model_live = tf.keras.models.load_model(model_path)
            st.session_state["model_live"] = model_live
            st.markdown('<div class="status-box status-ok">✓ Model loaded successfully!</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="status-box status-crit">✗ Could not load model: {e}</div>', unsafe_allow_html=True)

        if os.path.exists(scaler_path):
            try:
                sc = joblib.load(scaler_path)
                st.session_state["scaler_live"] = sc
                st.markdown('<div class="status-box status-ok">✓ Scaler loaded!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-box status-warn">⚠ Scaler not loaded: {e}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
      <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#555; letter-spacing:0.1em; margin-bottom:1.2rem; text-transform:uppercase;">Simulate Engine Sensors</p>
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
    predict_btn = st.button("Predict RUL")

    if predict_btn:
        feature_order = list(sensor_defaults.keys())
        raw_vals = np.array([[slider_vals[f] for f in feature_order]], dtype=np.float32)

        if "scaler_live" in st.session_state:
            scaled_vals = st.session_state["scaler_live"].transform(raw_vals)
        else:
            scaled_vals = raw_vals

        seq = np.tile(scaled_vals, (30, 1))[np.newaxis, :, :]

        if "model_live" in st.session_state:
            pred_norm = st.session_state["model_live"].predict(seq, verbose=0).flatten()[0]
            pred_rul = float(pred_norm * 125.0)
        else:
            cn = slider_vals["cycle_norm"]
            pred_rul = max(0, round((1 - cn) * 125))
            st.markdown('<div class="status-box status-warn">⚠ Model not loaded — estimate from cycle_norm.</div>', unsafe_allow_html=True)

        pred_rul = max(0, min(125, pred_rul))

        if pred_rul > 80:
            status_class, status_icon, status_text = "status-ok", "●", "HEALTHY — Engine is fine"
        elif pred_rul > 40:
            status_class, status_icon, status_text = "status-warn", "●", "WARNING — Schedule maintenance soon"
        else:
            status_class, status_icon, status_text = "status-crit", "●", "CRITICAL — Immediate attention needed"

        st.markdown(f"""
        <div class="pred-box">
          <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#555; letter-spacing:0.15em; margin-bottom:0.8rem; text-transform:uppercase;">Predicted Remaining Useful Life</div>
          <div class="pred-number">{pred_rul:.0f}</div>
          <div class="pred-unit">Cycles Remaining</div>
          <div class="divider" style="margin:1.5rem 0;"></div>
          <div class="status-box {status_class}" style="text-align:left;">
            {status_icon} &nbsp; {status_text}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        bar_color = OK if pred_rul > 80 else (WARN if pred_rul > 40 else DANGER)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_rul,
            number={"font": {"color": WHITE, "size": 36, "family": "Bebas Neue"}, "suffix": " cycles"},
            gauge={
                "axis": {"range": [0, 125], "tickcolor": "#555", "tickfont": {"color": "#555", "size": 10}},
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": "#1a1a1a",
                "bordercolor": "#2a2a2a",
                "steps": [
                    {"range": [0, 40],   "color": "rgba(239,68,68,0.07)"},
                    {"range": [40, 80],  "color": "rgba(245,158,11,0.05)"},
                    {"range": [80, 125], "color": "rgba(110,231,183,0.05)"},
                ],
                "threshold": {"line": {"color": WHITE, "width": 2}, "thickness": 0.75, "value": pred_rul},
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af"),
                                 height=300, margin=dict(l=30, r=30, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:5rem 2rem; opacity:0.25;">
          <div style="font-size:3rem; font-family:'Bebas Neue',sans-serif; color:#f5f5f5; letter-spacing:0.1em;">✈</div>
          <p style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#555; letter-spacing:0.12em; margin-top:1rem; text-transform:uppercase;">
            Adjust Sensors → Click Predict
          </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 7 — TAKEAWAYS
# ─────────────────────────────────────────────
st.markdown('<div id="takeaways"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
  <div class="sec-label">07 &nbsp;/&nbsp; Takeaways</div>
  <h2 class="section-title">WHAT I LEARNED.</h2>
  <p class="section-sub">Key insights from building this end-to-end predictive maintenance system.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    takeaways = [
        ("Feature selection > more data", "Removing 7 useless sensors improved every model — noise hurts more than you think."),
        ("Engine-wise splits matter", "Random row splits cause data leakage. Always split by engine ID for time-series data."),
        ("Clipping RUL is smart", "Cap at 125 → model focuses on the degradation zone, not flat 'healthy' predictions."),
        ("cycle_norm beats raw cycle", "Normalizing cycle per mean max-cycle gives the model a portable 0→1 progress signal."),
    ]
    for title, detail in takeaways:
        st.markdown(f"""
        <div class="card-sm">
          <strong style="color:#f5f5f5; font-size:0.9rem; font-weight:500;">{title}</strong>
          <p style="font-size:0.82rem; color:#555; margin:0.3rem 0 0; line-height:1.6;">{detail}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    insights = [
        ("Stacked > Bidirectional", "LSTM(128)→LSTM(64) (RMSE 25.57) beat BiLSTM (27.45) — depth over width for this task."),
        ("Simple baseline is competitive", "Linear Regression got RMSE 26.45 — barely worse than deep models. Always baseline."),
        ("Low-RUL zone is hardest", "Models struggle when RUL < 40. Higher stakes = harder to predict. Future: class weighting."),
        ("EarlyStopping saved runs", "Without patience=15, models overfit hard. Validation monitoring is non-negotiable."),
    ]
    for title, detail in insights:
        st.markdown(f"""
        <div class="card-sm">
          <strong style="color:#f5f5f5; font-size:0.9rem; font-weight:500;">{title}</strong>
          <p style="font-size:0.82rem; color:#555; margin:0.3rem 0 0; line-height:1.6;">{detail}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
  <div class="metric-cell">
    <div class="metric-val accent">25.57</div>
    <div class="metric-lbl">Best RMSE</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">20.39</div>
    <div class="metric-lbl">Best MAE</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">0.51</div>
    <div class="metric-lbl">R² Score</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">7</div>
    <div class="metric-lbl">Models Tried</div>
  </div>
  <div class="metric-cell">
    <div class="metric-val">100%</div>
    <div class="metric-lbl">From Scratch</div>
  </div>
</div>

<div style="text-align:center; padding:1.5rem 0 0.5rem;">
  <p style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#3a3a3a; letter-spacing:0.15em; text-transform:uppercase;">
    Python &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; NASA CMAPSS Dataset
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

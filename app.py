import streamlit as st
import sys, os
sys.path.append(os.path.dirname(__file__))


try:
    from utils.style import apply_global_style
except Exception:
    from style import apply_global_style  # fallback if run standalone

st.set_page_config(
    page_title="Morocco FX & IT Readiness",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_style()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🇲🇦 Exchange Rate Regime Evolution in Morocco</div>
    <div class="hero-sub">Assessing Bank Al-Maghrib's readiness for the next step — Inflation Targeting</div>
    <div style="margin-top:16px;">
        <span class="pill pill-blue">Econometrics</span>
        <span class="pill pill-green">Monetary Policy</span>
        <span class="pill pill-amber">Time-Series Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PROJECT SUMMARY ─────────────────────────────────────────────────────────────
col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
    ### About this project
    This application investigates whether Morocco is ready to transition from its
    **managed exchange rate regime** toward a full **inflation-targeting (IT)** framework.

    Following Bank Al-Maghrib's gradual liberalization — the band widening to **±2.5%** in 2018
    and **±5%** in 2020 — a central empirical question arises: *do exchange rate movements transmit
    strongly to domestic prices?* If pass-through is low, greater exchange rate flexibility under IT
    would not destabilize inflation.

    The analysis combines **volatility modeling (GARCH)**, **risk simulation (Monte Carlo)**, and
    **pass-through estimation (VAR)** to produce a quantitative **IT-readiness assessment**.
    """)
with col_b:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">Headline Findings</div>
        <div style="margin-top:12px;">
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span style="color:#94A3B8;">Long-run ERPT (NEER)</span><b>≈ 0.03</b>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span style="color:#94A3B8;">FX share of CPI variance</span><b>≈ 0.4%</b>
            </div>
            <div style="display:flex;justify-content:space-between;padding:6px 0;">
                <span style="color:#94A3B8;">Sample period</span><b>2007–2026</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PAGE NAVIGATION CARDS ───────────────────────────────────────────────────────
st.markdown("### Explore the analysis")

pages = [
    ("📊", "Data Hub", "Load and validate the Morocco dataset and inspect coverage."),
    ("📈", "Exchange Rate Diagnostics", "Historical dynamics, structural breaks, NEER/REER, events."),
    ("🌊", "Volatility Modeling", "GARCH estimation, conditional volatility, regime comparison."),
    ("🎲", "Risk Simulation", "Monte Carlo fan charts, VaR/CVaR, stress testing."),
    ("📡", "ERPT Analysis", "VAR pass-through: IRF, FEVD, rolling ERPT. The core analysis."),
    ("🎯", "IT Readiness", "Four-pillar composite index benchmarked against peers."),
    ("📋", "Executive Summary", "Auto-generated findings, recommendations, and export."),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card" style="margin-bottom:16px;">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── METHODOLOGY STRIP ───────────────────────────────────────────────────────────
st.markdown("### Methodology at a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Models", "3 core", "GARCH · MC · VAR")
m2.metric("Variables", "up to 6", "FX, CPI, rate, oil…")
m3.metric("Observations", "231", "monthly, 2007–2026")
m4.metric("Comparators", "Chile · Peru · Turkey", "at IT adoption")

st.markdown("""
<div style="margin-top:20px;color:#64748B;font-size:0.85rem;text-align:center;">
    Final-year project (PFE) · ENSA Agadir · Built with Streamlit, statsmodels, arch & plotly
</div>
""", unsafe_allow_html=True)
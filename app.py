import streamlit as st

st.set_page_config(
    page_title="Morocco FX & IT Readiness",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇲🇦 Morocco Exchange Rate Dynamics")
st.subheader("A Decision Support Application for Analyzing Morocco's Exchange Rate and BAM's Readiness for Inflation Targeting")

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("👈 Use the sidebar to navigate between pages")
with col2:
    st.success("📂 Start with **Page 0 — Data Hub** to load your data")
with col3:
    st.warning("📊 All analyses run on your local CSV files")

st.markdown("---")
st.markdown("""
### About this app
This application supports the analysis of Morocco's exchange rate regime evolution and evaluates 
Bank Al-Maghrib's readiness for inflation targeting.

**Pages:**
- **0 — Data Hub**: Load and validate your dataset
- **1 — Exchange Rate Diagnostics**: Historical dynamics, structural breaks, rolling statistics
- **2 — Volatility Modeling**: GARCH framework, persistence analysis
- **3 — Risk & Simulation**: Monte Carlo engine, VaR, CVaR, stress testing
- **4 — ERPT Analysis**: VAR/BVAR model, IRF, variance decomposition
- **5 — IT Readiness Dashboard**: Composite index, pillar scoring
- **6 — Executive Summary**: Key findings and export
""")

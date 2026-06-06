import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_country, validate_dataframe, COLUMN_LABELS
try:
    from utils.style import apply_global_style
except Exception:
    def apply_global_style(): pass

st.set_page_config(page_title="Data Hub", page_icon="📊", layout="wide")
apply_global_style()

st.title("📊 Page 0 — Data Hub")
st.caption("Load, inspect, and validate the Morocco dataset that powers the entire analysis.")

# ════════════════════════════════════════════════════════════════════════════
# Load Morocco data
# ════════════════════════════════════════════════════════════════════════════
df = load_country("Morocco")

if df is None:
    st.error("⚠️ Could not find `morocco.csv`. Place it in the `data/` folder and reload.")
    st.markdown("""
    Expected location:
    ```
    pfe_app/data/morocco.csv
    ```
    The file should contain a `date` column plus the economic variables
    (cpi, fx_eur, fx_usd, neer, reer, policy_rate, reserves, oil_brent).
    """)
    st.stop()

report = validate_dataframe(df, "Morocco")

# ════════════════════════════════════════════════════════════════════════════
# Top-line status
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{report['row_count']}")
start, end = report["date_range"]
c2.metric("Start", start.strftime("%b %Y") if start is not None else "—")
c3.metric("End", end.strftime("%b %Y") if end is not None else "—")
c4.metric("Variables", f"{len(report['present_cols'])}")

if report["issues"]:
    for issue in report["issues"]:
        st.warning(f"⚠️ {issue}")
else:
    st.success("✅ Dataset is clean — continuous monthly coverage, no missing values, all core variables present.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# Variable coverage
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Variable Coverage")

# Build coverage rows (output_gap intentionally excluded everywhere)
coverage = []
for col, label in COLUMN_LABELS.items():
    if col == "output_gap":
        continue
    present = col in df.columns
    if present:
        n_missing = int(df[col].isna().sum())
        status = "✅ Complete" if n_missing == 0 else f"⚠️ {n_missing} missing"
        last_val = df[col].dropna().iloc[-1] if df[col].notna().any() else None
        last_str = f"{last_val:,.2f}" if last_val is not None else "—"
    else:
        status = "➖ Not in dataset"
        last_str = "—"
    coverage.append({
        "Variable": label,
        "Column": col,
        "Status": status,
        "Latest value": last_str,
    })

st.dataframe(pd.DataFrame(coverage), use_container_width=True, hide_index=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# Data preview
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Data Preview")

show_cols = [c for c in df.columns if c != "output_gap"]
preview = df[show_cols].copy()
preview.index = preview.index.strftime("%Y-%m")

n_rows = st.radio("Rows to show", [10, 25, 50, "All"], horizontal=True, index=0)
if n_rows == "All":
    st.dataframe(preview, use_container_width=True)
else:
    head_tail = st.toggle("Show first & last rows", value=False,
                          help="Show the first and last rows together instead of just the first.")
    if head_tail:
        half = n_rows // 2
        st.dataframe(pd.concat([preview.head(half), preview.tail(half)]),
                     use_container_width=True)
    else:
        st.dataframe(preview.head(n_rows), use_container_width=True)

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# Quick visual inspection
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Quick Visual Inspection")

plottable = [c for c in df.columns if c != "output_gap" and df[c].notna().any()]
default_sel = [c for c in ["fx_eur", "cpi", "neer"] if c in plottable]
sel = st.multiselect("Variables to plot (normalized to 100 at start for comparison)",
                     plottable,
                     default=default_sel,
                     format_func=lambda c: COLUMN_LABELS.get(c, c))

if sel:
    fig = go.Figure()
    colors = ["#2563EB", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2", "#be185d"]
    for i, col in enumerate(sel):
        series = df[col].dropna()
        if len(series) == 0:
            continue
        norm = series / series.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            mode="lines", name=COLUMN_LABELS.get(col, col),
            line=dict(color=colors[i % len(colors)], width=2)))
    # Liberalization markers
    for d, lbl in [("2018-01-15", "Band ±2.5%"), ("2020-03-20", "Band ±5%")]:
        ev = pd.Timestamp(d)
        if df.index.min() <= ev <= df.index.max():
            fig.add_shape(type="line", xref="x", yref="paper",
                          x0=ev, x1=ev, y0=0, y1=1,
                          line=dict(color="orange", width=1.2, dash="dash"))
            fig.add_annotation(xref="x", yref="paper", x=ev, y=1.02,
                               text=lbl, showarrow=False,
                               font=dict(color="orange", size=10), xanchor="left")
    fig.update_layout(
        height=420, title="Normalized series (start = 100)",
        xaxis_title="Date", yaxis_title="Index (start = 100)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        legend=dict(orientation="h", y=1.12), margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select one or more variables above to plot them.")

st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Summary Statistics")

stat_cols = [c for c in df.columns if c != "output_gap"]
stats = df[stat_cols].describe().T
stats = stats.rename(columns={
    "count": "N", "mean": "Mean", "std": "Std",
    "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"})
stats.index = [COLUMN_LABELS.get(c, c) for c in stats.index]
st.dataframe(stats.round(2), use_container_width=True)

st.markdown("""
<div style="margin-top:24px;color:#64748B;font-size:0.82rem;text-align:center;">
    Morocco monthly dataset · used across all analysis pages · Bank Al-Maghrib, HCP, IMF & FRED sources
</div>
""", unsafe_allow_html=True)
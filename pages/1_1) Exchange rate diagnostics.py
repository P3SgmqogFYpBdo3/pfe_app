import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_country, load_all_countries, sidebar_country_selector, get_available_countries

from utils.style import apply_global_style
apply_global_style()

st.set_page_config(page_title="Exchange Rate Diagnostics", page_icon="📈", layout="wide")
st.title("📈 Page 1 — Exchange Rate Diagnostics")
st.caption("Historical dynamics, structural breaks, rolling statistics, and event analysis.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
country = sidebar_country_selector("Morocco")
df = load_country(country)

if df is None:
    st.error(f"No data found for {country}. Please check your /data folder.")
    st.stop()

# FX pair selector
fx_col = st.sidebar.selectbox("Exchange rate", ["fx_eur", "fx_usd"],
    format_func=lambda x: "MAD/EUR" if x == "fx_eur" else "MAD/USD")
fx_label = "MAD/EUR" if fx_col == "fx_eur" else "MAD/USD"

# Date range
min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()
date_range = st.sidebar.slider("Date range",
    min_value=min_date, max_value=max_date,
    value=(min_date, max_date), format="YYYY-MM")

df_filtered = df.loc[date_range[0]:date_range[1]].copy()

# Morocco-specific event markers
EVENTS = {
    "Morocco": [
        {"date": "2008-06-01", "label": "Global food/fuel shock", "color": "orange"},
        {"date": "2011-02-01", "label": "Arab Spring", "color": "red"},
        {"date": "2018-01-15", "label": "Band widening ±2.5%", "color": "blue"},
        {"date": "2020-03-20", "label": "Band widening ±5%", "color": "blue"},
        {"date": "2020-04-01", "label": "COVID-19 shock", "color": "red"},
        {"date": "2022-02-01", "label": "Russia-Ukraine / inflation surge", "color": "orange"},
    ]
}
events = EVENTS.get(country, [])

# ── Helper: log returns ───────────────────────────────────────────────────────
def log_returns(series):
    return np.log(series / series.shift(1)).dropna()


# ════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historical Dynamics",
    "💥 Structural Breaks",
    "🔄 Rolling Statistics",
    "📐 NEER / REER",
    "📅 Event Analysis"
])


# ── TAB 1: Historical Dynamics ───────────────────────────────────────────────
with tab1:
    st.subheader(f"{fx_label} — Levels & Returns ({country})")

    returns = log_returns(df_filtered[fx_col]) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"{fx_label} Level", "Log Returns (%)"),
        row_heights=[0.65, 0.35], vertical_spacing=0.08)

    # Level
    fig.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered[fx_col],
        mode="lines", name=fx_label,
        line=dict(color="#2563EB", width=1.5)), row=1, col=1)

    # Event lines on level chart
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if date_range[0] <= ev_date <= date_range[1]:
            fig.add_vline(x=ev_date, line_dash="dash",
                line_color=ev["color"], line_width=1, row=1, col=1)
            fig.add_annotation(x=ev_date, y=df_filtered[fx_col].max(),
                text=ev["label"], showarrow=False,
                textangle=-90, font=dict(size=9, color=ev["color"]),
                yshift=0, xshift=5, row=1, col=1)

    # Returns — colored by positive/negative
    colors = ["#16a34a" if r < 0 else "#dc2626" for r in returns]
    fig.add_trace(go.Bar(
        x=returns.index, y=returns.values,
        marker_color=colors, name="Log return", opacity=0.7), row=2, col=1)

    fig.update_layout(height=500, showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    st.markdown("**Summary statistics**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean return", f"{returns.mean():.4f}%")
    c2.metric("Std deviation", f"{returns.std():.4f}%")
    c3.metric("Min return", f"{returns.min():.3f}%")
    c4.metric("Max return", f"{returns.max():.3f}%")
    c5.metric("Observations", len(returns))


# ── TAB 2: Structural Breaks ─────────────────────────────────────────────────
with tab2:
    st.subheader("Structural Break Analysis — Pre vs Post Liberalization")
    st.caption("Comparing exchange rate behavior before and after key regime changes.")

    if country == "Morocco":
        break_dates = {
            "2018-01-15": "Band widening ±2.5% (Jan 2018)",
            "2020-03-20": "Band widening ±5% (Mar 2020)"
        }
    else:
        break_dates = {}

    # Allow custom break date
    st.markdown("**Set break date for analysis**")
    col_a, col_b = st.columns(2)
    with col_a:
        preset = st.selectbox("Preset break points",
            ["Custom"] + list(break_dates.values()) if break_dates else ["Custom"])
    with col_b:
        if preset != "Custom" and break_dates:
            break_date_str = [k for k, v in break_dates.items() if v == preset][0]
            break_date = pd.Timestamp(break_date_str)
        else:
            break_date = pd.Timestamp(
                st.date_input("Custom break date", value=pd.Timestamp("2018-01-01"),
                    min_value=min_date, max_value=max_date))

    returns_full = log_returns(df[fx_col]) * 100
    pre = returns_full[returns_full.index < break_date]
    post = returns_full[returns_full.index >= break_date]

    if len(pre) < 10 or len(post) < 10:
        st.warning("Not enough data on one side of the break date.")
    else:
        # Stats comparison
        stats = pd.DataFrame({
            "Metric": ["Observations", "Mean return (%)", "Std deviation (%)",
                       "Min (%)", "Max (%)", "Annualized vol (%)"],
            f"Pre ({pre.index.min().strftime('%Y-%m')} – {pre.index.max().strftime('%Y-%m')})": [
                len(pre), round(pre.mean(), 4), round(pre.std(), 4),
                round(pre.min(), 3), round(pre.max(), 3),
                round(pre.std() * np.sqrt(12), 3)
            ],
            f"Post ({post.index.min().strftime('%Y-%m')} – {post.index.max().strftime('%Y-%m')})": [
                len(post), round(post.mean(), 4), round(post.std(), 4),
                round(post.min(), 3), round(post.max(), 3),
                round(post.std() * np.sqrt(12), 3)
            ]
        })
        st.dataframe(stats, use_container_width=True, hide_index=True)

        # Visual comparison
        fig2 = make_subplots(rows=1, cols=2,
            subplot_titles=(
                f"Pre-break distribution",
                f"Post-break distribution"))

        for i, (series, label, color) in enumerate([
            (pre, "Pre", "#2563EB"), (post, "Post", "#dc2626")
        ], 1):
            fig2.add_trace(go.Histogram(
                x=series, name=label, marker_color=color,
                opacity=0.7, nbinsx=30,
                histnorm="probability density"), row=1, col=i)
            fig2.add_vline(x=series.mean(), line_dash="dash",
                line_color=color, row=1, col=i)

        fig2.update_layout(height=350, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        # Chow-test approximation (F-test on variance ratio)
        from scipy import stats as scipy_stats
        f_stat = (post.std() ** 2) / (pre.std() ** 2)
        p_value = 2 * min(
            scipy_stats.f.cdf(f_stat, len(post)-1, len(pre)-1),
            1 - scipy_stats.f.cdf(f_stat, len(post)-1, len(pre)-1)
        )
        st.markdown("**Variance ratio test (structural break significance)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("F-statistic", f"{f_stat:.4f}")
        c2.metric("p-value", f"{p_value:.4f}")
        c3.metric("Conclusion",
            "Significant break ✓" if p_value < 0.05 else "Not significant")


# ── TAB 3: Rolling Statistics ────────────────────────────────────────────────
with tab3:
    st.subheader("Rolling Volatility & Mean")

    window = st.slider("Rolling window (months)", 3, 24, 12)

    returns_roll = log_returns(df_filtered[fx_col]) * 100
    roll_vol = returns_roll.rolling(window).std() * np.sqrt(12)  # annualized
    roll_mean = df_filtered[fx_col].rolling(window).mean()

    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            f"Rolling {window}-month Annualized Volatility (%)",
            f"Rolling {window}-month Mean — {fx_label}"),
        vertical_spacing=0.1)

    fig3.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        fill="tozeroy", fillcolor="rgba(220,38,38,0.1)",
        line=dict(color="#dc2626", width=1.5), name="Rolling vol"), row=1, col=1)

    fig3.add_trace(go.Scatter(
        x=roll_mean.index, y=roll_mean.values,
        line=dict(color="#2563EB", width=1.5), name="Rolling mean"), row=2, col=1)

    # Add actual FX level faded
    fig3.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered[fx_col],
        line=dict(color="#93c5fd", width=1, dash="dot"),
        name=fx_label, opacity=0.5), row=2, col=1)

    # Event lines
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if date_range[0] <= ev_date <= date_range[1]:
            for row in [1, 2]:
                fig3.add_vline(x=ev_date, line_dash="dash",
                    line_color=ev["color"], line_width=1, row=row, col=1)

    fig3.update_layout(height=500, showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig3.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig3, use_container_width=True)

    # Instability detection
    vol_mean = roll_vol.mean()
    vol_std = roll_vol.std()
    high_vol_periods = roll_vol[roll_vol > vol_mean + 1.5 * vol_std]
    if len(high_vol_periods) > 0:
        st.warning(f"⚠️ {len(high_vol_periods)} months with abnormally high volatility detected "
                   f"(>{vol_mean + 1.5*vol_std:.2f}% annualized)")
        st.dataframe(
            pd.DataFrame({
                "Date": high_vol_periods.index.strftime("%Y-%m"),
                "Annualized Vol (%)": high_vol_periods.round(3).values
            }), use_container_width=True, hide_index=True)


# ── TAB 4: NEER / REER ───────────────────────────────────────────────────────
with tab4:
    st.subheader("Nominal & Real Effective Exchange Rate")
    st.caption("NEER measures competitiveness vs trading partners. REER adjusts for inflation differentials.")

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["neer"],
        mode="lines", name="NEER",
        line=dict(color="#2563EB", width=2)))

    fig4.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["reer"],
        mode="lines", name="REER",
        line=dict(color="#dc2626", width=2)))

    # Reference line at 100
    fig4.add_hline(y=100, line_dash="dot", line_color="gray",
        annotation_text="Base = 100", annotation_position="right")

    # Event markers
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        if date_range[0] <= ev_date <= date_range[1]:
            fig4.add_vline(x=ev_date, line_dash="dash",
                line_color=ev["color"], line_width=1)

    fig4.update_layout(
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig4, use_container_width=True)

    # NEER vs REER gap
    gap_series = df_filtered["neer"] - df_filtered["reer"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NEER (latest)", f"{df_filtered['neer'].iloc[-1]:.1f}")
    c2.metric("REER (latest)", f"{df_filtered['reer'].iloc[-1]:.1f}")
    c3.metric("NEER-REER gap", f"{gap_series.iloc[-1]:.2f}")
    c4.metric("REER trend",
        "Appreciating 📈" if df_filtered["reer"].iloc[-1] > df_filtered["reer"].iloc[-12] else "Depreciating 📉")

    st.markdown("**Interpretation**")
    latest_reer = df_filtered["reer"].iloc[-1]
    if latest_reer < 95:
        st.info("🟢 REER below 100 — currency is **undervalued** relative to base period, supporting export competitiveness.")
    elif latest_reer > 105:
        st.warning("🔴 REER above 100 — currency is **overvalued**, which may hurt export competitiveness.")
    else:
        st.success("🟡 REER near 100 — currency broadly in line with historical equilibrium.")


# ── TAB 5: Event Analysis ────────────────────────────────────────────────────
with tab5:
    st.subheader("Event-Based Analysis")
    st.caption("Impact of key macroeconomic events on exchange rate dynamics.")

    if not events:
        st.info("No predefined events for this country. You can add custom events below.")
    else:
        returns_ev = log_returns(df[fx_col]) * 100
        vol_ev = returns_ev.rolling(6).std() * np.sqrt(12)

        event_results = []
        for ev in events:
            ev_date = pd.Timestamp(ev["date"])
            pre_window = returns_ev[
                (returns_ev.index >= ev_date - pd.DateOffset(months=6)) &
                (returns_ev.index < ev_date)]
            post_window = returns_ev[
                (returns_ev.index >= ev_date) &
                (returns_ev.index < ev_date + pd.DateOffset(months=6))]

            if len(pre_window) > 2 and len(post_window) > 2:
                event_results.append({
                    "Event": ev["label"],
                    "Date": ev["date"][:7],
                    "Pre-event vol (%)": round(pre_window.std() * np.sqrt(12), 3),
                    "Post-event vol (%)": round(post_window.std() * np.sqrt(12), 3),
                    "Vol change": "↑ Increased" if post_window.std() > pre_window.std() else "↓ Decreased",
                    "FX level at event": round(df.loc[df.index >= ev_date, fx_col].iloc[0], 3)
                        if len(df.loc[df.index >= ev_date]) > 0 else "N/A"
                })

        if event_results:
            st.dataframe(pd.DataFrame(event_results),
                use_container_width=True, hide_index=True)

        # Timeline chart
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=df.index, y=df[fx_col],
            mode="lines", name=fx_label,
            line=dict(color="#2563EB", width=1.5)))

        for ev in events:
            ev_date = pd.Timestamp(ev["date"])
            if ev_date in df.index or (df.index.min() <= ev_date <= df.index.max()):
                y_val = df[fx_col].max() * 0.98
                fig5.add_vline(x=ev_date, line_dash="dash",
                    line_color=ev["color"], line_width=1.5)
                fig5.add_annotation(
                    x=ev_date, y=y_val,
                    text=ev["label"], showarrow=False,
                    textangle=-90, font=dict(size=9, color=ev["color"]),
                    xshift=8)

        fig5.update_layout(
            height=420, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
            xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig5, use_container_width=True)

    # Custom event marker
    st.markdown("---")
    st.markdown("**Add a custom event marker**")
    col1, col2 = st.columns(2)
    with col1:
        custom_date = st.date_input("Event date",
            value=pd.Timestamp("2020-03-01"),
            min_value=min_date, max_value=max_date)
    with col2:
        custom_label = st.text_input("Event label", value="Custom event")

    if st.button("Add to chart"):
        fig_custom = go.Figure()
        fig_custom.add_trace(go.Scatter(
            x=df.index, y=df[fx_col],
            mode="lines", line=dict(color="#2563EB", width=1.5)))
        fig_custom.add_vline(x=pd.Timestamp(custom_date),
            line_dash="dash", line_color="purple", line_width=2,
            annotation_text=custom_label,
            annotation_position="top right")
        fig_custom.update_layout(
            height=350, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_custom, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_country, sidebar_country_selector

from utils.style import apply_global_style
apply_global_style()

st.set_page_config(page_title="Risk & Simulation", page_icon="🎲", layout="wide")
st.title("🎲 Page 3 — Risk & Scenario Simulation")
st.caption("Monte Carlo simulation using GARCH-implied volatility. Models the distribution of outcomes — not directional forecasts.")

st.info("⚠️ **Methodological note:** This page quantifies uncertainty and tail risk. "
        "Simulated paths show the *range of possible outcomes*, not predictions of future exchange rate direction.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
country = sidebar_country_selector("Morocco")
df = load_country(country)

if df is None:
    st.error(f"No data found for {country}.")
    st.stop()

fx_col = st.sidebar.selectbox("Exchange rate", ["fx_eur", "fx_usd"],
    format_func=lambda x: "MAD/EUR" if x == "fx_eur" else "MAD/USD")
fx_label = "MAD/EUR" if fx_col == "fx_eur" else "MAD/USD"

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
n_simulations = st.sidebar.select_slider("Number of simulations",
    options=[500, 1000, 2000, 5000], value=1000)
horizon = st.sidebar.slider("Forecast horizon (months)", 1, 24, 12)
confidence_level = st.sidebar.slider("Confidence level (%)", 90, 99, 95)
np.random.seed(st.sidebar.number_input("Random seed", value=42, step=1))

# ── Compute returns & volatility ─────────────────────────────────────────────
returns = np.log(df[fx_col] / df[fx_col].shift(1)).dropna() * 100
mu = returns.mean()
sigma = returns.std()
current_fx = df[fx_col].iloc[-1]
last_date = df.index[-1]

# Try to get GARCH volatility from session state (fitted in Page 2)
garch_vol = None
if "garch_result" in st.session_state:
    try:
        garch_vol = st.session_state["garch_result"].conditional_volatility.iloc[-1]
        st.sidebar.success(f"✅ Using GARCH σ = {garch_vol:.4f}%")
    except Exception:
        pass

if garch_vol is None:
    sigma_used = sigma
    st.sidebar.info(f"Using historical σ = {sigma:.4f}%\n(Fit GARCH on Page 2 for GARCH-implied vol)")
else:
    sigma_used = garch_vol
    st.sidebar.success(f"Using GARCH σ = {garch_vol:.4f}%")


# ── Core simulation function ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(S0, mu, sigma, horizon, n_sims, seed=42):
    np.random.seed(seed)
    dt = 1
    shocks = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        size=(n_sims, horizon)
    ) / 100  # convert back from %
    log_paths = np.cumsum(shocks, axis=1)
    paths = S0 * np.exp(log_paths)
    return paths


# ── Generate dates for horizon ───────────────────────────────────────────────
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=horizon, freq="MS")


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Fan Chart",
    "📊 VaR & CVaR",
    "🔚 Terminal Distribution",
    "💥 Stress Testing",
    "📋 Risk Summary"
])


# ── TAB 1: Fan Chart ─────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Monte Carlo Fan Chart — {fx_label} ({n_simulations} paths, {horizon} months)")

    with st.spinner("Running simulation..."):
        paths = run_simulation(current_fx, mu, sigma_used, horizon, n_simulations)

    # Percentile bands
    p5   = np.percentile(paths, 5,  axis=0)
    p10  = np.percentile(paths, 10, axis=0)
    p25  = np.percentile(paths, 25, axis=0)
    p50  = np.percentile(paths, 50, axis=0)
    p75  = np.percentile(paths, 75, axis=0)
    p90  = np.percentile(paths, 90, axis=0)
    p95  = np.percentile(paths, 95, axis=0)

    # Historical tail (last 24 months)
    hist_tail = df[fx_col].iloc[-24:]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        mode="lines", line=dict(color="#1e3a5f", width=2),
        name="Historical"))

    # Fan bands
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.07)",
        line=dict(width=0), name="5–95%", showlegend=True))

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(p90) + list(p10[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.12)",
        line=dict(width=0), name="10–90%", showlegend=True))

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(p75) + list(p25[::-1]),
        fill="toself", fillcolor="rgba(37,99,235,0.20)",
        line=dict(width=0), name="25–75%", showlegend=True))

    # Median
    fig.add_trace(go.Scatter(
        x=future_dates, y=p50,
        mode="lines", line=dict(color="#2563EB", width=2.5, dash="dash"),
        name="Median"))

    # A sample of individual paths
    n_show = min(80, n_simulations)
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=future_dates, y=paths[i],
            mode="lines", line=dict(color="rgba(37,99,235,0.04)", width=0.5),
            showlegend=False))

    # Current level reference
    fig.add_hline(y=current_fx, line_dash="dot",
        line_color="gray", line_width=1,
        annotation_text=f"Current: {current_fx:.3f}",
        annotation_position="right")

    fig.update_layout(
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", title=fx_label),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current FX", f"{current_fx:.3f}")
    c2.metric(f"Median ({horizon}m)", f"{p50[-1]:.3f}")
    c3.metric(f"5th pct ({horizon}m)", f"{p5[-1]:.3f}")
    c4.metric(f"95th pct ({horizon}m)", f"{p95[-1]:.3f}")


# ── TAB 2: VaR & CVaR ────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Value at Risk & Conditional VaR — {confidence_level}% Confidence")

    if "paths" not in dir():
        paths = run_simulation(current_fx, mu, sigma_used, horizon, n_simulations)

    # Terminal returns (% change from current)
    terminal_returns = (paths[:, -1] - current_fx) / current_fx * 100
    alpha = (100 - confidence_level) / 100

    var = np.percentile(terminal_returns, alpha * 100)
    cvar = terminal_returns[terminal_returns <= var].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR ({confidence_level}%, {horizon}m)",
        f"{var:.3f}%",
        help=f"With {confidence_level}% confidence, the {horizon}-month loss will not exceed this")
    c2.metric(f"CVaR ({confidence_level}%, {horizon}m)",
        f"{cvar:.3f}%",
        help="Expected loss in the worst scenarios beyond VaR")
    c3.metric("VaR in FX units",
        f"{current_fx * (1 + var/100):.3f}",
        delta=f"{var:.3f}%")
    c4.metric("CVaR in FX units",
        f"{current_fx * (1 + cvar/100):.3f}",
        delta=f"{cvar:.3f}%")

    # Distribution of terminal outcomes
    fig2 = go.Figure()

    fig2.add_trace(go.Histogram(
        x=terminal_returns, nbinsx=60,
        marker_color="#2563EB", opacity=0.7,
        histnorm="probability density",
        name="Terminal returns"))

    # VaR line
    fig2.add_vline(x=var, line_dash="dash", line_color="#dc2626", line_width=2,
        annotation_text=f"VaR {confidence_level}%: {var:.2f}%",
        annotation_position="top right",
        annotation_font_color="#dc2626")

    # CVaR line
    fig2.add_vline(x=cvar, line_dash="dash", line_color="#7c3aed", line_width=2,
        annotation_text=f"CVaR: {cvar:.2f}%",
        annotation_position="top left",
        annotation_font_color="#7c3aed")

    # Shade tail
    x_tail = np.linspace(terminal_returns.min(), var, 100)
    fig2.add_trace(go.Scatter(
        x=list(x_tail) + [var, terminal_returns.min()],
        y=[0] * len(x_tail) + [0, 0],
        fill="toself", fillcolor="rgba(220,38,38,0.15)",
        line=dict(width=0), name=f"Tail ({alpha*100:.0f}%)"))

    fig2.update_layout(
        title=f"Distribution of {horizon}-month Terminal Returns",
        height=400, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Return (%)", gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig2, use_container_width=True)

    # VaR across horizons
    st.markdown("**VaR across different horizons**")
    horizons_list = [1, 3, 6, 12, 18, 24]
    var_table = []
    for h in horizons_list:
        p_h = run_simulation(current_fx, mu, sigma_used, h, n_simulations)
        t_ret = (p_h[:, -1] - current_fx) / current_fx * 100
        v = np.percentile(t_ret, alpha * 100)
        cv = t_ret[t_ret <= v].mean()
        var_table.append({
            "Horizon (months)": h,
            f"VaR {confidence_level}% (%)": round(v, 3),
            f"CVaR {confidence_level}% (%)": round(cv, 3),
            "FX at VaR": round(current_fx * (1 + v/100), 3),
            "FX at CVaR": round(current_fx * (1 + cv/100), 3),
        })
    st.dataframe(pd.DataFrame(var_table), use_container_width=True, hide_index=True)


# ── TAB 3: Terminal Distribution ─────────────────────────────────────────────
with tab3:
    st.subheader(f"Terminal FX Distribution at {horizon}-month Horizon")

    if "paths" not in dir():
        paths = run_simulation(current_fx, mu, sigma_used, horizon, n_simulations)

    terminal_fx = paths[:, -1]

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=terminal_fx, nbinsx=60,
        marker_color="#2563EB", opacity=0.75,
        histnorm="probability density",
        name="Terminal FX"))

    # Percentile markers
    for pct, color, dash in [(5, "#dc2626", "dash"), (25, "#f59e0b", "dot"),
                              (50, "#16a34a", "solid"), (75, "#f59e0b", "dot"),
                              (95, "#dc2626", "dash")]:
        val = np.percentile(terminal_fx, pct)
        fig3.add_vline(x=val, line_dash=dash, line_color=color, line_width=1.5,
            annotation_text=f"P{pct}: {val:.3f}",
            annotation_font_color=color, annotation_position="top right")

    fig3.update_layout(
        height=400, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title=fx_label, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
    st.plotly_chart(fig3, use_container_width=True)

    # Probability of depreciation / appreciation
    prob_depreciation = (terminal_fx > current_fx).mean() * 100
    prob_appreciation = (terminal_fx <= current_fx).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(depreciation)", f"{prob_depreciation:.1f}%")
    c2.metric("P(appreciation)", f"{prob_appreciation:.1f}%")
    c3.metric("Expected FX", f"{terminal_fx.mean():.3f}")
    c4.metric("FX std dev", f"{terminal_fx.std():.3f}")

    # Percentile table
    pct_df = pd.DataFrame({
        "Percentile": ["1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%"],
        f"{fx_label} level": [round(np.percentile(terminal_fx, p), 3)
                              for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]],
        "Change from current (%)": [round((np.percentile(terminal_fx, p) - current_fx)
                                    / current_fx * 100, 2)
                                    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)


# ── TAB 4: Stress Testing ────────────────────────────────────────────────────
with tab4:
    st.subheader("Stress Testing — External Shock Scenarios")
    st.caption("Simulate the impact of defined macroeconomic shocks on the exchange rate distribution.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Predefined shock scenarios**")
        scenario = st.selectbox("Select scenario", [
            "Custom",
            "EUR depreciation -10% (trading partner shock)",
            "Global oil spike +40% (commodity shock)",
            "Capital outflow — vol doubles",
            "2020 COVID-style shock",
            "2008 Global Financial Crisis",
        ])

        # Map scenario to shock parameters
        scenario_params = {
            "EUR depreciation -10% (trading partner shock)":
                dict(mu_shock=-0.15, sigma_mult=1.5),
            "Global oil spike +40% (commodity shock)":
                dict(mu_shock=0.10, sigma_mult=1.3),
            "Capital outflow — vol doubles":
                dict(mu_shock=0.05, sigma_mult=2.0),
            "2020 COVID-style shock":
                dict(mu_shock=0.08, sigma_mult=2.5),
            "2008 Global Financial Crisis":
                dict(mu_shock=0.12, sigma_mult=2.2),
        }

        if scenario == "Custom":
            st.markdown("**Custom shock parameters**")
            mu_shock = st.slider("Drift shock (% per month)", -2.0, 2.0, 0.1, 0.05)
            sigma_mult = st.slider("Volatility multiplier", 1.0, 5.0, 1.5, 0.1)
        else:
            params = scenario_params[scenario]
            mu_shock = params["mu_shock"]
            sigma_mult = params["sigma_mult"]
            st.info(f"Drift shock: **{mu_shock:+.2f}%/month** | Vol multiplier: **{sigma_mult}×**")

    with col2:
        st.markdown("**Shock horizon**")
        shock_horizon = st.slider("Shock duration (months)", 1, 12, 3)
        recovery = st.checkbox("Apply gradual recovery after shock", value=True)

    if st.button("▶ Run Stress Simulation", type="primary"):
        # Stressed simulation
        n_stress = 1000
        np.random.seed(42)

        # Build path: shock period + recovery period
        total_horizon = horizon
        stressed_paths = np.zeros((n_stress, total_horizon))
        stressed_paths[:, 0] = current_fx

        for t in range(1, total_horizon):
            if t <= shock_horizon:
                mu_t = (mu + mu_shock) / 100
                sigma_t = sigma_used * sigma_mult / 100
            else:
                if recovery:
                    fade = (t - shock_horizon) / max(total_horizon - shock_horizon, 1)
                    mu_t = ((mu + mu_shock * (1 - fade)) / 100)
                    sigma_t = (sigma_used * (sigma_mult - (sigma_mult - 1) * fade)) / 100
                else:
                    mu_t = mu / 100
                    sigma_t = sigma_used / 100

            shocks_t = np.random.normal(mu_t - 0.5 * sigma_t**2, sigma_t, n_stress)
            stressed_paths[:, t] = stressed_paths[:, t-1] * np.exp(shocks_t)

        # Baseline paths
        baseline_paths = run_simulation(current_fx, mu, sigma_used, total_horizon, n_stress)

        # Fan chart comparison
        fig4 = go.Figure()

        for label, p_arr, color, dash in [
            ("Baseline", baseline_paths, "#2563EB", "solid"),
            ("Stressed", stressed_paths, "#dc2626", "solid")
        ]:
            p5_s  = np.percentile(p_arr, 5,  axis=0)
            p50_s = np.percentile(p_arr, 50, axis=0)
            p95_s = np.percentile(p_arr, 95, axis=0)

            fig4.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(p95_s) + list(p5_s[::-1]),
                fill="toself",
                fillcolor=f"rgba({'220,38,38' if color == '#dc2626' else '37,99,235'},0.08)",
                line=dict(width=0), showlegend=False))

            fig4.add_trace(go.Scatter(
                x=future_dates, y=p50_s,
                mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                name=f"{label} median"))

        # Shock period shading
        if shock_horizon < len(future_dates):
            fig4.add_vrect(
                x0=future_dates[0], x1=future_dates[min(shock_horizon-1, len(future_dates)-1)],
                fillcolor="rgba(220,38,38,0.08)", line_width=0,
                annotation_text="Shock period", annotation_position="top left")

        fig4.add_hline(y=current_fx, line_dash="dot", line_color="gray",
            annotation_text=f"Current: {current_fx:.3f}", annotation_position="right")

        fig4.update_layout(
            title=f"Baseline vs Stressed Fan Chart — {scenario if scenario != 'Custom' else 'Custom Shock'}",
            height=440,
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=0, r=0, t=50, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)", title=fx_label),
            xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig4, use_container_width=True)

        # Comparison metrics
        alpha_s = (100 - confidence_level) / 100
        base_terminal = (baseline_paths[:, -1] - current_fx) / current_fx * 100
        stress_terminal = (stressed_paths[:, -1] - current_fx) / current_fx * 100

        base_var = np.percentile(base_terminal, alpha_s * 100)
        stress_var = np.percentile(stress_terminal, alpha_s * 100)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline VaR", f"{base_var:.3f}%")
        c2.metric("Stressed VaR", f"{stress_var:.3f}%",
            delta=f"{stress_var - base_var:.3f}%")
        c3.metric("Baseline median FX", f"{np.median(baseline_paths[:,-1]):.3f}")
        c4.metric("Stressed median FX", f"{np.median(stressed_paths[:,-1]):.3f}",
            delta=f"{np.median(stressed_paths[:,-1]) - np.median(baseline_paths[:,-1]):.3f}")


# ── TAB 5: Risk Summary ──────────────────────────────────────────────────────
with tab5:
    st.subheader("Risk Summary Dashboard")

    if "paths" not in dir():
        paths = run_simulation(current_fx, mu, sigma_used, horizon, n_simulations)

    terminal_fx_s = paths[:, -1]
    terminal_ret_s = (terminal_fx_s - current_fx) / current_fx * 100
    alpha_s = (100 - confidence_level) / 100
    var_s = np.percentile(terminal_ret_s, alpha_s * 100)
    cvar_s = terminal_ret_s[terminal_ret_s <= var_s].mean()

    # Key metrics grid
    st.markdown("### Key Risk Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Current State**")
        st.metric("Current FX level", f"{current_fx:.4f}")
        st.metric("Historical volatility (ann.)", f"{sigma * np.sqrt(12):.3f}%")
        if garch_vol:
            st.metric("GARCH volatility", f"{garch_vol * np.sqrt(12):.3f}%")

    with col2:
        st.markdown(f"**{horizon}-month Horizon**")
        st.metric(f"VaR {confidence_level}%", f"{var_s:.3f}%")
        st.metric(f"CVaR {confidence_level}%", f"{cvar_s:.3f}%")
        st.metric("Median expected FX", f"{np.median(terminal_fx_s):.4f}")

    with col3:
        st.markdown("**Tail Probabilities**")
        st.metric("P(depreciation >5%)",
            f"{(terminal_ret_s > 5).mean()*100:.1f}%")
        st.metric("P(depreciation >10%)",
            f"{(terminal_ret_s > 10).mean()*100:.1f}%")
        st.metric("P(appreciation >5%)",
            f"{(terminal_ret_s < -5).mean()*100:.1f}%")

    st.markdown("---")
    st.markdown("### Interpretation for IT Readiness")
    st.markdown(f"""
    - **Volatility level:** The {horizon}-month annualized FX volatility 
      is approximately **{sigma_used * np.sqrt(12):.2f}%**.
    - **Tail risk:** At {confidence_level}% confidence, the maximum {horizon}-month 
      depreciation is **{var_s:.2f}%** — meaning the exchange rate could reach 
      **{current_fx * (1 + var_s/100):.3f}** in a severe scenario.
    - **Policy implication:** {"High FX volatility complicates inflation targeting — BAM would need credible anchors to prevent pass-through amplification." if sigma_used * np.sqrt(12) > 3 else "Moderate FX volatility suggests the MAD is relatively stable, which is favorable for a transition to inflation targeting."}
    """)

    # Feed-forward note
    st.success("📡 These volatility estimates feed directly into **Page 4 (ERPT)** and "
               "**Page 5 (IT Readiness)** to assess the transmission risk and policy readiness score.")
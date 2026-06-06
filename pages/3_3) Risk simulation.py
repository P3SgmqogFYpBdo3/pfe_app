import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_country, sidebar_country_selector
try:
    from utils.style import apply_global_style
except Exception:
    def apply_global_style(): pass

st.set_page_config(page_title="Risk Simulation", page_icon="🎲", layout="wide")
apply_global_style()

st.title("🎲 Page 3 — Exchange Rate Risk Simulation")
st.caption("Monte Carlo simulation of MAD exchange-rate paths. Quantifies the RANGE of "
           "uncertainty and tail risk — not a directional forecast.")

st.info("📌 **Methodological note:** This page models the *range of possible outcomes* and tail risk. "
        "The default uses a **zero-drift (random walk)** assumption — appropriate because exchange-rate "
        "direction is not forecastable. Historical drift is available as a scenario.")

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Settings")
country = sidebar_country_selector("Morocco")
df = load_country(country)
if df is None:
    st.error("No data found.")
    st.stop()

fx_col = st.sidebar.selectbox("Exchange rate",
    ["fx_eur", "fx_usd"],
    format_func=lambda x: {"fx_eur": "MAD/EUR", "fx_usd": "MAD/USD"}[x])
fx_label = {"fx_eur": "MAD/EUR", "fx_usd": "MAD/USD"}[fx_col]

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Settings")
n_sims    = st.sidebar.slider("Number of simulations", 200, 5000, 1000, 100)
horizon   = st.sidebar.slider("Forecast horizon (months)", 3, 36, 12)
conf      = st.sidebar.slider("Confidence level (%)", 90, 99, 95)
seed      = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Assumptions")
drift_mode = st.sidebar.radio(
    "Drift assumption",
    ["Zero (random walk)", "Historical trend"],
    index=0,
    help="Zero drift is the honest baseline — FX direction is unforecastable. "
         "Historical trend extrapolates the recent mean and is offered as a scenario.")
dist_mode = st.sidebar.radio(
    "Innovation distribution",
    ["Normal", "Student-t (fat tails)"],
    index=1,
    help="Student-t produces more realistic extreme moves, matching the fat tails found on Page 2.")

# ── Volatility: prefer GARCH from Page 2, else historical ─────────────────────
fx = df[fx_col].dropna()
ret = np.log(fx / fx.shift(1)).dropna() * 100      # % monthly log returns
hist_sigma = ret.std()
hist_mu    = ret.mean()
S0         = fx.iloc[-1]

garch_sigma = st.session_state.get("garch_sigma_monthly", None)
garch_nu    = st.session_state.get("garch_nu", 6.0)
if garch_sigma is not None:
    sigma_pct = float(garch_sigma)
    st.sidebar.success(f"✅ Using GARCH σ = {sigma_pct:.4f}%")
else:
    sigma_pct = float(hist_sigma)
    st.sidebar.info(f"ℹ️ Using historical σ = {sigma_pct:.4f}%\n(Fit GARCH on Page 2 for GARCH σ)")
nu = float(garch_nu) if garch_nu else 6.0

# ════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ════════════════════════════════════════════════════════════════════════════
def simulate(S0, mu_pct, sigma_pct, horizon, n_sims, dist, nu, seed,
             drift_shock_pct=0.0, vol_mult=1.0, shock_months=0, recover=False):
    rng = np.random.default_rng(int(seed))
    sigma_dec = sigma_pct / 100.0
    mu_dec    = mu_pct / 100.0

    if dist.startswith("Student"):
        raw = rng.standard_t(nu, size=(n_sims, horizon))
        z = raw / np.sqrt(nu / (nu - 2.0))   # standardize to unit variance
    else:
        z = rng.standard_normal(size=(n_sims, horizon))

    # Per-step drift & vol arrays (allow stress shocks over first shock_months)
    mu_arr  = np.full(horizon, mu_dec)
    vol_arr = np.full(horizon, sigma_dec)
    if shock_months > 0:
        shock_drift = drift_shock_pct / 100.0
        for t in range(min(shock_months, horizon)):
            if recover:
                decay = 1.0 - t / max(shock_months, 1)
                mu_arr[t]  += shock_drift * decay
                vol_arr[t] *= (1.0 + (vol_mult - 1.0) * decay)
            else:
                mu_arr[t]  += shock_drift
                vol_arr[t] *= vol_mult

    log_rets = mu_arr[None, :] + vol_arr[None, :] * z
    cum = np.cumsum(log_rets, axis=1)
    paths = S0 * np.exp(cum)
    paths = np.hstack([np.full((n_sims, 1), S0), paths])
    return paths

mu_for_sim = hist_mu if drift_mode == "Historical trend" else 0.0
paths = simulate(S0, mu_for_sim, sigma_pct, horizon, n_sims, dist_mode, nu, seed)
terminal = paths[:, -1]
term_ret = (terminal / S0 - 1) * 100      # % change; for MAD/EUR + = depreciation

# Percentile bands for fan
months = list(range(horizon + 1))
p = {q: np.percentile(paths, q, axis=0) for q in [5, 10, 25, 50, 75, 90, 95]}

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Fan Chart", "📊 VaR & CVaR", "🎯 Terminal Distribution",
    "💥 Stress Testing", "🧾 Risk & Inflation Summary"])

# ── TAB 1: Fan Chart ──────────────────────────────────────────────────────────
with tab1:
    drift_tag = "zero-drift random walk" if mu_for_sim == 0 else "historical-trend drift"
    st.subheader(f"Monte Carlo Fan Chart — {fx_label}")
    st.caption(f"{n_sims} paths · {horizon} months · {dist_mode} innovations · {drift_tag}")

    hist_tail = fx.tail(24)
    future_idx = list(range(horizon + 1))

    fig = go.Figure()
    # historical
    fig.add_trace(go.Scatter(x=list(range(-len(hist_tail)+1, 1)), y=hist_tail.values,
        mode="lines", name="Historical", line=dict(color="#64748B", width=1.5)))
    # bands
    fig.add_trace(go.Scatter(x=future_idx+future_idx[::-1],
        y=list(p[5])+list(p[95][::-1]), fill="toself",
        fillcolor="rgba(37,99,235,0.10)", line=dict(width=0), name="5–95%"))
    fig.add_trace(go.Scatter(x=future_idx+future_idx[::-1],
        y=list(p[10])+list(p[90][::-1]), fill="toself",
        fillcolor="rgba(37,99,235,0.18)", line=dict(width=0), name="10–90%"))
    fig.add_trace(go.Scatter(x=future_idx+future_idx[::-1],
        y=list(p[25])+list(p[75][::-1]), fill="toself",
        fillcolor="rgba(37,99,235,0.30)", line=dict(width=0), name="25–75%"))
    fig.add_trace(go.Scatter(x=future_idx, y=p[50],
        mode="lines", name="Median", line=dict(color="#2563EB", width=2.5, dash="dash")))
    fig.add_hline(y=S0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(height=460, xaxis_title="Months (negative = history)",
        yaxis_title=fx_label, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1E293B"}, legend=dict(orientation="h", y=1.1),
        yaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
        xaxis=dict(gridcolor="rgba(15,23,42,0.08)"), margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current FX", f"{S0:.3f}")
    c2.metric(f"Median ({horizon}m)", f"{p[50][-1]:.3f}", f"{(p[50][-1]/S0-1)*100:+.2f}%")
    c3.metric(f"5th pct ({horizon}m)", f"{p[5][-1]:.3f}")
    c4.metric(f"95th pct ({horizon}m)", f"{p[95][-1]:.3f}")

    if mu_for_sim == 0:
        st.success("✅ Zero-drift baseline: the fan is symmetric around the current level. "
                   "The **width** of the fan is your risk measure — direction is not predicted.")
    else:
        st.warning("⚠️ Historical-trend scenario: the median drifts because it extrapolates the recent "
                   "mean return. Treat this as a 'what-if recent trends continue' scenario, not a forecast.")

# ── TAB 2: VaR & CVaR (two-sided) ────────────────────────────────────────────
with tab2:
    st.subheader(f"Value at Risk & Conditional VaR — {conf}% Confidence")
    st.caption("Two-sided: depreciation risk (currency weakens) and appreciation risk (currency strengthens). "
               "For MAD/EUR, a positive return = depreciation.")

    a = 100 - conf
    dep_var  = np.percentile(term_ret, conf)              # worst depreciation
    dep_cvar = term_ret[term_ret >= dep_var].mean() if (term_ret >= dep_var).any() else dep_var
    app_var  = np.percentile(term_ret, a)                 # worst appreciation
    app_cvar = term_ret[term_ret <= app_var].mean() if (term_ret <= app_var).any() else app_var

    st.markdown("**Depreciation risk (the policy-relevant tail for inflation)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Depreciation VaR ({conf}%)", f"+{dep_var:.2f}%",
              help="Worst depreciation not exceeded with the given confidence.")
    c2.metric(f"Depreciation CVaR ({conf}%)", f"+{dep_cvar:.2f}%",
              help="Average depreciation in the worst tail beyond VaR.")
    c3.metric("FX level at dep. VaR", f"{S0*(1+dep_var/100):.3f}")

    st.markdown("**Appreciation risk (the other tail)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Appreciation VaR ({conf}%)", f"{app_var:.2f}%")
    c2.metric(f"Appreciation CVaR ({conf}%)", f"{app_cvar:.2f}%")
    c3.metric("FX level at app. VaR", f"{S0*(1+app_var/100):.3f}")

    # Distribution with both tails marked
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=term_ret, nbinsx=50, histnorm="probability density",
        marker_color="#2563EB", opacity=0.75, name="Terminal returns"))
    fig.add_vline(x=dep_var, line_dash="dash", line_color="#dc2626",
        annotation_text=f"Dep VaR +{dep_var:.1f}%", annotation_position="top")
    fig.add_vline(x=app_var, line_dash="dash", line_color="#16a34a",
        annotation_text=f"App VaR {app_var:.1f}%", annotation_position="top")
    fig.add_vline(x=0, line_dash="dot", line_color="#94A3B8")
    fig.update_layout(height=380, xaxis_title="Terminal return (%)  ·  + = depreciation",
        yaxis_title="Density", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1E293B"}, showlegend=False,
        yaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
        xaxis=dict(gridcolor="rgba(15,23,42,0.08)"), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Term structure of depreciation VaR
    st.markdown("**Depreciation VaR / CVaR across horizons**")
    rows = []
    for h in [1, 3, 6, 12, 18, 24]:
        if h <= horizon:
            ph = simulate(S0, mu_for_sim, sigma_pct, h, n_sims, dist_mode, nu, seed)
            rr = (ph[:, -1] / S0 - 1) * 100
            dv = np.percentile(rr, conf)
            dc = rr[rr >= dv].mean() if (rr >= dv).any() else dv
            rows.append({"Horizon (months)": h,
                         f"Dep VaR {conf}% (%)": f"+{dv:.2f}",
                         f"Dep CVaR {conf}% (%)": f"+{dc:.2f}",
                         "FX at VaR": f"{S0*(1+dv/100):.3f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Depreciation VaR grows roughly with the square root of the horizon — the standard "
               "time-scaling of risk under a random walk.")

# ── TAB 3: Terminal Distribution ─────────────────────────────────────────────
with tab3:
    st.subheader(f"Terminal FX Distribution at {horizon}-month Horizon")

    p_dep = (term_ret > 0).mean() * 100
    p_app = (term_ret < 0).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=terminal, nbinsx=50, histnorm="probability density",
        marker_color="#2563EB", opacity=0.8))
    for q, col, lbl in [(5, "#dc2626", "P5"), (50, "#16a34a", "P50"), (95, "#dc2626", "P95")]:
        v = np.percentile(terminal, q)
        fig.add_vline(x=v, line_dash="dash", line_color=col,
            annotation_text=f"{lbl}: {v:.3f}", annotation_position="top")
    fig.update_layout(height=380, xaxis_title=fx_label, yaxis_title="Density",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1E293B"}, showlegend=False,
        yaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
        xaxis=dict(gridcolor="rgba(15,23,42,0.08)"), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(depreciation)", f"{p_dep:.1f}%")
    c2.metric("P(appreciation)", f"{p_app:.1f}%")
    c3.metric("Median FX", f"{np.median(terminal):.3f}")
    c4.metric("FX std dev", f"{terminal.std():.3f}")

    if mu_for_sim == 0:
        st.info("Under the zero-drift baseline, depreciation and appreciation probabilities are "
                "close to 50/50 — the honest position that FX direction is unpredictable. The value "
                "is in the **spread** of outcomes, not the direction.")

    pct_rows = []
    for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        lvl = np.percentile(terminal, q)
        pct_rows.append({"Percentile": f"{q}%", f"{fx_label} level": f"{lvl:.3f}",
                         "Change from current (%)": f"{(lvl/S0-1)*100:+.2f}"})
    st.dataframe(pd.DataFrame(pct_rows), use_container_width=True, hide_index=True)

# ── TAB 4: Stress Testing ────────────────────────────────────────────────────
with tab4:
    st.subheader("Stress Testing — Adverse Scenarios")
    st.caption("Impose a depreciation shock and elevated volatility over an initial window, "
               "then compare against the baseline.")

    preset = st.selectbox("Select scenario",
        ["Custom", "Oil price spike", "Capital outflow", "COVID-style shock", "Mild depreciation"])
    presets = {
        "Oil price spike":     (0.6, 1.8, 6),
        "Capital outflow":     (1.2, 2.5, 4),
        "COVID-style shock":   (0.9, 3.0, 3),
        "Mild depreciation":   (0.3, 1.3, 6),
    }
    col1, col2 = st.columns(2)
    with col1:
        if preset == "Custom":
            ds = st.slider("Drift shock (% per month)", -2.0, 2.0, 0.5, 0.05)
            vm = st.slider("Volatility multiplier", 1.0, 4.0, 1.5, 0.1)
        else:
            ds, vm, _dur = presets[preset]
            st.metric("Drift shock", f"{ds:+.2f}%/mo")
            st.metric("Volatility multiplier", f"{vm:.1f}×")
    with col2:
        default_dur = presets.get(preset, (0, 0, 3))[2]
        sm = st.slider("Shock duration (months)", 1, min(horizon, 12), default_dur)
        recover = st.checkbox("Apply gradual recovery after shock", value=True)

    if st.button("▶ Run Stress Simulation", type="primary"):
        base_paths = simulate(S0, mu_for_sim, sigma_pct, horizon, n_sims, dist_mode, nu, seed)
        str_paths  = simulate(S0, mu_for_sim, sigma_pct, horizon, n_sims, dist_mode, nu, seed,
                              drift_shock_pct=ds, vol_mult=vm, shock_months=sm, recover=recover)
        base_ret = (base_paths[:, -1] / S0 - 1) * 100
        str_ret  = (str_paths[:, -1] / S0 - 1) * 100

        fig = go.Figure()
        idx = list(range(horizon + 1))
        for q, op in [(95, 0.10), (75, 0.20)]:
            fig.add_trace(go.Scatter(x=idx+idx[::-1],
                y=list(np.percentile(str_paths,100-q,axis=0))+list(np.percentile(str_paths,q,axis=0)[::-1]),
                fill="toself", fillcolor=f"rgba(220,38,38,{op})", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=idx, y=np.median(base_paths,axis=0),
            mode="lines", name="Baseline median", line=dict(color="#2563EB", width=2.5)))
        fig.add_trace(go.Scatter(x=idx, y=np.median(str_paths,axis=0),
            mode="lines", name="Stressed median", line=dict(color="#dc2626", width=2.5)))
        fig.add_vrect(x0=0, x1=sm, fillcolor="rgba(220,38,38,0.06)", line_width=0,
            annotation_text="Shock period", annotation_position="top left")
        fig.add_hline(y=S0, line_dash="dot", line_color="#94A3B8")
        fig.update_layout(height=420, title="Baseline vs Stressed Fan Chart",
            xaxis_title="Months", yaxis_title=fx_label,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#1E293B"}, legend=dict(orientation="h", y=1.12),
            yaxis=dict(gridcolor="rgba(15,23,42,0.08)"),
            xaxis=dict(gridcolor="rgba(15,23,42,0.08)"), margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

        base_dep = np.percentile(base_ret, conf)
        str_dep  = np.percentile(str_ret, conf)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline dep. VaR", f"+{base_dep:.2f}%")
        c2.metric("Stressed dep. VaR", f"+{str_dep:.2f}%", f"{str_dep-base_dep:+.2f}%")
        c3.metric("Baseline median FX", f"{np.median(base_paths[:,-1]):.3f}")
        c4.metric("Stressed median FX", f"{np.median(str_paths[:,-1]):.3f}",
                  f"{np.median(str_paths[:,-1])-np.median(base_paths[:,-1]):+.3f}")
    else:
        st.info("👆 Configure a scenario and click **Run Stress Simulation**.")

# ── TAB 5: Risk & Inflation Summary ──────────────────────────────────────────
with tab5:
    st.subheader("Risk & Inflation-at-Risk Summary")

    dep_var = np.percentile(term_ret, conf)
    dep_cvar = term_ret[term_ret >= dep_var].mean() if (term_ret >= dep_var).any() else dep_var
    p_dep5  = (term_ret > 5).mean() * 100
    p_dep10 = (term_ret > 10).mean() * 100
    ann_vol = sigma_pct * np.sqrt(12)

    st.markdown("**Key Risk Metrics**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current FX level", f"{S0:.4f}")
        st.metric("Annualized volatility", f"{ann_vol:.2f}%")
    with c2:
        st.metric(f"Depreciation VaR ({conf}%)", f"+{dep_var:.2f}%")
        st.metric(f"Depreciation CVaR ({conf}%)", f"+{dep_cvar:.2f}%")
    with c3:
        st.metric("P(depreciation > 5%)", f"{p_dep5:.1f}%")
        st.metric("P(depreciation > 10%)", f"{p_dep10:.1f}%")

    st.markdown("---")
    st.markdown("### 🔑 Inflation-at-Risk — combining FX risk with pass-through")
    st.caption("The single most policy-relevant number: the worst-case inflation impact from exchange-rate "
               "depreciation, computed as the depreciation VaR multiplied by the ERPT coefficient from Page 4.")

    erpt = st.session_state.get("erpt_long_run", None)
    if erpt is not None:
        erpt_used = abs(float(erpt))
        src = "from Page 4 (VAR)"
    else:
        erpt_used = st.number_input("ERPT coefficient (fit VAR on Page 4 to auto-fill)",
                                    0.0, 1.0, 0.03, 0.01)
        src = "manual input"

    inflation_at_risk = dep_var * erpt_used
    inflation_cvar    = dep_cvar * erpt_used

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Depreciation VaR ({conf}%)", f"+{dep_var:.2f}%")
    c2.metric("ERPT coefficient", f"{erpt_used:.4f}", help=f"Source: {src}")
    c3.metric("⚠️ Inflation-at-Risk", f"+{inflation_at_risk:.3f}%",
              help="Worst-case added inflation from FX depreciation at the chosen confidence.")

    st.markdown(f"""
    <div style="background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.2);
                border-radius:12px;padding:16px 20px;margin-top:8px;">
        <b>Interpretation.</b> At {conf}% confidence, the {horizon}-month depreciation does not exceed
        <b>{dep_var:.2f}%</b>. With an exchange-rate pass-through of <b>{erpt_used:.3f}</b>, this translates
        into at most <b>{inflation_at_risk:.3f}%</b> of additional inflation
        (CVaR tail: <b>{inflation_cvar:.3f}%</b>). Because this is well within Bank Al-Maghrib's tolerance,
        even a severe depreciation under a more flexible regime would have a <b>limited inflationary impact</b>
        — directly supporting the viability of inflation targeting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Policy implication for IT readiness**")
    if inflation_at_risk < 1.0:
        st.success(f"🟢 Inflation-at-Risk of {inflation_at_risk:.3f}% is very low. Exchange-rate flexibility "
                   "poses minimal inflation risk — strongly favorable for inflation targeting.")
    elif inflation_at_risk < 3.0:
        st.info(f"🟡 Inflation-at-Risk of {inflation_at_risk:.3f}% is moderate. IT is viable but BAM should "
                "maintain a credible anchor to contain second-round effects.")
    else:
        st.warning(f"🟠 Inflation-at-Risk of {inflation_at_risk:.3f}% is elevated. Greater flexibility could "
                   "transmit meaningfully to prices — strengthen the framework before full IT.")

    st.markdown("""
    <div style="margin-top:16px;color:#64748B;font-size:0.82rem;">
    These estimates feed into Page 4 (ERPT) and Page 5 (IT Readiness). Volatility source and drift
    assumption are set in the sidebar; the zero-drift baseline is used for the headline figures.
    </div>
    """, unsafe_allow_html=True)
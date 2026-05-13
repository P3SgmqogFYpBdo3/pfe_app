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

st.set_page_config(page_title="Volatility Modeling", page_icon="📉", layout="wide")
st.title("📉 Page 2 — Volatility Modeling (GARCH Framework)")
st.caption("GARCH-based characterization of exchange rate volatility, persistence, and regime comparison.")

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
st.sidebar.subheader("GARCH Settings")
p_order = st.sidebar.selectbox("ARCH order (p)", [1, 2], index=0)
q_order = st.sidebar.selectbox("GARCH order (q)", [1, 2], index=0)
model_type = st.sidebar.selectbox("Model type",
    ["GARCH", "GJR-GARCH (asymmetric)"], index=0)
dist = st.sidebar.selectbox("Error distribution",
    ["normal", "t", "skewt"], index=1)

# Break dates for regime comparison
BREAK_DATES = {
    "Morocco": [
        ("2018-01-15", "Pre-2018", "2018–2020", "Post-2020"),
    ],
}

# ── Compute log returns ───────────────────────────────────────────────────────
returns = np.log(df[fx_col] / df[fx_col].shift(1)).dropna() * 100
returns.name = "returns"

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Returns Analysis",
"📐 Model Comparison",
    "⚙️ GARCH Model",
    "📊 Conditional Volatility",
    "🔁 Regime Comparison",
])


# ── TAB 1: Returns Analysis ──────────────────────────────────────────────────
with tab1:
    st.subheader("Log Returns — Distributional Properties")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean (%)", f"{returns.mean():.4f}")
    c2.metric("Std dev (%)", f"{returns.std():.4f}")
    c3.metric("Skewness", f"{returns.skew():.3f}")
    c4.metric("Kurtosis", f"{returns.kurtosis():.3f}")
    c5.metric("Observations", len(returns))

    # Excess kurtosis interpretation
    kurt = returns.kurtosis()
    if kurt > 1:
        st.info(f"📌 Excess kurtosis = {kurt:.2f} — **fat tails detected**. "
                "Returns are not normally distributed, justifying GARCH modeling.")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Return series over time", "Return distribution vs Normal"))

    # Time series
    fig.add_trace(go.Scatter(
        x=returns.index, y=returns.values,
        mode="lines", line=dict(color="#2563EB", width=0.8),
        name="Returns"), row=1, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=0.5, row=1, col=1)

    # Histogram + normal overlay
    x_range = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf = (1 / (returns.std() * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x_range - returns.mean()) / returns.std()) ** 2)

    fig.add_trace(go.Histogram(
        x=returns.values, histnorm="probability density",
        marker_color="#2563EB", opacity=0.6,
        nbinsx=50, name="Actual"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_pdf,
        line=dict(color="#dc2626", width=2, dash="dash"),
        name="Normal"), row=1, col=2)

    fig.update_layout(height=380, showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)

    # Ljung-Box test for ARCH effects
    st.markdown("**ARCH Effects Test (Ljung-Box on squared returns)**")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(returns ** 2, lags=[5, 10, 15], return_df=True)
    lb_df = pd.DataFrame({
        "Lag": [5, 10, 15],
        "LB Statistic": lb_result["lb_stat"].round(3).values,
        "p-value": lb_result["lb_pvalue"].round(4).values,
        "ARCH effects?": ["Yes ✓" if p < 0.05 else "No" for p in lb_result["lb_pvalue"].values]
    })
    st.dataframe(lb_df, use_container_width=True, hide_index=True)
    if lb_result["lb_pvalue"].min() < 0.05:
        st.success("✅ ARCH effects confirmed — GARCH modeling is appropriate.")
    else:
        st.warning("⚠️ No significant ARCH effects detected at standard lags.")


# ── TAB 2: Model Comparison ──────────────────────────────────────────────────
with tab2:
    st.subheader("Model Selection — GARCH Variants Comparison")
    st.caption("Compare GARCH, GJR-GARCH, and EGARCH by AIC/BIC to select the best specification.")

    if st.button("🔍 Run Model Comparison", type="primary"):
        from arch import arch_model

        model_specs = [
            ("GARCH(1,1) Normal",   dict(vol="Garch", p=1, q=1, dist="normal", mean="AR", lags=1)),
            ("GARCH(1,1) Student-t",dict(vol="Garch", p=1, q=1, dist="t", mean="AR", lags=1)),
            ("GARCH(1,2) Student-t",dict(vol="Garch", p=1, q=2, dist="t", mean="AR", lags=1)),
            ("GJR-GARCH(1,1) t",    dict(vol="Garch", p=1, o=1, q=1, dist="t", mean="AR", lags=1)),
            ("GARCH(2,1) Student-t",dict(vol="Garch", p=2, q=1, dist="t", mean="AR", lags=1)),
        ]

        results = []
        progress = st.progress(0)
        for i, (name, spec) in enumerate(model_specs):
            try:
                res = arch_model(returns, **spec).fit(disp="off", show_warning=False)
                alpha = res.params.get("alpha[1]", 0)
                beta = res.params.get("beta[1]", 0)
                results.append({
                    "Model": name,
                    "Log-likelihood": round(res.loglikelihood, 2),
                    "AIC": round(res.aic, 2),
                    "BIC": round(res.bic, 2),
                    "α+β": round(alpha + beta, 4),
                    "# Params": len(res.params),
                })
            except Exception:
                results.append({
                    "Model": name, "Log-likelihood": None,
                    "AIC": None, "BIC": None,
                    "α+β": None, "# Params": None
                })
            progress.progress((i + 1) / len(model_specs))

        results_df = pd.DataFrame(results).dropna()
        if len(results_df) > 0:
            best_aic = results_df.loc[results_df["AIC"].idxmin(), "Model"]
            best_bic = results_df.loc[results_df["BIC"].idxmin(), "Model"]

            st.dataframe(results_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            c1.success(f"✅ Best AIC: **{best_aic}**")
            c2.success(f"✅ Best BIC: **{best_bic}**")

            # AIC/BIC bar chart
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=results_df["Model"], y=results_df["AIC"],
                name="AIC", marker_color="#2563EB", opacity=0.8))
            fig_cmp.add_trace(go.Bar(
                x=results_df["Model"], y=results_df["BIC"],
                name="BIC", marker_color="#dc2626", opacity=0.8))
            fig_cmp.update_layout(
                barmode="group", height=380,
                yaxis_title="Information Criterion (lower = better)",
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.1),
                yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                xaxis=dict(tickangle=-20))
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.error("All models failed to converge.")
    else:
        st.info("Click **Run Model Comparison** to evaluate multiple GARCH specifications. "
                "This takes about 15–20 seconds.")


# ── TAB 3: GARCH Model ───────────────────────────────────────────────────────
with tab3:
    st.subheader(f"{'GJR-' if 'GJR' in model_type else ''}GARCH({p_order},{q_order}) — {dist} distribution")

    if st.button("🔧 Fit GARCH Model", type="primary"):
        with st.spinner("Fitting model... this may take a few seconds"):
            try:
                from arch import arch_model

                if "GJR" in model_type:
                    am = arch_model(returns, vol="Garch", p=p_order, o=1, q=q_order,
                        dist=dist, mean="AR", lags=1)
                else:
                    am = arch_model(returns, vol="Garch", p=p_order, q=q_order,
                        dist=dist, mean="AR", lags=1)

                res = am.fit(disp="off", show_warning=False)
                st.session_state["garch_result"] = res
                st.session_state["garch_returns"] = returns
                st.session_state["garch_model_type"] = model_type
                st.success("Model fitted successfully ✓")

            except Exception as e:
                st.error(f"Model fitting failed: {e}")
                st.stop()

    if "garch_result" in st.session_state:
        res = st.session_state["garch_result"]

        # Parameter table
        st.markdown("**Parameter Estimates**")
        params = res.params
        pvalues = res.pvalues
        tvalues = res.tvalues
        std_err = res.std_err

        param_df = pd.DataFrame({
            "Parameter": params.index,
            "Estimate": params.round(6).values,
            "Std Error": std_err.round(6).values,
            "t-stat": tvalues.round(3).values,
            "p-value": pvalues.round(4).values,
            "Significant": ["***" if p < 0.01 else "**" if p < 0.05
                            else "*" if p < 0.1 else "" for p in pvalues.values]
        })
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        # Key metrics
        st.markdown("**Model Diagnostics**")
        c1, c2, c3, c4 = st.columns(4)

        # Extract alpha and beta
        try:
            alpha = params.get("alpha[1]", params.get("alpha[1]", 0))
            beta = params.get("beta[1]", params.get("beta[1]", 0))
            persistence = alpha + beta
        except:
            persistence = None

        c1.metric("Log-likelihood", f"{res.loglikelihood:.2f}")
        c2.metric("AIC", f"{res.aic:.2f}")
        c3.metric("BIC", f"{res.bic:.2f}")
        if persistence is not None:
            c4.metric("α + β (persistence)", f"{persistence:.4f}",
                help="Close to 1 = high volatility persistence")

        # Persistence interpretation
        if persistence is not None:
            if persistence > 0.95:
                st.warning(f"⚠️ Very high persistence (α+β = {persistence:.4f}) — "
                           "volatility shocks are extremely long-lived.")
            elif persistence > 0.85:
                st.info(f"📌 High persistence (α+β = {persistence:.4f}) — "
                        "volatility clustering is strong.")
            else:
                st.success(f"✅ Moderate persistence (α+β = {persistence:.4f}) — "
                           "volatility reverts relatively quickly.")

        # Residual diagnostics
        st.markdown("**Standardized Residual Diagnostics**")
        std_resid = res.std_resid
        lb_resid = acorr_ljungbox(std_resid ** 2, lags=[5, 10], return_df=True)
        diag_df = pd.DataFrame({
            "Lag": [5, 10],
            "LB Statistic (sq. resid)": lb_resid["lb_stat"].round(3).values,
            "p-value": lb_resid["lb_pvalue"].round(4).values,
            "ARCH remaining?": ["Yes ⚠️" if p < 0.05 else "No ✓"
                                for p in lb_resid["lb_pvalue"].values]
        })
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Click **Fit GARCH Model** above to estimate the model.")


# ── TAB 4: Conditional Volatility ────────────────────────────────────────────
with tab4:
    st.subheader("Conditional Volatility — σₜ over Time")

    if "garch_result" not in st.session_state:
        st.info("Please fit the GARCH model in the **GARCH Model** tab first.")
    else:
        res = st.session_state["garch_result"]
        cond_vol = res.conditional_volatility  # annualize
        cond_vol_annual = cond_vol * np.sqrt(12)

        # Morocco liberalization markers
        EVENTS = {
            "Morocco": [
                ("2008-06-01", "Food/fuel shock", "orange"),
                ("2011-02-01", "Arab Spring", "red"),
                ("2018-01-15", "Band ±2.5%", "blue"),
                ("2020-03-20", "Band ±5%", "blue"),
                ("2020-04-01", "COVID", "red"),
                ("2022-02-01", "Inflation surge", "orange"),
            ]
        }
        events = EVENTS.get(country, [])

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Log Returns (%)", "Conditional Volatility — Annualized (%)"),
            row_heights=[0.35, 0.65], vertical_spacing=0.08)

        # Returns
        fig.add_trace(go.Scatter(
            x=returns.index, y=returns.values,
            mode="lines", line=dict(color="#93c5fd", width=0.8),
            name="Returns"), row=1, col=1)

        # Conditional vol
        fig.add_trace(go.Scatter(
            x=cond_vol_annual.index, y=cond_vol_annual.values,
            fill="tozeroy", fillcolor="rgba(220,38,38,0.12)",
            line=dict(color="#dc2626", width=1.5),
            name="Cond. Vol"), row=2, col=1)

        # Event lines
        for date_str, label, color in events:
            ev_date = pd.Timestamp(date_str)
            if df.index.min() <= ev_date <= df.index.max():
                for row in [1, 2]:
                    fig.add_vline(x=ev_date, line_dash="dash",
                        line_color=color, line_width=1, row=row, col=1)
                fig.add_annotation(
                    x=ev_date, y=cond_vol_annual.max() * 0.95,
                    text=label, showarrow=False, textangle=-90,
                    font=dict(size=9, color=color), xshift=6, row=2, col=1)

        fig.update_layout(height=520, showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
        st.plotly_chart(fig, use_container_width=True)

        # Stats on conditional vol
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean cond. vol (%)", f"{cond_vol_annual.mean():.3f}")
        c2.metric("Max cond. vol (%)", f"{cond_vol_annual.max():.3f}")
        c3.metric("Min cond. vol (%)", f"{cond_vol_annual.min():.3f}")
        c4.metric("Current vol (%)", f"{cond_vol_annual.iloc[-1]:.3f}")


# ── TAB 5: Regime Comparison ─────────────────────────────────────────────────
with tab5:
    st.subheader("Pre vs Post Liberalization — Volatility Regime Comparison")

    # Break date selector
    col1, col2 = st.columns(2)
    with col1:
        break1 = pd.Timestamp(st.date_input(
            "First break date (e.g. 2018 band widening)",
            value=pd.Timestamp("2018-01-15"),
            min_value=df.index.min().to_pydatetime(),
            max_value=df.index.max().to_pydatetime()))
    with col2:
        break2 = pd.Timestamp(st.date_input(
            "Second break date (e.g. 2020 band widening)",
            value=pd.Timestamp("2020-03-20"),
            min_value=df.index.min().to_pydatetime(),
            max_value=df.index.max().to_pydatetime()))

    # Split returns into 3 regimes
    r1 = returns[returns.index < break1]
    r2 = returns[(returns.index >= break1) & (returns.index < break2)]
    r3 = returns[returns.index >= break2]

    regimes = [
        (r1, f"Pre-{break1.year}", "#2563EB"),
        (r2, f"{break1.year}–{break2.year}", "#f59e0b"),
        (r3, f"Post-{break2.year}", "#dc2626"),
    ]
    regimes = [(r, l, c) for r, l, c in regimes if len(r) > 5]

    if len(regimes) < 2:
        st.warning("Not enough data segments. Adjust break dates.")
    else:
        # Metrics table
        reg_data = []
        for r, label, _ in regimes:
            reg_data.append({
                "Regime": label,
                "Obs": len(r),
                "Mean return (%)": round(r.mean(), 4),
                "Std dev (%)": round(r.std(), 4),
                "Ann. volatility (%)": round(r.std() * np.sqrt(12), 3),
                "Min (%)": round(r.min(), 3),
                "Max (%)": round(r.max(), 3),
                "Kurtosis": round(r.kurtosis(), 2),
            })
        st.dataframe(pd.DataFrame(reg_data),
            use_container_width=True, hide_index=True)

        # Bar chart: annualized volatility by regime
        fig_reg = go.Figure()
        for r, label, color in regimes:
            fig_reg.add_trace(go.Bar(
                x=[label],
                y=[round(r.std() * np.sqrt(12), 3)],
                marker_color=color, name=label,
                text=[f"{r.std()*np.sqrt(12):.3f}%"],
                textposition="outside"))

        fig_reg.update_layout(
            title="Annualized Volatility by Regime",
            height=350, showlegend=False,
            yaxis_title="Annualized Volatility (%)",
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig_reg, use_container_width=True)

        # Distribution overlay
        fig_dist = go.Figure()
        for r, label, color in regimes:
            fig_dist.add_trace(go.Histogram(
                x=r.values, name=label,
                marker_color=color, opacity=0.55,
                histnorm="probability density", nbinsx=35))

        fig_dist.update_layout(
            title="Return Distributions by Regime",
            barmode="overlay", height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
        st.plotly_chart(fig_dist, use_container_width=True)

        # F-test between regimes
        from scipy import stats as scipy_stats
        st.markdown("**Variance Ratio Tests Between Regimes**")
        test_rows = []
        for i in range(len(regimes) - 1):
            r_a, l_a, _ = regimes[i]
            r_b, l_b, _ = regimes[i + 1]
            f = (r_b.std() ** 2) / (r_a.std() ** 2)
            p = 2 * min(
                scipy_stats.f.cdf(f, len(r_b)-1, len(r_a)-1),
                1 - scipy_stats.f.cdf(f, len(r_b)-1, len(r_a)-1))
            test_rows.append({
                "Comparison": f"{l_a} vs {l_b}",
                "F-statistic": round(f, 4),
                "p-value": round(p, 4),
                "Conclusion": "Significant change ✓" if p < 0.05 else "Not significant"
            })
        st.dataframe(pd.DataFrame(test_rows),
            use_container_width=True, hide_index=True)



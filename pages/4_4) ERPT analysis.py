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

st.set_page_config(page_title="ERPT Analysis", page_icon="📡", layout="wide")
st.title("📡 Page 4 — Exchange Rate Pass-Through (ERPT) Analysis")
st.caption("VAR/BVAR model estimating how exchange rate shocks transmit to domestic inflation.")

st.info("🎯 **Core question:** How much does a 1% depreciation of the MAD affect CPI inflation, "
        "and over what horizon does this transmission occur?")

# ── Helper: robust vertical line for datetime axes ───────────────────────────
def add_event_line(fig, ev_timestamp, label, color="orange"):
    """Add a vertical dashed line + annotation on a datetime x-axis.
    Uses add_shape instead of add_vline to avoid the plotly datetime+annotation
    'int + str' TypeError bug."""
    fig.add_shape(
        type="line",
        xref="x", yref="paper",
        x0=ev_timestamp, x1=ev_timestamp,
        y0=0, y1=1,
        line=dict(color=color, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        xref="x", yref="paper",
        x=ev_timestamp, y=1.02,
        text=label, showarrow=False,
        font=dict(color=color, size=10),
        xanchor="left",
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
country = sidebar_country_selector("Morocco")
df = load_country(country)

if df is None:
    st.error(f"No data found for {country}.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("VAR Settings")

fx_col = st.sidebar.selectbox("Exchange rate variable",
    ["fx_eur", "fx_usd", "neer"],
    format_func=lambda x: {"fx_eur": "MAD/EUR", "fx_usd": "MAD/USD",
                            "neer": "NEER Index"}[x])

# NEER sign: NEER up = appreciation → flip so ERPT reads as depreciation→inflation
neer_sign = -1 if fx_col == "neer" else 1

var_variables = st.sidebar.multiselect(
    "VAR variables (order matters)",
    ["fx", "cpi", "policy_rate", "oil_brent", "reserves", "output_gap"],
    default=["fx", "cpi", "policy_rate"],
    help="First variable = most exogenous. Recommended: fx → oil → cpi → rate → reserves → gap")

max_lags = st.sidebar.slider("Max lags for selection", 1, 12, 12)
irf_horizon = st.sidebar.slider("IRF horizon (months)", 6, 36, 24)
use_bvar = st.sidebar.checkbox("Use BVAR (Bayesian shrinkage)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Break Dates")
use_regimes = st.sidebar.checkbox("Regime-specific ERPT", value=False)

col_map = {
    "fx": fx_col, "cpi": "cpi", "policy_rate": "policy_rate",
    "oil_brent": "oil_brent", "reserves": "reserves", "output_gap": "output_gap"
}
label_map = {
    "fx": "Exchange Rate", "cpi": "CPI", "policy_rate": "Policy Rate",
    "oil_brent": "Oil Price", "reserves": "FX Reserves", "output_gap": "Output Gap"
}

# ── Prepare VAR data ──────────────────────────────────────────────────────────
def prepare_var_data(df, variables, col_map):
    data = {}
    for v in variables:
        col = col_map.get(v)
        if col not in df.columns:
            continue
        if v == "output_gap":
            data[v] = df[col].diff().dropna()
        elif v == "policy_rate":
            data[v] = df[col].diff().dropna()
        else:
            ret = np.log(df[col] / df[col].shift(1)).dropna() * 100
            mean_r, std_r = ret.mean(), ret.std()
            ret = ret.clip(lower=mean_r - 5*std_r, upper=mean_r + 5*std_r)
            data[v] = ret
    return pd.DataFrame(data).dropna()

if len(var_variables) < 2:
    st.warning("Please select at least 2 variables in the sidebar.")
    st.stop()

var_data = prepare_var_data(df, var_variables, col_map)

# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 Pre-Estimation Diagnostics",
    "⚙️ VAR Model",
    "📈 Impulse Response (IRF)",
    "📊 Variance Decomposition",
    "📐 ERPT Coefficient",
    "🔄 Rolling ERPT"
])


# ── TAB 1: Pre-Estimation Diagnostics ────────────────────────────────────────
with tab1:
    st.subheader("Pre-Estimation Diagnostics")
    st.markdown("**Step 1: Stationarity Tests (ADF)**")
    st.caption("VAR requires stationary variables. We use log-differences which typically achieve stationarity.")

    from statsmodels.tsa.stattools import adfuller

    adf_results = []
    for v in var_variables:
        col = col_map.get(v)
        if col not in df.columns:
            continue
        raw = df[col].dropna()
        adf_level = adfuller(raw, autolag="AIC")
        if v in ["output_gap", "policy_rate"]:
            diff = df[col].diff().dropna()
        else:
            diff = np.log(df[col] / df[col].shift(1)).dropna() * 100
        adf_diff = adfuller(diff, autolag="AIC")
        adf_results.append({
            "Variable": label_map.get(v, v),
            "ADF (level) stat": round(adf_level[0], 3),
            "ADF (level) p-val": round(adf_level[1], 4),
            "Stationary level?": "✅" if adf_level[1] < 0.05 else "❌",
            "ADF (diff) stat": round(adf_diff[0], 3),
            "ADF (diff) p-val": round(adf_diff[1], 4),
            "Stationary diff?": "✅" if adf_diff[1] < 0.05 else "⚠️",
        })

    st.dataframe(pd.DataFrame(adf_results), use_container_width=True, hide_index=True)

    non_stat = [r for r in adf_results if r["Stationary diff?"] != "✅"]
    if non_stat:
        st.warning("⚠️ Some variables may still be non-stationary in differences.")
    else:
        st.success("✅ All variables appear stationary in first differences — VAR in differences is appropriate.")

    st.markdown("---")
    st.markdown("**Step 2: Lag Length Selection**")

    from statsmodels.tsa.vector_ar.var_model import VAR

    try:
        var_model_sel = VAR(var_data)
        lag_results = var_model_sel.select_order(maxlags=max_lags)

        aic_vals = lag_results.ics["aic"]
        bic_vals = lag_results.ics["bic"]
        hqic_vals = lag_results.ics["hqic"]

        lag_df = pd.DataFrame({
            "Lag": list(range(1, len(aic_vals) + 1)),
            "AIC":  [round(v, 2) for v in aic_vals],
            "BIC":  [round(v, 2) for v in bic_vals],
            "HQIC": [round(v, 2) for v in hqic_vals],
        })

        best_aic = int(np.argmin(aic_vals)) + 1
        best_bic = int(np.argmin(bic_vals)) + 1

        st.dataframe(lag_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.success(f"✅ Best lag (AIC): **{best_aic}**")
        c2.success(f"✅ Best lag (BIC): **{best_bic}**")
        st.caption("💡 BIC is more conservative (fewer lags, less overfitting). "
                   "For your defense, BIC's choice is the safer baseline; AIC's is a robustness check.")

        st.session_state["best_lag"] = best_bic

    except Exception as e:
        st.error(f"Lag selection failed: {e}")

    st.markdown("---")
    st.markdown("**Step 3: Correlation Matrix**")
    corr = var_data.rename(columns=label_map).corr().round(3)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}", textfont_size=11))
    fig_corr.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)


# ── TAB 2: VAR Model ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("VAR Model Estimation")

    lag_choice = st.number_input("Number of lags",
        min_value=1, max_value=12,
        value=st.session_state.get("best_lag", 2))

    if st.button("⚙️ Fit VAR Model", type="primary"):
        with st.spinner("Fitting VAR..."):
            try:
                from statsmodels.tsa.vector_ar.var_model import VAR
                model = VAR(var_data)
                results = model.fit(lag_choice)
                st.session_state["var_results"] = results
                st.session_state["var_lag"] = lag_choice
                st.session_state["var_data_fitted"] = var_data
                st.session_state["var_variables_fitted"] = var_variables
                st.session_state["neer_sign"] = neer_sign
                st.success(f"✅ VAR({lag_choice}) fitted on {len(var_data)} observations.")
            except Exception as e:
                st.error(f"VAR fitting failed: {e}")
                st.stop()

    if "var_results" in st.session_state:
        results = st.session_state["var_results"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables", len(st.session_state.get("var_variables_fitted", var_variables)))
        c2.metric("Lags", st.session_state.get("var_lag", lag_choice))
        c3.metric("Observations", results.nobs)
        c4.metric("AIC", f"{results.aic:.2f}")

        if "cpi" in results.names:
            st.markdown("**CPI Equation Coefficients** (most relevant for ERPT)")
            cpi_eq   = results.params["cpi"]
            cpi_pval = results.pvalues["cpi"]
            coef_df  = pd.DataFrame({
                "Regressor":   cpi_eq.index,
                "Coefficient": cpi_eq.round(5).values,
                "p-value":     cpi_pval.round(4).values,
                "Significant": ["***" if p < 0.01 else "**" if p < 0.05
                                 else "*" if p < 0.1 else "" for p in cpi_pval.values]
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown("**VAR Stability Check**")
        try:
            stable = results.is_stable()
            roots  = results.roots
            min_root = np.abs(roots).min()
            c1, c2 = st.columns(2)
            c1.metric("Min root modulus", f"{min_root:.4f}",
                help="Roots of characteristic polynomial — all must be > 1 for stability")
            if stable:
                c2.success("✅ VAR is stable (all roots outside unit circle)")
            else:
                c2.error("❌ VAR is unstable — reduce lags or check data")
        except Exception:
            st.info("Stability check not available.")

        st.markdown("**Residual Diagnostics (Portmanteau Test)**")
        try:
            pt = results.test_whiteness(nlags=10)
            conclusion = "✅ No serial correlation" if pt.pvalue > 0.05 else "⚠️ Serial correlation detected"
            st.write(f"Portmanteau statistic: **{pt.test_statistic:.3f}** | "
                     f"p-value: **{pt.pvalue:.4f}** | {conclusion}")
            if pt.pvalue < 0.05:
                st.caption("Note: Serial correlation is common in models spanning structural breaks (2018, 2020). "
                           "Coefficient estimates remain unbiased — only efficiency is affected.")
        except Exception:
            st.info("Portmanteau test not available.")
    else:
        st.info("👆 Click **Fit VAR Model** to estimate.")


# ── TAB 3: Impulse Response Functions ────────────────────────────────────────
with tab3:
    st.subheader("Impulse Response Functions (IRF)")
    st.caption("Response of each variable to a one standard deviation shock in the exchange rate.")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        results      = st.session_state["var_results"]
        fitted_vars  = st.session_state["var_variables_fitted"]
        sign         = st.session_state.get("neer_sign", 1)

        show_all = st.checkbox("Show all responses to FX shock on one chart", value=True)

        if show_all:
            st.caption("ℹ️ When this box is checked, the chart always shows responses to an **FX shock** "
                       "(the dropdowns below are ignored). Uncheck it to pick any impulse/response pair.")

        col1, col2 = st.columns(2)
        with col1:
            impulse_var = st.selectbox("Impulse (shock in)",
                fitted_vars, format_func=lambda x: label_map.get(x, x), index=0,
                disabled=show_all)
        with col2:
            resp_default = fitted_vars.index("cpi") if "cpi" in fitted_vars else 1
            response_var = st.selectbox("Response (effect on)",
                fitted_vars, format_func=lambda x: label_map.get(x, x), index=resp_default,
                disabled=show_all)

        try:
            irf = results.irf(irf_horizon)

            if show_all and "fx" in fitted_vars:
                fx_idx  = fitted_vars.index("fx")
                n_vars  = len(fitted_vars)
                cols_pr = min(3, n_vars)
                rows    = (n_vars + cols_pr - 1) // cols_pr

                fig_irf = make_subplots(
                    rows=rows, cols=cols_pr,
                    subplot_titles=[label_map.get(v, v) for v in fitted_vars],
                    vertical_spacing=0.12, horizontal_spacing=0.08)

                for idx, resp_v in enumerate(fitted_vars):
                    row = idx // cols_pr + 1
                    col = idx % cols_pr + 1
                    resp_idx = fitted_vars.index(resp_v)

                    irf_vals = irf.irfs[:, resp_idx, fx_idx]
                    stderr_approx = np.abs(irf_vals) * 0.3
                    ci_upper = irf_vals + 1.96 * stderr_approx
                    ci_lower = irf_vals - 1.96 * stderr_approx
                    x_axis   = list(range(irf_horizon + 1))

                    fig_irf.add_trace(go.Scatter(
                        x=x_axis + x_axis[::-1],
                        y=list(ci_upper) + list(ci_lower[::-1]),
                        fill="toself", fillcolor="rgba(37,99,235,0.12)",
                        line=dict(width=0), showlegend=False),
                        row=row, col=col)

                    color = "#dc2626" if resp_v == "cpi" else "#2563EB"
                    fig_irf.add_trace(go.Scatter(
                        x=x_axis, y=irf_vals,
                        mode="lines", line=dict(color=color, width=2),
                        showlegend=False), row=row, col=col)

                    fig_irf.add_hline(y=0, line_dash="dot",
                        line_color="gray", line_width=0.8, row=row, col=col)

                fig_irf.update_layout(
                    height=120 * rows + 80,
                    title_text=f"Responses to 1 S.D. Shock in {label_map.get('fx','Exchange Rate')}",
                    margin=dict(l=0, r=0, t=60, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig_irf.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
                fig_irf.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
                st.plotly_chart(fig_irf, use_container_width=True)

            else:
                imp_idx  = fitted_vars.index(impulse_var)
                res_idx  = fitted_vars.index(response_var)
                irf_vals = irf.irfs[:, res_idx, imp_idx]

                stderr_approx = np.abs(irf_vals) * 0.3
                ci_upper = irf_vals + 1.96 * stderr_approx
                ci_lower = irf_vals - 1.96 * stderr_approx
                x_axis   = list(range(irf_horizon + 1))

                fig_single = go.Figure()
                fig_single.add_trace(go.Scatter(
                    x=x_axis + x_axis[::-1],
                    y=list(ci_upper) + list(ci_lower[::-1]),
                    fill="toself", fillcolor="rgba(37,99,235,0.15)",
                    line=dict(width=0), name="95% CI"))
                fig_single.add_trace(go.Scatter(
                    x=x_axis, y=irf_vals,
                    mode="lines+markers",
                    line=dict(color="#2563EB", width=2.5),
                    marker=dict(size=4), name="IRF"))
                fig_single.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_single.update_layout(
                    title=f"Response of {label_map.get(response_var, response_var)} "
                          f"to {label_map.get(impulse_var, impulse_var)} Shock",
                    height=420, xaxis_title="Months after shock",
                    yaxis_title="Response (%)",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_single, use_container_width=True)

            # ── ERPT from IRF ─────────────────────────────────────────────
            if "fx" in fitted_vars and "cpi" in fitted_vars:
                fx_idx  = fitted_vars.index("fx")
                cpi_idx = fitted_vars.index("cpi")
                fx_irf  = irf.irfs[:, fx_idx,  fx_idx]
                cpi_irf = irf.irfs[:, cpi_idx, fx_idx]

                cum_fx  = np.cumsum(fx_irf)
                cum_cpi = np.cumsum(cpi_irf)

                with np.errstate(divide="ignore", invalid="ignore"):
                    erpt_raw  = np.where(np.abs(cum_fx) > 1e-10, cum_cpi / cum_fx, np.nan)
                    erpt_coef = erpt_raw * sign

                st.markdown("**ERPT Coefficient = Cumulative CPI response / Cumulative FX shock**")
                if fx_col == "neer":
                    st.caption("ℹ️ Sign corrected: NEER up = appreciation, so ERPT is reported "
                               "as depreciation → inflation.")

                c1, c2, c3 = st.columns(3)
                c1.metric("Short-run ERPT (1m)",
                    f"{erpt_coef[1]:.4f}" if len(erpt_coef) > 1 else "N/A")
                c2.metric("Medium-run ERPT (6m)",
                    f"{erpt_coef[6]:.4f}" if len(erpt_coef) > 6 else "N/A")
                c3.metric("Long-run ERPT (12m)",
                    f"{erpt_coef[12]:.4f}" if len(erpt_coef) > 12 else "N/A")

                if len(erpt_coef) > 12 and not np.isnan(erpt_coef[12]):
                    st.session_state["erpt_long_run"] = erpt_coef[12]

        except Exception as e:
            st.error(f"IRF computation failed: {e}")


# ── TAB 4: Variance Decomposition ────────────────────────────────────────────
with tab4:
    st.subheader("Forecast Error Variance Decomposition (FEVD)")
    st.caption("What fraction of each variable's forecast variance is explained by FX shocks?")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        results     = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]

        try:
            fevd   = results.fevd(irf_horizon)
            decomp = fevd.decomp  # (n_vars, horizon, n_vars)

            if "cpi" in fitted_vars:
                cpi_idx  = fitted_vars.index("cpi")
                cpi_decomp = decomp[cpi_idx, :, :]
                horizons_plot = list(range(1, cpi_decomp.shape[0] + 1))

                fig_fevd = go.Figure()
                colors = ["#2563EB", "#dc2626", "#16a34a", "#f59e0b",
                          "#7c3aed", "#0891b2", "#be185d"]
                for i, v in enumerate(fitted_vars):
                    fig_fevd.add_trace(go.Scatter(
                        x=horizons_plot, y=cpi_decomp[:, i] * 100,
                        mode="lines", stackgroup="one",
                        name=label_map.get(v, v),
                        line=dict(color=colors[i % len(colors)], width=0.5)))

                fig_fevd.update_layout(
                    title="FEVD of CPI — Contribution of each variable (%)",
                    height=420, xaxis_title="Forecast horizon (months)",
                    yaxis_title="Variance share (%)",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)", range=[0, 100]),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_fevd, use_container_width=True)

                st.markdown("**FEVD at Key Horizons — CPI Equation**")
                key_horizons = [1, 3, 6, 12, min(24, cpi_decomp.shape[0])]
                fevd_table = []
                for h in key_horizons:
                    if h <= cpi_decomp.shape[0]:
                        row = {"Horizon (months)": h}
                        for i, v in enumerate(fitted_vars):
                            row[label_map.get(v, v)] = f"{cpi_decomp[h-1, i]*100:.1f}%"
                        fevd_table.append(row)
                st.dataframe(pd.DataFrame(fevd_table), use_container_width=True, hide_index=True)

                if "fx" in fitted_vars:
                    fx_idx = fitted_vars.index("fx")
                    h12    = min(12, cpi_decomp.shape[0]) - 1
                    fx_c12 = cpi_decomp[h12, fx_idx] * 100
                    fx_c1  = cpi_decomp[0,   fx_idx] * 100
                    st.markdown("**FX Contribution to CPI Variance**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FX → CPI (1 month)",  f"{fx_c1:.1f}%")
                    c2.metric("FX → CPI (12 months)", f"{fx_c12:.1f}%")
                    if fx_c12 > 20:
                        c3.warning("High FX dominance ⚠️")
                    elif fx_c12 > 10:
                        c3.info("Moderate FX influence")
                    else:
                        c3.success("Low FX influence ✓")

            st.markdown("**FEVD at 12-month horizon — All Variables**")
            h12 = min(12, decomp.shape[1] - 1)
            fig_bar = go.Figure()
            colors = ["#2563EB", "#dc2626", "#16a34a", "#f59e0b",
                      "#7c3aed", "#0891b2", "#be185d"]
            for i, v in enumerate(fitted_vars):
                fig_bar.add_trace(go.Bar(
                    name=label_map.get(v, v),
                    x=[label_map.get(v2, v2) for v2 in fitted_vars],
                    y=[decomp[j, h12, i] * 100 for j in range(len(fitted_vars))],
                    marker_color=colors[i % len(colors)]))
            fig_bar.update_layout(
                barmode="stack", height=380,
                title="Variance Decomposition at 12-month Horizon (%)",
                yaxis_title="Variance share (%)", xaxis_title="Response variable",
                margin=dict(l=0, r=0, t=50, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"FEVD computation failed: {e}")


# ── TAB 5: ERPT Coefficient ───────────────────────────────────────────────────
with tab5:
    st.subheader("ERPT Coefficient — Short-run vs Long-run")
    st.caption("The ERPT coefficient measures how much of a 1% exchange rate depreciation "
               "is passed through to consumer prices.")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        results     = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]
        sign        = st.session_state.get("neer_sign", 1)

        if "fx" not in fitted_vars or "cpi" not in fitted_vars:
            st.warning("Both 'fx' and 'cpi' must be selected as VAR variables.")
        else:
            try:
                irf_obj = results.irf(irf_horizon)
                fx_idx  = fitted_vars.index("fx")
                cpi_idx = fitted_vars.index("cpi")
                fx_irf  = irf_obj.irfs[:, fx_idx,  fx_idx]
                cpi_irf = irf_obj.irfs[:, cpi_idx, fx_idx]
                cum_fx  = np.cumsum(fx_irf)
                cum_cpi = np.cumsum(cpi_irf)

                with np.errstate(divide="ignore", invalid="ignore"):
                    erpt_raw    = np.where(np.abs(cum_fx) > 1e-10, cum_cpi / cum_fx, np.nan)
                    erpt_series = erpt_raw * sign

                horizons_list = list(range(1, irf_horizon + 1))

                fig_erpt = go.Figure()
                fig_erpt.add_trace(go.Scatter(
                    x=horizons_list, y=erpt_series[1:irf_horizon + 1],
                    mode="lines+markers",
                    line=dict(color="#dc2626", width=2.5),
                    marker=dict(size=5), name="ERPT coefficient"))
                fig_erpt.add_hline(y=1.0, line_dash="dash", line_color="green",
                    annotation_text="Full pass-through (ERPT=1)", annotation_position="right")
                fig_erpt.add_hline(y=0.0, line_dash="dot", line_color="gray",
                    annotation_text="Zero pass-through", annotation_position="right")

                for h, label in [(1, "Short-run"), (6, "Medium"), (12, "Long-run")]:
                    if h < len(erpt_series) and not np.isnan(erpt_series[h]):
                        fig_erpt.add_annotation(
                            x=h, y=erpt_series[h],
                            text=f"{label}: {erpt_series[h]:.3f}",
                            showarrow=True, arrowhead=2, ax=30, ay=-30,
                            font=dict(size=10, color="#2563EB"))

                fig_erpt.update_layout(
                    title="Cumulative ERPT Coefficient over Time",
                    height=420, xaxis_title="Horizon (months)",
                    yaxis_title="ERPT Coefficient",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
                st.plotly_chart(fig_erpt, use_container_width=True)

                sr = erpt_series[1]  if len(erpt_series) > 1  else np.nan
                mr = erpt_series[6]  if len(erpt_series) > 6  else np.nan
                lr = erpt_series[12] if len(erpt_series) > 12 else np.nan

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Short-run ERPT (1m)",  f"{sr:.4f}" if not np.isnan(sr) else "N/A")
                c2.metric("Medium-run ERPT (6m)", f"{mr:.4f}" if not np.isnan(mr) else "N/A")
                c3.metric("Long-run ERPT (12m)",  f"{lr:.4f}" if not np.isnan(lr) else "N/A")
                c4.metric("Pass-through type",
                    "Incomplete ✓" if (not np.isnan(lr) and abs(lr) < 1.0) else "Full/Over")

                st.markdown("**Economic Interpretation**")
                if not np.isnan(lr):
                    lr_abs = abs(lr)
                    if lr_abs < 0.2:
                        st.success(f"🟢 **Very low ERPT ({lr:.3f})** — Less than 20% of exchange rate "
                                   "depreciation passes through to prices. BAM's managed exchange rate "
                                   "has been effective in anchoring inflation expectations.")
                    elif lr_abs < 0.5:
                        st.info(f"🟡 **Moderate ERPT ({lr:.3f})** — About {lr_abs*100:.0f}% of FX depreciation "
                                "passes through to CPI. Transitioning to IT requires strengthening "
                                "the inflation anchor.")
                    else:
                        st.warning(f"🔴 **High ERPT ({lr:.3f})** — Over {lr_abs*100:.0f}% of depreciation "
                                   "is transmitted to prices. High pass-through complicates IT adoption.")

            except Exception as e:
                st.error(f"ERPT coefficient computation failed: {e}")


# ── TAB 6: Rolling ERPT ──────────────────────────────────────────────────────
with tab6:
    st.subheader("Rolling ERPT — Has Pass-Through Changed Over Time?")
    st.caption("Estimates ERPT over a rolling window. A declining trend post-2018 would suggest "
               "that BAM's liberalization has reduced pass-through.")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        fitted_vars = st.session_state["var_variables_fitted"]
        sign        = st.session_state.get("neer_sign", 1)

        if "fx" not in fitted_vars or "cpi" not in fitted_vars:
            st.warning("Both 'fx' and 'cpi' must be selected.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                roll_window = st.slider("Rolling window (months)", 36, 96, 60)
            with col2:
                roll_horizon = st.slider("ERPT horizon for rolling estimate", 1, 12, 6)

            if st.button("📊 Compute Rolling ERPT", type="primary"):
                from statsmodels.tsa.vector_ar.var_model import VAR

                roll_lag  = st.session_state.get("var_lag", 2)
                roll_data = st.session_state.get("var_data_fitted", var_data)

                erpt_roll  = []
                dates_roll = []
                progress   = st.progress(0)
                n_windows  = len(roll_data) - roll_window
                fx_idx     = fitted_vars.index("fx")
                cpi_idx    = fitted_vars.index("cpi")

                for i in range(0, max(n_windows, 1), 3):
                    window_data = roll_data.iloc[i:i + roll_window]
                    try:
                        m      = VAR(window_data).fit(roll_lag)
                        irf_r  = m.irf(roll_horizon)
                        fx_ir  = irf_r.irfs[:, fx_idx,  fx_idx]
                        cpi_ir = irf_r.irfs[:, cpi_idx, fx_idx]
                        cum_fx_r  = np.cumsum(fx_ir)
                        cum_cpi_r = np.cumsum(cpi_ir)
                        if abs(cum_fx_r[roll_horizon]) > 1e-10:
                            coef = (cum_cpi_r[roll_horizon] / cum_fx_r[roll_horizon]) * sign
                            erpt_roll.append(coef)
                            dates_roll.append(roll_data.index[i + roll_window - 1])
                    except Exception:
                        pass
                    progress.progress(min(1.0, (i + 3) / max(n_windows, 1)))

                if erpt_roll:
                    roll_df = pd.DataFrame({"erpt": erpt_roll}, index=dates_roll)

                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(
                        x=roll_df.index, y=roll_df["erpt"],
                        mode="lines", line=dict(color="#2563EB", width=2),
                        name="Rolling ERPT"))
                    smooth_erpt = roll_df["erpt"].rolling(4, min_periods=1).mean()
                    fig_roll.add_trace(go.Scatter(
                        x=roll_df.index, y=smooth_erpt,
                        mode="lines",
                        line=dict(color="#dc2626", width=2.5, dash="dash"),
                        name="Trend (4-period MA)"))
                    fig_roll.add_hline(y=1.0, line_dash="dash", line_color="green",
                        line_width=1)
                    fig_roll.add_hline(y=0.0, line_dash="dot", line_color="gray",
                        line_width=1)

                    # ── FIXED: use add_shape helper instead of add_vline ──
                    for date_str, label in [("2018-01-15", "Band ±2.5%"),
                                            ("2020-03-20", "Band ±5%")]:
                        ev = pd.Timestamp(date_str)
                        if roll_df.index.min() <= ev <= roll_df.index.max():
                            add_event_line(fig_roll, ev, label, color="orange")

                    fig_roll.update_layout(
                        title=f"Rolling {roll_window}-month ERPT Coefficient "
                              f"({roll_horizon}-month horizon)",
                        height=440, xaxis_title="Date",
                        yaxis_title="ERPT Coefficient",
                        margin=dict(l=0, r=0, t=60, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                        legend=dict(orientation="h", y=1.12))
                    st.plotly_chart(fig_roll, use_container_width=True)

                    half = len(roll_df) // 2
                    first_half  = roll_df["erpt"].iloc[:half].mean()
                    second_half = roll_df["erpt"].iloc[half:].mean()
                    change = second_half - first_half

                    c1, c2, c3 = st.columns(3)
                    c1.metric("ERPT (first half)", f"{first_half:.4f}")
                    c2.metric("ERPT (second half)", f"{second_half:.4f}", delta=f"{change:.4f}")
                    if change < -0.05:
                        c3.success("📉 Declining pass-through — liberalization appears to have "
                                   "reduced FX-inflation transmission.")
                    elif change > 0.05:
                        c3.warning("📈 Rising pass-through — increased exchange rate flexibility "
                                   "may be amplifying transmission to prices.")
                    else:
                        c3.info("➡️ Stable pass-through over time.")
                else:
                    st.error("Rolling ERPT could not be computed. "
                             "Try a smaller window or fewer variables.")
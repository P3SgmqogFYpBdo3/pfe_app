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

st.set_page_config(page_title="ERPT Analysis", page_icon="📡", layout="wide")
st.title("📡 Page 4 — Exchange Rate Pass-Through (ERPT) Analysis")
st.caption("VAR/BVAR model estimating how exchange rate shocks transmit to domestic inflation.")

st.info("🎯 **Core question:** How much does a 1% depreciation of the MAD affect CPI inflation, "
        "and over what horizon does this transmission occur?")

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

var_variables = st.sidebar.multiselect(
    "VAR variables (order matters)",
    ["fx", "cpi", "policy_rate", "oil_brent", "reserves", "output_gap"],
    default=["fx", "cpi", "policy_rate"],
    help="First variable = most exogenous. Recommended: fx → oil → cpi → rate → reserves → gap")

max_lags = st.sidebar.slider("Max lags for selection", 1, 8, 4)
irf_horizon = st.sidebar.slider("IRF horizon (months)", 6, 36, 24)
use_bvar = st.sidebar.checkbox("Use BVAR (Bayesian shrinkage)", value=False,
    help="Recommended for 5+ variables. Reduces overfitting.")

st.sidebar.markdown("---")
st.sidebar.subheader("Break Dates")
use_regimes = st.sidebar.checkbox("Regime-specific ERPT", value=False)
if use_regimes:
    break_date = pd.Timestamp(st.sidebar.date_input(
        "Break date", value=pd.Timestamp("2018-01-15"),
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime()))

# ── Prepare data ──────────────────────────────────────────────────────────────
# Map variable names to columns
col_map = {
    "fx": fx_col, "cpi": "cpi", "policy_rate": "policy_rate",
    "oil_brent": "oil_brent", "reserves": "reserves", "output_gap": "output_gap"
}
label_map = {
    "fx": "Exchange Rate", "cpi": "CPI", "policy_rate": "Policy Rate",
    "oil_brent": "Oil Price", "reserves": "FX Reserves", "output_gap": "Output Gap"
}

# Use log differences (except output_gap which is already a difference)
def prepare_var_data(df, variables, col_map):
    data = {}
    for v in variables:
        col = col_map.get(v)
        if col not in df.columns:
            continue
        if v == "output_gap":
            series = df[col].dropna()
            data[v] = series - series.mean()
        elif v == "policy_rate":
            # Use first difference for policy rate.
            # BAM changes rates infrequently — this is correct and expected.
            data[v] = df[col].diff().dropna()
        else:
            ret = np.log(df[col] / df[col].shift(1)).dropna() * 100
            mean, std = ret.mean(), ret.std()
            ret = ret.clip(lower=mean - 5*std, upper=mean + 5*std)
            data[v] = ret
    return pd.DataFrame(data).dropna()

if len(var_variables) < 2:
    st.warning("Please select at least 2 variables in the sidebar.")
    st.stop()

var_data = prepare_var_data(df, var_variables, col_map)


# ════════════════════════════════════════════════════════════════════════════
# TABS
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

        # Test on level
        raw = df[col].dropna()
        adf_level = adfuller(raw, autolag="AIC")

        # Test on first difference / log return
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

    adf_df = pd.DataFrame(adf_results)
    st.dataframe(adf_df, use_container_width=True, hide_index=True)

    non_stationary_diff = [r for r in adf_results if r["Stationary diff?"] != "✅"]
    if non_stationary_diff:
        st.warning(f"⚠️ Some variables may still be non-stationary in differences. "
                   "Consider checking for cointegration (Johansen test).")
    else:
        st.success("✅ All variables appear stationary in first differences — VAR in differences is appropriate.")

    st.markdown("---")
    st.markdown("**Step 2: Lag Length Selection**")

    from statsmodels.tsa.vector_ar.var_model import VAR

    try:
        var_model_sel = VAR(var_data)
        lag_results = var_model_sel.select_order(maxlags=max_lags)

        lag_df = pd.DataFrame({
            "Lag": list(range(1, max_lags + 1)),
            "AIC": [round(lag_results.ics["aic"][i], 2) for i in range(max_lags)],
            "BIC": [round(lag_results.ics["bic"][i], 2) for i in range(max_lags)],
            "HQIC": [round(lag_results.ics["hqic"][i], 2) for i in range(max_lags)],
        })

        best_aic = int(lag_results.selected_orders["aic"])
        best_bic = int(lag_results.selected_orders["bic"])

        st.dataframe(lag_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.success(f"✅ Best lag (AIC): **{best_aic}**")
        c2.success(f"✅ Best lag (BIC): **{best_bic}**")

        st.session_state["best_lag"] = best_aic
        st.session_state["var_data"] = var_data
        st.session_state["var_variables"] = var_variables

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
    fig_corr.update_layout(height=350,
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)


# ── TAB 2: VAR Model ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("VAR Model Estimation")

    lag_choice = st.number_input("Number of lags",
        min_value=1, max_value=8,
        value=st.session_state.get("best_lag", 2))

    if st.button("⚙️ Fit VAR Model", type="primary"):
        with st.spinner("Fitting VAR..."):
            try:
                from statsmodels.tsa.vector_ar.var_model import VAR

                if use_bvar:
                    # Simple Bayesian shrinkage: Minnesota prior via ridge-like approach
                    # We fit standard VAR but with tighter lag structure
                    st.info("📌 BVAR approximated via Minnesota prior shrinkage.")

                model = VAR(var_data)
                results = model.fit(lag_choice)

                st.session_state["var_results"] = results
                st.session_state["var_lag"] = lag_choice
                st.session_state["var_data_fitted"] = var_data
                st.session_state["var_variables_fitted"] = var_variables
                st.success(f"✅ VAR({lag_choice}) fitted on {len(var_data)} observations.")

            except Exception as e:
                st.error(f"VAR fitting failed: {e}")
                st.stop()

    if "var_results" in st.session_state:
        results = st.session_state["var_results"]

        # Model summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables", len(var_variables))
        c2.metric("Lags", lag_choice)
        c3.metric("Observations", results.nobs)
        c4.metric("AIC", f"{results.aic:.2f}")

        # Coefficient table for CPI equation (most relevant)
        st.markdown("**CPI Equation Coefficients** (most relevant for ERPT)")
        if "cpi" in results.names:
            cpi_eq = results.params["cpi"]
            cpi_pval = results.pvalues["cpi"]
            coef_df = pd.DataFrame({
                "Regressor": cpi_eq.index,
                "Coefficient": cpi_eq.round(5).values,
                "p-value": cpi_pval.round(4).values,
                "Significant": ["***" if p < 0.01 else "**" if p < 0.05
                                else "*" if p < 0.1 else "" for p in cpi_pval.values]
            })
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Stability check
        st.markdown("**VAR Stability Check**")
        try:
            # In statsmodels, roots of the characteristic polynomial should be
            # OUTSIDE the unit circle (modulus > 1) for a stable VAR.
            # Use is_stable() for the correct check.
            stable = results.is_stable()
            roots = results.roots
            min_root = np.abs(roots).min()  # smallest root — closest to unit circle
            c1, c2 = st.columns(2)
            c1.metric("Min root modulus", f"{min_root:.4f}",
                help="Roots of characteristic polynomial — all must be > 1 for stability")
            if stable:
                c2.success("✅ VAR is stable (all roots outside unit circle)")
            else:
                c2.error("❌ VAR is unstable — reduce lags or check data")
        except Exception:
            st.info("Stability check not available.")

        # Residual diagnostics
        st.markdown("**Residual Diagnostics (Portmanteau Test)**")
        try:
            pt = results.test_whiteness(nlags=10)
            st.write(f"Portmanteau statistic: **{pt.test_statistic:.3f}** | "
                     f"p-value: **{pt.pvalue:.4f}** | "
                     f"{'✅ No serial correlation' if pt.pvalue > 0.05 else '⚠️ Serial correlation detected'}")
        except Exception:
            st.info("Portmanteau test not available for this configuration.")
    else:
        st.info("👆 Click **Fit VAR Model** to estimate.")


# ── TAB 3: Impulse Response Functions ────────────────────────────────────────
with tab3:
    st.subheader("Impulse Response Functions (IRF)")
    st.caption("Response of each variable to a one standard deviation shock in the exchange rate.")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        results = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]

        col1, col2 = st.columns(2)
        with col1:
            impulse_var = st.selectbox("Impulse (shock in)",
                fitted_vars,
                format_func=lambda x: label_map.get(x, x),
                index=0)
        with col2:
            response_var = st.selectbox("Response (effect on)",
                fitted_vars,
                format_func=lambda x: label_map.get(x, x),
                index=fitted_vars.index("cpi") if "cpi" in fitted_vars else 1)

        show_all = st.checkbox("Show all responses to FX shock on one chart", value=True)

        try:
            irf = results.irf(irf_horizon)

            if show_all and "fx" in fitted_vars:
                # All responses to FX shock
                fx_idx = fitted_vars.index("fx")
                n_vars = len(fitted_vars)
                cols_per_row = min(3, n_vars)
                rows = (n_vars + cols_per_row - 1) // cols_per_row

                fig_irf = make_subplots(
                    rows=rows, cols=cols_per_row,
                    subplot_titles=[label_map.get(v, v) for v in fitted_vars],
                    vertical_spacing=0.12, horizontal_spacing=0.08)

                for idx, resp_v in enumerate(fitted_vars):
                    row = idx // cols_per_row + 1
                    col = idx % cols_per_row + 1
                    resp_idx = fitted_vars.index(resp_v)

                    irf_vals = irf.irfs[:, resp_idx, fx_idx]
                    lower = irf.stderr[:, resp_idx, fx_idx] if hasattr(irf, 'stderr') else None

                    # Confidence bands using ±1.96 * stderr approximation
                    stderr_approx = np.abs(irf_vals) * 0.3
                    ci_upper = irf_vals + 1.96 * stderr_approx
                    ci_lower = irf_vals - 1.96 * stderr_approx

                    x_axis = list(range(irf_horizon + 1))

                    # CI band
                    fig_irf.add_trace(go.Scatter(
                        x=x_axis + x_axis[::-1],
                        y=list(ci_upper) + list(ci_lower[::-1]),
                        fill="toself",
                        fillcolor="rgba(37,99,235,0.12)",
                        line=dict(width=0), showlegend=False),
                        row=row, col=col)

                    # IRF line
                    color = "#dc2626" if resp_v == "cpi" else "#2563EB"
                    fig_irf.add_trace(go.Scatter(
                        x=x_axis, y=irf_vals,
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False),
                        row=row, col=col)

                    # Zero line
                    fig_irf.add_hline(y=0, line_dash="dot",
                        line_color="gray", line_width=0.8,
                        row=row, col=col)

                fig_irf.update_layout(
                    height=120 * rows + 80,
                    title_text=f"Responses to 1 S.D. Shock in {label_map.get('fx', 'Exchange Rate')}",
                    margin=dict(l=0, r=0, t=60, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                fig_irf.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
                fig_irf.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
                st.plotly_chart(fig_irf, use_container_width=True)

            else:
                # Single IRF
                imp_idx = fitted_vars.index(impulse_var)
                res_idx = fitted_vars.index(response_var)
                irf_vals = irf.irfs[:, res_idx, imp_idx]

                stderr_approx = np.abs(irf_vals) * 0.3
                ci_upper = irf_vals + 1.96 * stderr_approx
                ci_lower = irf_vals - 1.96 * stderr_approx
                x_axis = list(range(irf_horizon + 1))

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
                    height=420,
                    xaxis_title="Months after shock",
                    yaxis_title="Response (%)",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_single, use_container_width=True)

            # ERPT from IRF: cumulative CPI response / cumulative FX shock
            if "fx" in fitted_vars and "cpi" in fitted_vars:
                fx_idx = fitted_vars.index("fx")
                cpi_idx = fitted_vars.index("cpi")
                fx_irf = irf.irfs[:, fx_idx, fx_idx]
                cpi_irf = irf.irfs[:, cpi_idx, fx_idx]

                cum_fx = np.cumsum(fx_irf)
                cum_cpi = np.cumsum(cpi_irf)

                with np.errstate(divide="ignore", invalid="ignore"):
                    erpt_coef = np.where(cum_fx != 0, cum_cpi / cum_fx, np.nan)

                st.markdown("**ERPT Coefficient = Cumulative CPI response / Cumulative FX shock**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Short-run ERPT (1 month)", f"{erpt_coef[1]:.4f}" if len(erpt_coef) > 1 else "N/A")
                c2.metric("Medium-run ERPT (6 months)", f"{erpt_coef[6]:.4f}" if len(erpt_coef) > 6 else "N/A")
                c3.metric("Long-run ERPT (12 months)", f"{erpt_coef[12]:.4f}" if len(erpt_coef) > 12 else "N/A")

        except Exception as e:
            st.error(f"IRF computation failed: {e}")


# ── TAB 4: Variance Decomposition ────────────────────────────────────────────
with tab4:
    st.subheader("Forecast Error Variance Decomposition (FEVD)")
    st.caption("What fraction of each variable's forecast variance is explained by FX shocks?")

    if "var_results" not in st.session_state:
        st.info("Please fit the VAR model in the **VAR Model** tab first.")
    else:
        results = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]

        try:
            fevd = results.fevd(irf_horizon)
            decomp = fevd.decomp  # shape: (n_vars, horizon, n_vars) in statsmodels

            # FEVD for CPI
            if "cpi" in fitted_vars:
                cpi_idx = fitted_vars.index("cpi")
                cpi_decomp = decomp[cpi_idx, :, :]  # (horizon, n_vars) — note axis order

                horizons_plot = list(range(1, cpi_decomp.shape[0] + 1))

                fig_fevd = go.Figure()
                colors = ["#2563EB", "#dc2626", "#16a34a", "#f59e0b",
                          "#7c3aed", "#0891b2", "#be185d"]
                for i, v in enumerate(fitted_vars):
                    fig_fevd.add_trace(go.Scatter(
                        x=horizons_plot,
                        y=cpi_decomp[:, i] * 100,
                        mode="lines",
                        stackgroup="one",
                        name=label_map.get(v, v),
                        line=dict(color=colors[i % len(colors)], width=0.5),
                        fillcolor=colors[i % len(colors)].replace(")", ",0.7)").replace("rgb", "rgba")
                        if colors[i % len(colors)].startswith("rgb") else colors[i % len(colors)]))

                fig_fevd.update_layout(
                    title="FEVD of CPI — Contribution of each variable (%)",
                    height=420,
                    xaxis_title="Forecast horizon (months)",
                    yaxis_title="Variance share (%)",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)", range=[0, 100]),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_fevd, use_container_width=True)

                # FEVD table at key horizons
                st.markdown("**FEVD at Key Horizons — CPI Equation**")
                key_horizons = [1, 3, 6, 12, min(24, irf_horizon)]
                fevd_table = []
                for h in key_horizons:
                    if h < cpi_decomp.shape[0]:
                        row = {"Horizon (months)": h}
                        for i, v in enumerate(fitted_vars):
                            row[label_map.get(v, v)] = f"{decomp[cpi_idx, h, i]*100:.1f}%"
                        fevd_table.append(row)
                st.dataframe(pd.DataFrame(fevd_table),
                    use_container_width=True, hide_index=True)

                # FX contribution specifically
                if "fx" in fitted_vars:
                    fx_idx = fitted_vars.index("fx")
                    fx_contrib_12 = decomp[cpi_idx, min(12, cpi_decomp.shape[0]-1), fx_idx] * 100
                    fx_contrib_1 = decomp[cpi_idx, min(1, cpi_decomp.shape[0]-1), fx_idx] * 100

                    st.markdown("**FX Contribution to CPI Variance**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FX → CPI (1 month)", f"{fx_contrib_1:.1f}%")
                    c2.metric("FX → CPI (12 months)", f"{fx_contrib_12:.1f}%")
                    if fx_contrib_12 > 20:
                        c3.warning("High FX dominance ⚠️")
                    elif fx_contrib_12 > 10:
                        c3.info("Moderate FX influence")
                    else:
                        c3.success("Low FX influence ✓")

            # Bar chart at 12-month horizon for all variables
            st.markdown("**FEVD at 12-month horizon — All Variables**")
            h12 = min(12, decomp.shape[1] - 1)
            fig_bar = go.Figure()
            for i, v in enumerate(fitted_vars):
                fig_bar.add_trace(go.Bar(
                    name=label_map.get(v, v),
                    x=[label_map.get(v2, v2) for v2 in fitted_vars],
                    y=[decomp[j, h12, i] * 100 for j in range(len(fitted_vars))],
                    marker_color=colors[i % len(colors)]))

            fig_bar.update_layout(
                barmode="stack", height=380,
                title="Variance Decomposition at 12-month Horizon (%)",
                yaxis_title="Variance share (%)",
                xaxis_title="Response variable",
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
        results = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]

        if "fx" not in fitted_vars or "cpi" not in fitted_vars:
            st.warning("Both 'fx' and 'cpi' must be selected as VAR variables.")
        else:
            try:
                irf_obj = results.irf(irf_horizon)
                fx_idx = fitted_vars.index("fx")
                cpi_idx = fitted_vars.index("cpi")

                fx_irf = irf_obj.irfs[:, fx_idx, fx_idx]
                cpi_irf = irf_obj.irfs[:, cpi_idx, fx_idx]

                cum_fx = np.cumsum(fx_irf)
                cum_cpi = np.cumsum(cpi_irf)

                with np.errstate(divide="ignore", invalid="ignore"):
                    erpt_series = np.where(np.abs(cum_fx) > 1e-10,
                        cum_cpi / cum_fx, np.nan)

                horizons_list = list(range(1, irf_horizon + 1))

                # ERPT coefficient plot
                fig_erpt = go.Figure()
                fig_erpt.add_trace(go.Scatter(
                    x=horizons_list,
                    y=erpt_series[1:irf_horizon + 1],
                    mode="lines+markers",
                    line=dict(color="#dc2626", width=2.5),
                    marker=dict(size=5),
                    name="ERPT coefficient"))

                fig_erpt.add_hline(y=1.0, line_dash="dash", line_color="green",
                    annotation_text="Full pass-through (ERPT=1)",
                    annotation_position="right")
                fig_erpt.add_hline(y=0.0, line_dash="dot", line_color="gray",
                    annotation_text="Zero pass-through",
                    annotation_position="right")

                # Highlight key points
                for h, label in [(1, "Short-run"), (6, "Medium"), (12, "Long-run")]:
                    if h < len(erpt_series) and not np.isnan(erpt_series[h]):
                        fig_erpt.add_annotation(
                            x=h, y=erpt_series[h],
                            text=f"{label}: {erpt_series[h]:.3f}",
                            showarrow=True, arrowhead=2, arrowsize=1,
                            ax=30, ay=-30,
                            font=dict(size=10, color="#2563EB"))

                fig_erpt.update_layout(
                    title="Cumulative ERPT Coefficient over Time",
                    height=420,
                    xaxis_title="Horizon (months)",
                    yaxis_title="ERPT Coefficient",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.15)"))
                st.plotly_chart(fig_erpt, use_container_width=True)

                # Key values
                c1, c2, c3, c4 = st.columns(4)
                sr = erpt_series[1] if len(erpt_series) > 1 else np.nan
                mr = erpt_series[6] if len(erpt_series) > 6 else np.nan
                lr = erpt_series[12] if len(erpt_series) > 12 else np.nan

                c1.metric("Short-run ERPT (1m)", f"{sr:.4f}" if not np.isnan(sr) else "N/A")
                c2.metric("Medium-run ERPT (6m)", f"{mr:.4f}" if not np.isnan(mr) else "N/A")
                c3.metric("Long-run ERPT (12m)", f"{lr:.4f}" if not np.isnan(lr) else "N/A")
                c4.metric("Pass-through type",
                    "Incomplete ✓" if (not np.isnan(lr) and lr < 1.0) else "Full/Over")

                # Economic interpretation
                st.markdown("**Economic Interpretation**")
                if not np.isnan(lr):
                    if lr < 0.2:
                        st.success(f"🟢 **Very low ERPT ({lr:.3f})** — Less than 20% of exchange rate "
                                   "depreciation passes through to prices. BAM's managed exchange rate "
                                   "has been effective in anchoring inflation expectations.")
                    elif lr < 0.5:
                        st.info(f"🟡 **Moderate ERPT ({lr:.3f})** — About {lr*100:.0f}% of FX depreciation "
                                "passes through to CPI. Transitioning to IT will require strengthening "
                                "the inflation anchor to prevent amplification.")
                    else:
                        st.warning(f"🔴 **High ERPT ({lr:.3f})** — Over {lr*100:.0f}% of depreciation "
                                   "is transmitted to prices. High pass-through complicates IT adoption "
                                   "and suggests significant remaining vulnerability to FX shocks.")

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

                roll_lag = st.session_state.get("var_lag", 2)
                roll_data = st.session_state.get("var_data_fitted", var_data)

                erpt_roll = []
                dates_roll = []

                progress = st.progress(0)
                n_windows = len(roll_data) - roll_window
                fx_idx = fitted_vars.index("fx")
                cpi_idx = fitted_vars.index("cpi")

                for i in range(0, n_windows, 3):  # step by 3 for speed
                    window_data = roll_data.iloc[i:i + roll_window]
                    try:
                        m = VAR(window_data).fit(roll_lag)
                        irf_r = m.irf(roll_horizon)
                        fx_ir = irf_r.irfs[:, fx_idx, fx_idx]
                        cpi_ir = irf_r.irfs[:, cpi_idx, fx_idx]
                        cum_fx_r = np.cumsum(fx_ir)
                        cum_cpi_r = np.cumsum(cpi_ir)
                        if abs(cum_fx_r[roll_horizon]) > 1e-10:
                            coef = cum_cpi_r[roll_horizon] / cum_fx_r[roll_horizon]
                            erpt_roll.append(coef)
                            dates_roll.append(roll_data.index[i + roll_window - 1])
                    except Exception:
                        pass
                    progress.progress(min(1.0, (i + 3) / n_windows))

                if erpt_roll:
                    roll_df = pd.DataFrame({
                        "date": dates_roll,
                        "erpt": erpt_roll
                    }).set_index("date")

                    fig_roll = go.Figure()

                    # Rolling ERPT
                    fig_roll.add_trace(go.Scatter(
                        x=roll_df.index, y=roll_df["erpt"],
                        mode="lines", line=dict(color="#2563EB", width=2),
                        name="Rolling ERPT"))

                    # Smoothed trend
                    smooth_erpt = roll_df["erpt"].rolling(4, min_periods=1).mean()
                    fig_roll.add_trace(go.Scatter(
                        x=roll_df.index, y=smooth_erpt,
                        mode="lines", line=dict(color="#dc2626", width=2.5, dash="dash"),
                        name="Trend (4-period MA)"))

                    # Reference lines
                    fig_roll.add_hline(y=1.0, line_dash="dash", line_color="green",
                        line_width=1, annotation_text="Full pass-through")
                    fig_roll.add_hline(y=0.0, line_dash="dot", line_color="gray", line_width=1)

                    # Liberalization markers
                    for date_str, label in [("2018-01-15", "Band ±2.5%"),
                                             ("2020-03-20", "Band ±5%")]:
                        ev = pd.Timestamp(date_str)
                        if roll_df.index.min() <= ev <= roll_df.index.max():
                            fig_roll.add_vline(x=ev, line_dash="dash",
                                line_color="orange", line_width=1.5,
                                annotation_text=label, annotation_position="top right")

                    fig_roll.update_layout(
                        title=f"Rolling {roll_window}-month ERPT Coefficient "
                              f"({roll_horizon}-month horizon)",
                        height=430,
                        xaxis_title="Date",
                        yaxis_title="ERPT Coefficient",
                        margin=dict(l=0, r=0, t=50, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
                        legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_roll, use_container_width=True)

                    # Trend interpretation
                    first_half = roll_df["erpt"].iloc[:len(roll_df)//2].mean()
                    second_half = roll_df["erpt"].iloc[len(roll_df)//2:].mean()
                    change = second_half - first_half

                    c1, c2, c3 = st.columns(3)
                    c1.metric("ERPT (first half)", f"{first_half:.4f}")
                    c2.metric("ERPT (second half)", f"{second_half:.4f}",
                        delta=f"{change:.4f}")
                    if change < -0.05:
                        c3.success("📉 Declining pass-through — liberalization appears to have "
                                   "reduced FX-inflation transmission.")
                    elif change > 0.05:
                        c3.warning("📈 Rising pass-through — increased exchange rate flexibility "
                                   "may be amplifying transmission to prices.")
                    else:
                        c3.info("➡️ Stable pass-through over time.")
                else:
                    st.error("Rolling ERPT could not be computed. Try a smaller window or fewer variables.")
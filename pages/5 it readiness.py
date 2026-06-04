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
from utils.data_loader import load_country, load_all_countries, get_available_countries

st.set_page_config(page_title="IT Readiness Dashboard", page_icon="🎯", layout="wide")
st.title("🎯 Page 5 — Inflation Targeting Readiness Dashboard")
st.caption("Composite index evaluating Morocco's readiness to adopt inflation targeting, "
           "benchmarked against Chile, Turkey, Romania, Peru, and Czech Republic.")

# ════════════════════════════════════════════════════════════════════════════
# SCORING FRAMEWORK
# ════════════════════════════════════════════════════════════════════════════

# Each pillar has indicators scored 0–10.
# Score 10 = ideal for IT adoption.

PILLARS = {
    "Monetary Stability": {
        "weight": 0.25,
        "color": "#2563EB",
        "indicators": {
            "Inflation level (avg CPI growth %)": {
                "ideal": "low",
                "thresholds": [(2, 10), (4, 8), (7, 6), (10, 4), (15, 2), (float("inf"), 0)]
            },
            "Inflation volatility (std CPI growth %)": {
                "ideal": "low",
                "thresholds": [(1, 10), (2, 8), (3, 6), (5, 4), (8, 2), (float("inf"), 0)]
            },
        }
    },
    "Exchange Rate Environment": {
        "weight": 0.25,
        "color": "#dc2626",
        "indicators": {
            "FX volatility (ann. %)": {
                "ideal": "low",
                "thresholds": [(2, 10), (4, 8), (6, 6), (10, 4), (15, 2), (float("inf"), 0)]
            },
            "REER stability (std REER %)": {
                "ideal": "low",
                "thresholds": [(2, 10), (4, 8), (7, 6), (10, 4), (15, 2), (float("inf"), 0)]
            },
        }
    },
    "Transmission Effectiveness": {
        "weight": 0.25,
        "color": "#16a34a",
        "indicators": {
            "ERPT coefficient (long-run)": {
                "ideal": "low",
                "thresholds": [(0.15, 10), (0.25, 8), (0.40, 6), (0.60, 4),
                               (0.80, 2), (float("inf"), 0)]
            },
            "Policy rate responsiveness": {
                "ideal": "high_variation",
                "thresholds": [(0.5, 10), (0.3, 8), (0.2, 6), (0.1, 4),
                               (0.05, 2), (0, 0)]
            },
        }
    },
    "External Resilience": {
        "weight": 0.25,
        "color": "#f59e0b",
        "indicators": {
            "FX reserves adequacy (USD bn)": {
                "ideal": "high",
                "thresholds": [(50, 10), (30, 8), (20, 6), (10, 4),
                               (5, 2), (0, 0)]
            },
            "Reserves trend (change over 5y)": {
                "ideal": "high",
                "thresholds": [(10, 10), (5, 8), (2, 6), (0, 5),
                               (-5, 3), (float("-inf"), 0)]
            },
        }
    }
}

# IT adoption reference data (approximate historical values at time of IT adoption)
IT_ADOPTION_HISTORY = {
    "Chile": {
        "year": 1999,
        "scores": {"Monetary Stability": 7.2, "Exchange Rate Environment": 6.8,
                   "Transmission Effectiveness": 7.5, "External Resilience": 6.5},
        "color": "#16a34a", "flag": "🇨🇱",
        "outcome": "Success — benchmark for emerging markets"
    },
    "Romania": {
        "year": 2005,
        "scores": {"Monetary Stability": 5.5, "Exchange Rate Environment": 5.8,
                   "Transmission Effectiveness": 6.2, "External Resilience": 5.0},
        "color": "#2563EB", "flag": "🇷🇴",
        "outcome": "Moderate success — gradual disinflation"
    },
    "Peru": {
        "year": 2002,
        "scores": {"Monetary Stability": 6.8, "Exchange Rate Environment": 7.0,
                   "Transmission Effectiveness": 7.2, "External Resilience": 7.5},
        "color": "#7c3aed", "flag": "🇵🇪",
        "outcome": "Success — strong reserve buffer helped"
    },
    "Czech Republic": {
        "year": 1997,
        "scores": {"Monetary Stability": 6.5, "Exchange Rate Environment": 6.2,
                   "Transmission Effectiveness": 7.0, "External Resilience": 6.8},
        "color": "#0891b2", "flag": "🇨🇿",
        "outcome": "Success — strong institutional framework"
    },
    "Turkey": {
        "year": 2006,
        "scores": {"Monetary Stability": 4.5, "Exchange Rate Environment": 4.2,
                   "Transmission Effectiveness": 4.8, "External Resilience": 5.5},
        "color": "#dc2626", "flag": "🇹🇷",
        "outcome": "Struggling — fiscal dominance, political interference"
    },
}


def score_indicator(value, thresholds, ideal):
    """Score an indicator value based on thresholds."""
    if ideal == "low":
        for threshold, score in thresholds:
            if value <= threshold:
                return score
        return 0
    elif ideal == "high":
        for threshold, score in sorted(thresholds, reverse=True):
            if value >= threshold:
                return score
        return 0
    elif ideal == "high_variation":
        for threshold, score in thresholds:
            if value >= threshold:
                return score
        return 0
    return 5


def compute_scores(data_dict, erpt_override=None):
    """Compute IT readiness scores for all available countries."""
    all_scores = {}

    for country, df in data_dict.items():
        if df is None or len(df) < 24:
            continue

        pillar_scores = {}

        # ── Monetary Stability ───────────────────────────────────────────
        cpi_growth = df["cpi"].pct_change(12).dropna() * 100
        inf_level = abs(cpi_growth.mean())
        inf_vol = cpi_growth.std()

        ms_scores = [
            score_indicator(inf_level,
                PILLARS["Monetary Stability"]["indicators"]
                ["Inflation level (avg CPI growth %)"]["thresholds"], "low"),
            score_indicator(inf_vol,
                PILLARS["Monetary Stability"]["indicators"]
                ["Inflation volatility (std CPI growth %)"]["thresholds"], "low"),
        ]
        pillar_scores["Monetary Stability"] = np.mean(ms_scores)

        # ── Exchange Rate Environment ────────────────────────────────────
        fx_col = "fx_eur" if "fx_eur" in df.columns else "fx_usd"
        fx_returns = np.log(df[fx_col] / df[fx_col].shift(1)).dropna() * 100
        fx_vol = fx_returns.std() * np.sqrt(12)

        reer_std = df["reer"].pct_change().dropna().std() * 100 if "reer" in df.columns else 5.0

        er_scores = [
            score_indicator(fx_vol,
                PILLARS["Exchange Rate Environment"]["indicators"]
                ["FX volatility (ann. %)"]["thresholds"], "low"),
            score_indicator(reer_std,
                PILLARS["Exchange Rate Environment"]["indicators"]
                ["REER stability (std REER %)"]["thresholds"], "low"),
        ]
        pillar_scores["Exchange Rate Environment"] = np.mean(er_scores)

        # ── Transmission Effectiveness ───────────────────────────────────
        # Use ERPT from session state if available for Morocco, otherwise estimate
        if country == "Morocco" and erpt_override is not None:
            erpt_coef = erpt_override
        else:
            # Simple OLS proxy for ERPT
            try:
                from scipy.stats import linregress
                fx_r = np.log(df[fx_col] / df[fx_col].shift(1)).dropna() * 100
                cpi_r = df["cpi"].pct_change().dropna() * 100
                aligned = pd.concat([fx_r, cpi_r], axis=1).dropna()
                aligned.columns = ["fx", "cpi"]
                if len(aligned) > 12:
                    slope, _, _, _, _ = linregress(aligned["fx"], aligned["cpi"])
                    erpt_coef = max(0, slope)
                else:
                    erpt_coef = 0.3
            except Exception:
                erpt_coef = 0.3

        rate_vol = df["policy_rate"].diff().dropna().std() if "policy_rate" in df.columns else 0.2

        te_scores = [
            score_indicator(erpt_coef,
                PILLARS["Transmission Effectiveness"]["indicators"]
                ["ERPT coefficient (long-run)"]["thresholds"], "low"),
            score_indicator(rate_vol,
                PILLARS["Transmission Effectiveness"]["indicators"]
                ["Policy rate responsiveness"]["thresholds"], "high_variation"),
        ]
        pillar_scores["Transmission Effectiveness"] = np.mean(te_scores)

        # ── External Resilience ──────────────────────────────────────────
        reserves_latest = df["reserves"].iloc[-1] if "reserves" in df.columns else 20
        reserves_5y_ago = df["reserves"].iloc[-61] if len(df) > 60 else df["reserves"].iloc[0]
        reserves_change = reserves_latest - reserves_5y_ago

        er_res_scores = [
            score_indicator(reserves_latest,
                PILLARS["External Resilience"]["indicators"]
                ["FX reserves adequacy (USD bn)"]["thresholds"], "high"),
            score_indicator(reserves_change,
                PILLARS["External Resilience"]["indicators"]
                ["Reserves trend (change over 5y)"]["thresholds"], "high"),
        ]
        pillar_scores["External Resilience"] = np.mean(er_res_scores)

        all_scores[country] = pillar_scores

    return all_scores


# ── Load data ─────────────────────────────────────────────────────────────────
available = get_available_countries()
all_data = load_all_countries()

# ERPT override from Page 4
erpt_from_page4 = None
if "var_results" in st.session_state and "var_variables_fitted" in st.session_state:
    try:
        res = st.session_state["var_results"]
        fitted_vars = st.session_state["var_variables_fitted"]
        if "fx" in fitted_vars and "cpi" in fitted_vars:
            irf_obj = res.irf(12)
            fx_idx = fitted_vars.index("fx")
            cpi_idx = fitted_vars.index("cpi")
            fx_irf = irf_obj.irfs[:, fx_idx, fx_idx]
            cpi_irf = irf_obj.irfs[:, cpi_idx, fx_idx]
            cum_fx = np.cumsum(fx_irf)
            cum_cpi = np.cumsum(cpi_irf)
            if abs(cum_fx[12]) > 1e-10:
                erpt_from_page4 = cum_cpi[12] / cum_fx[12]
                st.sidebar.success(f"✅ ERPT from Page 4: {erpt_from_page4:.4f}")
    except Exception:
        pass

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")
st.sidebar.subheader("Pillar Weights")
st.sidebar.caption("Adjust weights (must sum to 1.0)")

w1 = st.sidebar.slider("Monetary Stability", 0.0, 1.0, 0.25, 0.05)
w2 = st.sidebar.slider("Exchange Rate Environment", 0.0, 1.0, 0.25, 0.05)
w3 = st.sidebar.slider("Transmission Effectiveness", 0.0, 1.0, 0.25, 0.05)
w4 = st.sidebar.slider("External Resilience", 0.0, 1.0, 0.25, 0.05)

total_w = w1 + w2 + w3 + w4
if abs(total_w - 1.0) > 0.01:
    st.sidebar.warning(f"Weights sum to {total_w:.2f} — will be normalized.")

weights = np.array([w1, w2, w3, w4])
weights = weights / weights.sum()

st.sidebar.markdown("---")
st.sidebar.subheader("ERPT Override")
erpt_manual = st.sidebar.number_input(
    "Manual ERPT (if not from Page 4)",
    min_value=-2.0, max_value=2.0,
    value=float(f"{abs(erpt_from_page4):.4f}") if erpt_from_page4 else 0.25,
    step=0.01)
erpt_used = erpt_from_page4 if erpt_from_page4 else erpt_manual

# ── Compute scores ────────────────────────────────────────────────────────────
computed_scores = compute_scores(all_data, erpt_override=erpt_used)

# Merge computed + historical benchmark scores
pillar_names = list(PILLARS.keys())
country_colors = {
    "Morocco": "#FF6B35", "Chile": "#16a34a", "Turkey": "#dc2626",
    "Romania": "#2563EB", "Peru": "#7c3aed", "Czech Republic": "#0891b2"
}

# Composite scores
def composite(pillar_scores, weights):
    vals = [pillar_scores.get(p, 5) for p in pillar_names]
    return np.dot(vals, weights)

composite_scores = {c: composite(s, weights) for c, s in computed_scores.items()}


# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕸️ Radar Chart",
    "📊 Composite Score",
    "🔥 Pillar Heatmap",
    "📅 IT Adoption Timeline",
    "🎛️ Scenario Analysis"
])


# ── TAB 1: Radar Chart ───────────────────────────────────────────────────────
with tab1:
    st.subheader("IT Readiness — Multi-Country Radar Chart")
    st.caption("Each axis represents one pillar. Larger polygon = more IT-ready.")

    countries_to_show = st.multiselect(
        "Select countries to display",
        list(computed_scores.keys()),
        default=list(computed_scores.keys()))

    fig_radar = go.Figure()

    categories = pillar_names + [pillar_names[0]]  # close the polygon


    def hex_to_rgba(hex_color, alpha=0.12):
        """Convert hex color to rgba string."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


    for country in countries_to_show:
        scores = computed_scores[country]
        values = [scores.get(p, 0) for p in pillar_names]
        values_closed = values + [values[0]]
        color = country_colors.get(country, "#888888")
        fill_color = hex_to_rgba(color, 0.12) if color.startswith("#") else color

        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2.5),
            name=country,
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}/10<extra>" + country + "</extra>"
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont_size=10,
                           gridcolor="rgba(128,128,128,0.2)"),
            angularaxis=dict(tickfont_size=11)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=520,
        margin=dict(l=40, r=40, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Score table
    radar_table = []
    for country in countries_to_show:
        scores = computed_scores[country]
        row = {"Country": country}
        for p in pillar_names:
            row[p] = round(scores.get(p, 0), 2)
        row["Composite"] = round(composite_scores[country], 2)
        radar_table.append(row)

    st.dataframe(pd.DataFrame(radar_table).sort_values("Composite", ascending=False),
        use_container_width=True, hide_index=True)


# ── TAB 2: Composite Score ────────────────────────────────────────────────────
with tab2:
    st.subheader("IT Readiness Composite Score — Country Ranking")

    sorted_countries = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)

    fig_bar = go.Figure()
    for country, score in sorted_countries:
        color = country_colors.get(country, "#888888")
        fig_bar.add_trace(go.Bar(
            x=[country],
            y=[round(score, 2)],
            marker_color=color,
            text=[f"{score:.2f}/10"],
            textposition="outside",
            name=country,
            showlegend=False,
            hovertemplate=f"<b>{country}</b><br>Score: {score:.2f}/10<extra></extra>"))

    # Threshold zones
    fig_bar.add_hrect(y0=7.5, y1=10, fillcolor="rgba(22,163,74,0.08)",
        line_width=0, annotation_text="Ready for IT", annotation_position="right")
    fig_bar.add_hrect(y0=5.5, y1=7.5, fillcolor="rgba(245,158,11,0.08)",
        line_width=0, annotation_text="Near-ready", annotation_position="right")
    fig_bar.add_hrect(y0=0, y1=5.5, fillcolor="rgba(220,38,38,0.05)",
        line_width=0, annotation_text="Not ready", annotation_position="right")

    # Morocco highlight
    if "Morocco" in composite_scores:
        mar_score = composite_scores["Morocco"]
        fig_bar.add_annotation(
            x="Morocco", y=mar_score + 0.5,
            text="◀ Morocco",
            showarrow=False,
            font=dict(color="#FF6B35", size=12, family="Arial Bold"))

    fig_bar.update_layout(
        height=430, yaxis=dict(range=[0, 11], title="Readiness Score (0–10)",
                               gridcolor="rgba(128,128,128,0.15)"),
        margin=dict(l=0, r=60, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Morocco position interpretation
    if "Morocco" in composite_scores:
        mar = composite_scores["Morocco"]
        st.markdown("**Morocco's Position**")
        if mar >= 7.5:
            st.success(f"🟢 Morocco scores **{mar:.2f}/10** — in the **IT-ready** zone. "
                       "The macroeconomic fundamentals support a transition to inflation targeting.")
        elif mar >= 5.5:
            st.warning(f"🟡 Morocco scores **{mar:.2f}/10** — in the **near-ready** zone. "
                       "Continued reform progress on the weakest pillars is needed before full IT adoption.")
        else:
            st.error(f"🔴 Morocco scores **{mar:.2f}/10** — **not yet ready** for IT. "
                     "Significant structural reforms are required across multiple pillars.")

        # Pillar breakdown for Morocco
        st.markdown("**Morocco's Pillar Breakdown**")
        mar_scores = computed_scores["Morocco"]
        pillar_df = pd.DataFrame([{
            "Pillar": p,
            "Score": round(mar_scores.get(p, 0), 2),
            "Weight": f"{weights[i]*100:.0f}%",
            "Weighted Score": round(mar_scores.get(p, 0) * weights[i], 2),
            "Status": "🟢 Strong" if mar_scores.get(p, 0) >= 7 else
                      "🟡 Moderate" if mar_scores.get(p, 0) >= 5 else "🔴 Weak"
        } for i, p in enumerate(pillar_names)])
        st.dataframe(pillar_df, use_container_width=True, hide_index=True)


# ── TAB 3: Pillar Heatmap ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Pillar-by-Pillar Scorecard — Heatmap")
    st.caption("Green = strong (close to 10), Red = weak (close to 0). "
               "Morocco vs all benchmark countries.")

    countries_heat = [c for c in ["Morocco", "Chile", "Peru", "Czech Republic",
                                   "Romania", "Turkey"] if c in computed_scores]
    heat_data = np.array([[computed_scores[c].get(p, 0)
                           for p in pillar_names] for c in countries_heat])

    fig_heat = go.Figure(go.Heatmap(
        z=heat_data,
        x=pillar_names,
        y=countries_heat,
        colorscale=[[0, "#dc2626"], [0.5, "#f59e0b"], [1, "#16a34a"]],
        zmin=0, zmax=10,
        text=np.round(heat_data, 2),
        texttemplate="%{text}",
        textfont_size=13,
        hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.2f}/10<extra></extra>"))

    fig_heat.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=-15))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Gap analysis: Morocco vs best-in-class
    st.markdown("**Gap Analysis — Morocco vs Best-in-Class**")
    if "Morocco" in computed_scores:
        gap_rows = []
        for p in pillar_names:
            mar_val = computed_scores["Morocco"].get(p, 0)
            best_country = max(
                [c for c in computed_scores if c != "Morocco"],
                key=lambda c: computed_scores[c].get(p, 0))
            best_val = computed_scores[best_country].get(p, 0)
            gap = best_val - mar_val
            gap_rows.append({
                "Pillar": p,
                "Morocco": round(mar_val, 2),
                f"Best ({best_country})": round(best_val, 2),
                "Gap": round(gap, 2),
                "Priority": "🔴 High" if gap > 2 else "🟡 Medium" if gap > 1 else "🟢 Low"
            })
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)


# ── TAB 4: IT Adoption Timeline ───────────────────────────────────────────────
with tab4:
    st.subheader("IT Adoption Timeline — When Did Each Country Adopt IT?")
    st.caption("Where does Morocco stand today relative to peers at their point of IT adoption?")

    # Current Morocco score vs historical scores at adoption
    fig_timeline = go.Figure()

    # Plot each benchmark at their adoption year
    for country, info in IT_ADOPTION_HISTORY.items():
        comp = composite(info["scores"], weights)
        fig_timeline.add_trace(go.Scatter(
            x=[info["year"]], y=[comp],
            mode="markers+text",
            marker=dict(size=18, color=info["color"],
                        symbol="star" if "Success" in info["outcome"] else "x"),
            text=[f"{info['flag']} {country} ({info['year']})"],
            textposition="top center",
            textfont=dict(size=10),
            name=country,
            hovertemplate=f"<b>{country}</b><br>Adoption year: {info['year']}"
                          f"<br>Readiness score: {comp:.2f}/10"
                          f"<br>{info['outcome']}<extra></extra>"))

    # Morocco current
    if "Morocco" in composite_scores:
        mar_score = composite_scores["Morocco"]
        import datetime
        current_year = datetime.datetime.now().year
        fig_timeline.add_trace(go.Scatter(
            x=[current_year], y=[mar_score],
            mode="markers+text",
            marker=dict(size=22, color="#FF6B35",
                        symbol="diamond", line=dict(width=2, color="white")),
            text=["🇲🇦 Morocco (Today)"],
            textposition="top center",
            textfont=dict(size=11, color="#FF6B35"),
            name="Morocco (current)",
            hovertemplate=f"<b>Morocco</b><br>Current score: {mar_score:.2f}/10<extra></extra>"))

    # Minimum threshold line
    fig_timeline.add_hline(y=5.5, line_dash="dash", line_color="orange",
        annotation_text="Near-ready threshold (5.5)",
        annotation_position="left")
    fig_timeline.add_hline(y=7.5, line_dash="dash", line_color="green",
        annotation_text="IT-ready threshold (7.5)",
        annotation_position="left")

    fig_timeline.update_layout(
        height=480,
        xaxis_title="Year",
        yaxis_title="IT Readiness Score at Adoption",
        yaxis=dict(range=[0, 11], gridcolor="rgba(128,128,128,0.15)"),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5))
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Country outcome table
    st.markdown("**IT Adoption Outcomes by Country**")
    outcome_df = pd.DataFrame([{
        "Country": f"{info['flag']} {c}",
        "IT Adoption Year": info["year"],
        "Readiness Score at Adoption": round(composite(info["scores"], weights), 2),
        "Outcome": info["outcome"]
    } for c, info in IT_ADOPTION_HISTORY.items()])
    st.dataframe(outcome_df, use_container_width=True, hide_index=True)

    # Morocco comparison note
    if "Morocco" in composite_scores:
        mar = composite_scores["Morocco"]
        chile_score = composite(IT_ADOPTION_HISTORY["Chile"]["scores"], weights)
        turkey_score = composite(IT_ADOPTION_HISTORY["Turkey"]["scores"], weights)
        st.markdown(f"""
        **Where does Morocco sit?**
        - Morocco today scores **{mar:.2f}/10**
        - Chile scored **{chile_score:.2f}/10** when it adopted IT (1999) → *successful*
        - Turkey scored **{turkey_score:.2f}/10** when it adopted IT (2006) → *struggled*
        - Morocco is {"**closer to Chile's profile**" if abs(mar - chile_score) < abs(mar - turkey_score) else "**closer to Turkey's profile**"} at the point of adoption
        """)


# ── TAB 5: Scenario Analysis ──────────────────────────────────────────────────
with tab5:
    st.subheader("Scenario Analysis — What Would It Take?")
    st.caption("Adjust Morocco's indicators to see how the readiness score responds.")

    st.markdown("**Simulate Morocco's score under different policy scenarios**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Monetary Stability*")
        sim_inf_level = st.slider("Inflation level (%)", 0.0, 20.0, 3.5, 0.5)
        sim_inf_vol = st.slider("Inflation volatility (%)", 0.0, 10.0, 1.5, 0.5)

        st.markdown("*Transmission*")
        sim_erpt = st.slider("ERPT coefficient", 0.0, 1.0,
            float(f"{erpt_used:.2f}") if erpt_used else 0.25, 0.05)

    with col2:
        st.markdown("*Exchange Rate*")
        sim_fx_vol = st.slider("FX annualized volatility (%)", 0.0, 20.0, 3.0, 0.5)
        sim_reer_std = st.slider("REER stability (%)", 0.0, 15.0, 2.5, 0.5)

        st.markdown("*External Resilience*")
        sim_reserves = st.slider("FX reserves (USD bn)", 0.0, 100.0, 25.0, 1.0)
        sim_res_trend = st.slider("5-year reserves change (USD bn)", -20.0, 30.0, 2.0, 1.0)

    # Compute simulated scores
    sim_scores = {
        "Monetary Stability": np.mean([
            score_indicator(sim_inf_level,
                PILLARS["Monetary Stability"]["indicators"]
                ["Inflation level (avg CPI growth %)"]["thresholds"], "low"),
            score_indicator(sim_inf_vol,
                PILLARS["Monetary Stability"]["indicators"]
                ["Inflation volatility (std CPI growth %)"]["thresholds"], "low"),
        ]),
        "Exchange Rate Environment": np.mean([
            score_indicator(sim_fx_vol,
                PILLARS["Exchange Rate Environment"]["indicators"]
                ["FX volatility (ann. %)"]["thresholds"], "low"),
            score_indicator(sim_reer_std,
                PILLARS["Exchange Rate Environment"]["indicators"]
                ["REER stability (std REER %)"]["thresholds"], "low"),
        ]),
        "Transmission Effectiveness": np.mean([
            score_indicator(sim_erpt,
                PILLARS["Transmission Effectiveness"]["indicators"]
                ["ERPT coefficient (long-run)"]["thresholds"], "low"),
            score_indicator(0.3,
                PILLARS["Transmission Effectiveness"]["indicators"]
                ["Policy rate responsiveness"]["thresholds"], "high_variation"),
        ]),
        "External Resilience": np.mean([
            score_indicator(sim_reserves,
                PILLARS["External Resilience"]["indicators"]
                ["FX reserves adequacy (USD bn)"]["thresholds"], "high"),
            score_indicator(sim_res_trend,
                PILLARS["External Resilience"]["indicators"]
                ["Reserves trend (change over 5y)"]["thresholds"], "high"),
        ]),
    }
    sim_composite = composite(sim_scores, weights)
    current_composite = composite_scores.get("Morocco", 5.0)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sim_composite,
        delta={"reference": current_composite, "valueformat": ".2f"},
        title={"text": "Simulated IT Readiness Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": "#FF6B35"},
            "steps": [
                {"range": [0, 5.5], "color": "rgba(220,38,38,0.1)"},
                {"range": [5.5, 7.5], "color": "rgba(245,158,11,0.1)"},
                {"range": [7.5, 10], "color": "rgba(22,163,74,0.1)"},
            ],
            "threshold": {
                "line": {"color": "green", "width": 3},
                "thickness": 0.75,
                "value": 7.5}
        }))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Before vs after
    c1, c2, c3 = st.columns(3)
    c1.metric("Current score", f"{current_composite:.2f}/10")
    c2.metric("Simulated score", f"{sim_composite:.2f}/10",
        delta=f"{sim_composite - current_composite:+.2f}")
    c3.metric("Status change",
        "🟢 Ready" if sim_composite >= 7.5 else
        "🟡 Near-ready" if sim_composite >= 5.5 else "🔴 Not ready")

    # Comparison radar
    fig_scenario_radar = go.Figure()
    cats = pillar_names + [pillar_names[0]]

    if "Morocco" in computed_scores:
        cur_vals = [computed_scores["Morocco"].get(p, 0) for p in pillar_names]
        fig_scenario_radar.add_trace(go.Scatterpolar(
            r=cur_vals + [cur_vals[0]], theta=cats,
            fill="toself", fillcolor="rgba(255,107,53,0.1)",
            line=dict(color="#FF6B35", width=2, dash="dash"),
            name="Current Morocco"))

    sim_vals = [sim_scores.get(p, 0) for p in pillar_names]
    fig_scenario_radar.add_trace(go.Scatterpolar(
        r=sim_vals + [sim_vals[0]], theta=cats,
        fill="toself", fillcolor="rgba(37,99,235,0.15)",
        line=dict(color="#2563EB", width=2.5),
        name="Simulated Morocco"))

    fig_scenario_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10],
                                   gridcolor="rgba(128,128,128,0.2)")),
        height=400,
        legend=dict(orientation="h", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scenario_radar, use_container_width=True)
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

st.set_page_config(page_title="IT Readiness", page_icon="🎯", layout="wide")
apply_global_style()

st.title("🎯 Page 5 — Inflation Targeting Readiness Dashboard")
st.caption("A four-pillar composite index scoring Morocco's preparedness to adopt inflation targeting, "
           "benchmarked against literature-based readiness thresholds.")

# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Settings")
country = sidebar_country_selector("Morocco")
df = load_country(country)
if df is None:
    st.error(f"No data found for {country}.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Index Configuration")

# Pillar weights (must sum to 1) — user-adjustable for sensitivity analysis
st.sidebar.caption("Pillar weights (sensitivity analysis)")
w_mon  = st.sidebar.slider("Monetary Stability",       0.0, 1.0, 0.30, 0.05)
w_fx   = st.sidebar.slider("Exchange Rate Environment", 0.0, 1.0, 0.25, 0.05)
w_tr   = st.sidebar.slider("Transmission Effectiveness", 0.0, 1.0, 0.25, 0.05)
w_ext  = st.sidebar.slider("External Resilience",       0.0, 1.0, 0.20, 0.05)
w_total = w_mon + w_fx + w_tr + w_ext
if w_total == 0:
    w_total = 1.0
weights = {
    "Monetary Stability":        w_mon / w_total,
    "Exchange Rate Environment": w_fx  / w_total,
    "Transmission Effectiveness": w_tr / w_total,
    "External Resilience":       w_ext / w_total,
}

st.sidebar.markdown("---")
st.sidebar.subheader("Transmission Input")
erpt_from_page4 = st.session_state.get("erpt_long_run", None)
if erpt_from_page4 is not None:
    st.sidebar.success(f"✅ Using ERPT from Page 4: {erpt_from_page4:.4f}")
    default_erpt = abs(float(erpt_from_page4))
else:
    st.sidebar.info("ℹ️ Fit the VAR on Page 4 to auto-fill ERPT.")
    default_erpt = 0.25
erpt = st.sidebar.number_input("Long-run ERPT (|value|)",
    min_value=0.0, max_value=2.0, value=round(default_erpt, 4), step=0.01)

# ════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE — converts real Morocco data into 0–10 pillar scores
# Thresholds are literature-based (IMF IT-readiness criteria).
# ════════════════════════════════════════════════════════════════════════════
def clamp(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))

def score_linear(value, best, worst):
    """Map a value to 0–10. best→10, worst→0, linear in between."""
    if best == worst:
        return 5.0
    s = 10.0 * (value - worst) / (best - worst)
    return clamp(s)

# ── Compute recent indicators from real data ─────────────────────────────────
cpi = df["cpi"].dropna()
infl_yoy = (cpi / cpi.shift(12) - 1) * 100
recent_infl = infl_yoy.tail(12).mean()           # avg inflation last 12m
infl_vol    = infl_yoy.tail(36).std()            # inflation volatility last 3y

fx = df["fx_eur"].dropna()
fx_ret = np.log(fx / fx.shift(1)).dropna() * 100
fx_vol = fx_ret.tail(36).std()                   # monthly FX volatility last 3y

has_reserves = "reserves" in df.columns and df["reserves"].notna().any()
if has_reserves:
    res = df["reserves"].dropna()
    res_level  = res.tail(1).values[0]
    res_trend  = (res.tail(12).mean() / res.tail(24).head(12).mean() - 1) * 100  # 12m growth
else:
    res_level, res_trend = None, None

# ── Pillar 1: Monetary Stability ──────────────────────────────────────────────
# Lower, stable inflation = ready. IMF: single-digit, stable inflation needed.
s_infl_level = score_linear(recent_infl, best=2.0, worst=10.0)   # 2% ideal, 10% poor
s_infl_vol   = score_linear(infl_vol,    best=0.5, worst=4.0)    # low vol ideal
monetary_score = 0.6 * s_infl_level + 0.4 * s_infl_vol

# ── Pillar 2: Exchange Rate Environment ───────────────────────────────────────
# IT needs SOME flexibility. Morocco moved to ±5% band (2020). Moderate vol ideal.
# Too low vol = still effectively pegged; too high = disorderly.
band_flex = 8.0  # ±5% band achieved → high flexibility score (literature: ±5% = meaningful float prep)
s_flex = band_flex
# Moderate FX volatility is ideal (around 0.8–1.2% monthly); penalize extremes
ideal_vol = 1.0
s_fx_vol = clamp(10.0 - abs(fx_vol - ideal_vol) * 4.0)
fx_env_score = 0.5 * s_flex + 0.5 * s_fx_vol

# ── Pillar 3: Transmission Effectiveness ──────────────────────────────────────
# LOW ERPT = favorable for IT (FX shocks don't destabilize inflation).
# Literature: ERPT < 0.20 favorable, 0.20–0.50 moderate, > 0.50 risky.
s_erpt = score_linear(erpt, best=0.0, worst=0.6)   # 0 = perfect insulation, 0.6 = risky
# Interest-rate channel presence (proxy: rate variation exists)
rate_changes = df["policy_rate"].diff().abs().gt(0.001).sum() if "policy_rate" in df.columns else 0
s_rate_channel = clamp(score_linear(rate_changes, best=15, worst=2))
transmission_score = 0.7 * s_erpt + 0.3 * s_rate_channel

# ── Pillar 4: External Resilience ─────────────────────────────────────────────
if has_reserves:
    # Higher reserves level and positive trend = stronger buffer.
    # Normalize level against a rough adequacy band (Morocco ~ 20–55 USD bn equiv).
    s_res_level = score_linear(res_level, best=df["reserves"].max(), worst=df["reserves"].min())
    s_res_trend = clamp(5.0 + res_trend * 0.5)     # positive growth lifts score
    external_score = 0.7 * s_res_level + 0.3 * s_res_trend
else:
    external_score = 5.0  # neutral default if no reserves data

pillar_scores = {
    "Monetary Stability":        round(monetary_score, 2),
    "Exchange Rate Environment": round(fx_env_score, 2),
    "Transmission Effectiveness": round(transmission_score, 2),
    "External Resilience":       round(external_score, 2),
}

composite = sum(pillar_scores[p] * weights[p] for p in pillar_scores)

# Literature-based IT-ready threshold (composite ≥ 6.0 generally considered ready)
READY_THRESHOLD = 6.0

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Composite Verdict",
    "📡 Four-Pillar Radar",
    "🔍 Pillar Deep-Dive",
    "📊 Benchmark vs Thresholds",
    "🧪 Scenario & Sensitivity",
])

# ── TAB 1: Composite Verdict ──────────────────────────────────────────────────
with tab1:
    st.subheader("Overall IT Readiness Verdict")

    c1, c2 = st.columns([1, 1])
    with c1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=composite,
            delta={"reference": READY_THRESHOLD, "increasing": {"color": "#16a34a"},
                   "decreasing": {"color": "#dc2626"}},
            number={"font": {"size": 48}},
            gauge={
                "axis": {"range": [0, 10], "tickwidth": 1},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [0, 4],  "color": "rgba(220,38,38,0.25)"},
                    {"range": [4, 6],  "color": "rgba(245,158,11,0.25)"},
                    {"range": [6, 10], "color": "rgba(22,163,74,0.25)"},
                ],
                "threshold": {"line": {"color": "#FFFFFF", "width": 3},
                              "thickness": 0.8, "value": READY_THRESHOLD},
            },
            title={"text": "Composite Readiness Score (0–10)"},
        ))
        fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E2E8F0"})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        st.markdown("#### Verdict")
        if composite >= 6.5:
            st.success(f"**🟢 Ready for transition** — Composite score of **{composite:.2f}/10** "
                       f"exceeds the {READY_THRESHOLD} readiness threshold. Morocco's macroeconomic "
                       "preconditions for inflation targeting are largely satisfied.")
        elif composite >= READY_THRESHOLD:
            st.info(f"**🟡 Approaching readiness** — Composite score of **{composite:.2f}/10** "
                    f"meets the {READY_THRESHOLD} threshold but with limited margin. A phased "
                    "transition with continued institutional strengthening is advisable.")
        else:
            st.warning(f"**🟠 Not yet ready** — Composite score of **{composite:.2f}/10** "
                       f"falls below the {READY_THRESHOLD} threshold. Targeted reforms are needed "
                       "before adopting full inflation targeting.")

        st.markdown("#### Pillar contributions")
        for p, s in pillar_scores.items():
            contrib = s * weights[p]
            bar = "█" * int(round(s)) + "░" * (10 - int(round(s)))
            st.markdown(
                f"<div style='font-size:0.85rem;margin:2px 0;'>"
                f"<span style='color:#94A3B8;'>{p}</span> "
                f"<span style='color:#60A5FA;font-family:monospace;'>{bar}</span> "
                f"<b>{s:.1f}</b> <span style='color:#64748B;'>(w={weights[p]:.0%})</span></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Composite", f"{composite:.2f}", f"{composite-READY_THRESHOLD:+.2f} vs threshold")
    m2.metric("Monetary", f"{pillar_scores['Monetary Stability']:.1f}")
    m3.metric("FX Env.", f"{pillar_scores['Exchange Rate Environment']:.1f}")
    m4.metric("Transmission", f"{pillar_scores['Transmission Effectiveness']:.1f}")
    m5.metric("External", f"{pillar_scores['External Resilience']:.1f}")

# ── TAB 2: Radar ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Four-Pillar Readiness Profile")

    cats = list(pillar_scores.keys())
    vals = list(pillar_scores.values())

    fig_radar = go.Figure()
    # Readiness threshold ring
    fig_radar.add_trace(go.Scatterpolar(
        r=[READY_THRESHOLD]*len(cats) + [READY_THRESHOLD],
        theta=cats + [cats[0]],
        fill="toself", name=f"Ready threshold ({READY_THRESHOLD})",
        fillcolor="rgba(22,163,74,0.08)",
        line=dict(color="rgba(22,163,74,0.6)", dash="dash", width=1.5)))
    # Morocco profile
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself", name="Morocco (current)",
        fillcolor="rgba(37,99,235,0.25)",
        line=dict(color="#2563EB", width=2.5)))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="rgba(148,163,184,0.2)"),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
            bgcolor="rgba(0,0,0,0)"),
        height=480, paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=60, r=60, t=60, b=40))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.caption("The dashed green ring marks the literature-based readiness threshold. "
               "Where Morocco's blue profile extends beyond the ring, that pillar meets the IT-readiness bar.")

# ── TAB 3: Pillar Deep-Dive ───────────────────────────────────────────────────
with tab3:
    st.subheader("Pillar Deep-Dive — Sub-Indicators")

    deep = {
        "Monetary Stability": [
            ("Average inflation (12m)", f"{recent_infl:.2f}%", f"{s_infl_level:.1f}/10",
             "Target: 1–3%. Lower, stable inflation signals readiness."),
            ("Inflation volatility (3y std)", f"{infl_vol:.2f}", f"{s_infl_vol:.1f}/10",
             "Lower volatility indicates a stable price environment."),
        ],
        "Exchange Rate Environment": [
            ("Band flexibility", "±5% (since 2020)", f"{s_flex:.1f}/10",
             "IT requires meaningful exchange rate flexibility."),
            ("FX volatility (3y, monthly)", f"{fx_vol:.2f}%", f"{s_fx_vol:.1f}/10",
             "Moderate volatility (~1%) is ideal — not pegged, not disorderly."),
        ],
        "Transmission Effectiveness": [
            ("Exchange rate pass-through", f"{erpt:.3f}", f"{s_erpt:.1f}/10",
             "Low ERPT (<0.20) is favorable — FX shocks barely move prices."),
            ("Interest-rate channel", f"{int(rate_changes)} rate moves", f"{s_rate_channel:.1f}/10",
             "An active policy rate channel supports IT implementation."),
        ],
        "External Resilience": [
            ("Reserve level", f"{res_level:,.0f}" if has_reserves else "n/a",
             f"{external_score:.1f}/10" if has_reserves else "5.0/10 (default)",
             "Adequate reserves buffer external shocks during transition."),
        ],
    }

    for pillar, rows in deep.items():
        st.markdown(f"#### {pillar} — **{pillar_scores[pillar]:.1f}/10**")
        table = pd.DataFrame(
            [{"Indicator": r[0], "Value": r[1], "Score": r[2], "Interpretation": r[3]} for r in rows]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown("")

# ── TAB 4: Benchmark vs Thresholds ────────────────────────────────────────────
with tab4:
    st.subheader("Benchmark Against IT-Readiness Thresholds")
    st.caption("Comparison against literature-based criteria for IT adoption "
               "(IMF readiness frameworks). Green = meets threshold.")

    benchmarks = [
        ("Inflation level", recent_infl, "< 5%", recent_infl < 5, "%"),
        ("Inflation volatility", infl_vol, "< 2.0", infl_vol < 2.0, ""),
        ("Exchange-rate flexibility", 5.0, "≥ ±2.5% band", True, "% band"),
        ("Pass-through (ERPT)", erpt, "< 0.20", erpt < 0.20, ""),
    ]
    if has_reserves:
        # Rough proxy: reserves trend positive
        benchmarks.append(("Reserve adequacy", res_trend, "stable/rising", res_trend > -5, "% 12m"))

    rows = []
    met = 0
    for name, val, thresh, ok, unit in benchmarks:
        if ok: met += 1
        rows.append({
            "Criterion": name,
            "Morocco": f"{val:.2f}{unit}" if isinstance(val, (int, float)) else str(val),
            "Threshold": thresh,
            "Status": "✅ Met" if ok else "❌ Not met",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pct = met / len(benchmarks) * 100
    if pct >= 80:
        st.success(f"**{met}/{len(benchmarks)} criteria met ({pct:.0f}%)** — strong alignment with IT-readiness conditions.")
    elif pct >= 60:
        st.info(f"**{met}/{len(benchmarks)} criteria met ({pct:.0f}%)** — most conditions satisfied; some gaps remain.")
    else:
        st.warning(f"**{met}/{len(benchmarks)} criteria met ({pct:.0f}%)** — several conditions not yet satisfied.")

    st.markdown("""
    <div style="margin-top:16px;color:#64748B;font-size:0.82rem;">
    Thresholds drawn from IMF inflation-targeting readiness literature
    (e.g. Batini &amp; Laxton 2007; IMF Working Papers on IT preconditions).
    These represent commonly cited benchmarks, not a single official standard.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5: Scenario & Sensitivity ─────────────────────────────────────────────
with tab5:
    st.subheader("Scenario & Sensitivity Analysis")
    st.caption("How would the composite score change under different conditions? "
               "Adjust the levers below.")

    c1, c2, c3 = st.columns(3)
    with c1:
        scn_infl = st.slider("Inflation scenario (%)", 0.0, 12.0, float(round(recent_infl, 1)), 0.5)
    with c2:
        scn_erpt = st.slider("ERPT scenario", 0.0, 0.8, float(round(erpt, 2)), 0.05)
    with c3:
        scn_resshift = st.slider("Reserve change (%)", -30.0, 30.0, 0.0, 5.0)

    # Recompute scores under scenario
    s_infl_level_s = score_linear(scn_infl, best=2.0, worst=10.0)
    monetary_s = 0.6 * s_infl_level_s + 0.4 * s_infl_vol
    s_erpt_s = score_linear(scn_erpt, best=0.0, worst=0.6)
    transmission_s = 0.7 * s_erpt_s + 0.3 * s_rate_channel
    if has_reserves:
        s_res_level_s = clamp(score_linear(res_level, best=df["reserves"].max(),
                                           worst=df["reserves"].min()) + scn_resshift * 0.1)
        external_s = 0.7 * s_res_level_s + 0.3 * clamp(5.0 + res_trend * 0.5)
    else:
        external_s = 5.0

    scenario_scores = {
        "Monetary Stability": monetary_s,
        "Exchange Rate Environment": fx_env_score,
        "Transmission Effectiveness": transmission_s,
        "External Resilience": external_s,
    }
    composite_s = sum(scenario_scores[p] * weights[p] for p in scenario_scores)

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.metric("Scenario composite", f"{composite_s:.2f}",
                  f"{composite_s - composite:+.2f} vs current")
        if composite_s >= READY_THRESHOLD:
            st.success("Under this scenario, Morocco meets the readiness threshold.")
        else:
            st.warning("Under this scenario, Morocco falls below the readiness threshold.")

    with cc2:
        # Side-by-side bar
        fig_scn = go.Figure()
        fig_scn.add_trace(go.Bar(
            name="Current", x=list(pillar_scores.keys()), y=list(pillar_scores.values()),
            marker_color="#2563EB"))
        fig_scn.add_trace(go.Bar(
            name="Scenario", x=list(scenario_scores.keys()), y=list(scenario_scores.values()),
            marker_color="#f59e0b"))
        fig_scn.update_layout(
            barmode="group", height=320,
            yaxis=dict(range=[0, 10], gridcolor="rgba(148,163,184,0.15)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E2E8F0"}, legend=dict(orientation="h", y=1.15),
            margin=dict(l=0, r=0, t=30, b=80))
        fig_scn.update_xaxes(tickangle=-20, tickfont=dict(size=9))
        st.plotly_chart(fig_scn, use_container_width=True)

    st.markdown("---")
    st.markdown("**Tornado: sensitivity of composite to each pillar (±2 points)**")
    tornado = []
    for p in pillar_scores:
        base = composite
        up = base + 2 * weights[p]
        down = base - 2 * weights[p]
        tornado.append({"Pillar": p, "low": down, "high": up, "swing": up - down})
    tdf = pd.DataFrame(tornado).sort_values("swing")
    fig_t = go.Figure()
    for _, r in tdf.iterrows():
        fig_t.add_trace(go.Bar(
            y=[r["Pillar"]], x=[r["high"] - r["low"]], base=r["low"],
            orientation="h", marker_color="#7c3aed", showlegend=False))
    fig_t.add_vline(x=composite, line_dash="dash", line_color="#FFFFFF")
    fig_t.update_layout(
        height=240, xaxis=dict(range=[0, 10], title="Composite score",
                               gridcolor="rgba(148,163,184,0.15)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"}, margin=dict(l=0, r=0, t=10, b=30))
    st.plotly_chart(fig_t, use_container_width=True)
    st.caption("Longer bars = pillars whose change moves the composite most "
               "(driven by their weight). Useful for prioritizing reform focus.")
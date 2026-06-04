import streamlit as st
import pandas as pd
import sys
import os



sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import (
    load_all_countries,
    validate_dataframe,
    get_available_countries,
    COUNTRIES,
    REQUIRED_COLUMNS,
    COLUMN_LABELS,
    DATA_DIR,
)
#

from utils.style import apply_global_style
apply_global_style()

st.set_page_config(page_title="Data Hub", page_icon="📂", layout="wide")
st.title("📂 Page 0 — Data Hub & Configuration")
st.caption("Load, validate and preview your country datasets before running any analysis.")

st.markdown("---")

# ── Data folder status ──────────────────────────────────────────────────────
st.subheader("Data Folder Status")

available = get_available_countries()
all_countries = list(COUNTRIES.keys())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Countries available", f"{len(available)} / {len(all_countries)}")
with col2:
    st.metric("Data folder", os.path.basename(os.path.dirname(DATA_DIR)) + "/data")
with col3:
    if len(available) == len(all_countries):
        st.success("All country files found ✓")
    elif len(available) > 0:
        st.warning(f"{len(all_countries) - len(available)} files missing")
    else:
        st.error("No data files found")

# ── File availability matrix ─────────────────────────────────────────────────
st.markdown("### File Availability")

status_data = []
for country, filename in COUNTRIES.items():
    filepath = os.path.join(DATA_DIR, filename)
    exists = os.path.exists(filepath)
    status_data.append({
        "Country": country,
        "File": filename,
        "Status": "✅ Found" if exists else "❌ Missing",
    })

st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

# ── Load & validate ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Load & Validate Data")

if not available:
    st.warning("""
    No CSV files found in the `/data` folder.

    **Expected format:** one CSV per country named exactly as listed above.

    Each file must have the following columns:
    """)
    col_df = pd.DataFrame([
        {"Column": col, "Description": COLUMN_LABELS.get(col, col), "Type": "Monthly", "Example": ex}
        for col, ex in zip(
            [c for c in REQUIRED_COLUMNS if c != "date"],
            ["112.4", "3.00", "97.2", "98.5", "10.05", "10.89", "31.2", "82.4", "-0.3"]
        )
    ])
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    st.info("📅 **Date format:** YYYY-MM-DD (e.g. 2010-01-01), monthly frequency")
    st.stop()

if st.button("🔄 Load & Validate All Countries", type="primary"):
    with st.spinner("Loading data..."):
        data = load_all_countries()

    if not data:
        st.error("No data could be loaded.")
        st.stop()

    st.success(f"Loaded {len(data)} country datasets successfully.")

    # Validation reports
    st.markdown("### Validation Report")

    for country, df in data.items():
        report = validate_dataframe(df, country)
        with st.expander(f"{'✅' if not report['issues'] else '⚠️'} {country} — {report['row_count']} observations ({report['date_range'][0].strftime('%Y-%m')} to {report['date_range'][1].strftime('%Y-%m')})"):

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Rows", report["row_count"])
            with c2:
                start = report["date_range"][0].strftime("%Y-%m")
                end = report["date_range"][1].strftime("%Y-%m")
                st.metric("Date range", f"{start} → {end}")
            with c3:
                st.metric("Missing columns", len(report["missing_cols"]))

            if report["issues"]:
                for issue in report["issues"]:
                    st.warning(issue)
            else:
                st.success("No issues found")

            if report["missing_values"]:
                mv_df = pd.DataFrame([
                    {"Column": col, "Missing values": v["count"], "Percentage": f"{v['pct']}%"}
                    for col, v in report["missing_values"].items()
                ])
                st.markdown("**Missing values:**")
                st.dataframe(mv_df, use_container_width=True, hide_index=True)

    # ── Data preview ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Data Preview")

    selected = st.selectbox("Select country to preview", list(data.keys()))
    df_preview = data[selected]

    st.dataframe(df_preview.head(24), use_container_width=True)

    # ── Column coverage heatmap ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Column Coverage by Country")

    expected_cols = [c for c in REQUIRED_COLUMNS if c != "date"]
    coverage = {}
    for country, df in data.items():
        coverage[country] = {col: "✅" if col in df.columns else "❌" for col in expected_cols}

    coverage_df = pd.DataFrame(coverage).T
    coverage_df.columns = [COLUMN_LABELS.get(c, c) for c in expected_cols]
    st.dataframe(coverage_df, use_container_width=True)

    st.session_state["data_loaded"] = True
    st.session_state["data"] = data

else:
    st.info("Click the button above to load and validate your datasets.")

# ── CSV template download ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Download CSV Template")
st.caption("Use this template to build your country datasets in the correct format.")

template_rows = [
    {
        "date": f"2010-{str(m).zfill(2)}-01",
        "cpi": "",
        "policy_rate": "",
        "neer": "",
        "reer": "",
        "fx_usd": "",
        "fx_eur": "",
        "reserves": "",
        "oil_brent": "",
        "output_gap": "",
    }
    for m in range(1, 13)
]
template_df = pd.DataFrame(template_rows)
csv_template = template_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download CSV Template",
    data=csv_template,
    file_name="country_template.csv",
    mime="text/csv",
)
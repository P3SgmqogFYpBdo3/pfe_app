"""
Data loader for the PFE application — Morocco only.

Loads and validates the Morocco dataset with robust date parsing and
BOM-safe CSV reading. Public function signatures are preserved so the
existing pages continue to work unchanged.
"""
import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Single-country project.
COUNTRIES = {
    "Morocco": "morocco.csv",
}

# Known columns (output_gap intentionally removed — not used in this project).
COLUMN_LABELS = {
    "cpi":         "CPI (Inflation Index)",
    "policy_rate": "Policy Interest Rate (%)",
    "neer":        "NEER Index",
    "reer":        "REER Index",
    "fx_usd":      "MAD / USD",
    "fx_eur":      "MAD / EUR",
    "reserves":    "FX Reserves",
    "oil_brent":   "Brent Oil Price (USD)",
}

CORE_COLUMNS = ["date", "cpi", "fx_eur"]          # minimum to run the app
OPTIONAL_COLUMNS = ["fx_usd", "neer", "reer", "policy_rate", "reserves", "oil_brent"]

# Backward-compatibility alias (some pages import this name).
REQUIRED_COLUMNS = ["date"] + list(COLUMN_LABELS.keys())


@st.cache_data(show_spinner=False)
def load_country(country: str = "Morocco"):
    """Load and clean the Morocco dataset. Returns a month-indexed DataFrame,
    or None on failure."""
    filename = COUNTRIES.get(country)
    if not filename:
        return None

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None

    try:
        # utf-8-sig strips a leading BOM if present (common from Excel exports).
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        # ── Locate the date column case-insensitively ─────────────────────
        date_col = None
        for col in df.columns:
            if str(col).lower().strip() == "date":
                date_col = col
                break
        if date_col is None:
            st.error(f"No 'date' column found in {filename}. "
                     f"Columns: {list(df.columns)}")
            return None

        # ── Parse dates flexibly, drop unparseable rows ───────────────────
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

        n_bad = df["date"].isna().sum()
        if n_bad > 0:
            st.warning(f"{country}: {n_bad} row(s) with unparseable dates dropped.")
            df = df.dropna(subset=["date"])
        if len(df) == 0:
            st.error(f"No valid dates in {filename} (expected YYYY-MM-DD).")
            return None

        # ── Normalize to first-of-month and index ─────────────────────────
        df = df.sort_values("date")
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df = df.drop_duplicates(subset=["date"], keep="last").set_index("date")

        # ── Drop output_gap if present (not used in this project) ──────────
        if "output_gap" in df.columns:
            df = df.drop(columns=["output_gap"])

        # ── Coerce every remaining column to numeric ──────────────────────
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Error loading {country}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_all_countries():
    """Return {country: DataFrame} — single entry for Morocco."""
    result = {}
    for country in COUNTRIES:
        d = load_country(country)
        if d is not None:
            result[country] = d
    return result


def get_available_countries():
    """List countries that have a CSV present in /data (Morocco only)."""
    return [c for c, f in COUNTRIES.items()
            if os.path.exists(os.path.join(DATA_DIR, f))]


def validate_dataframe(df, country: str = "Morocco"):
    """Return a validation report dict for the loaded dataframe."""
    report = {
        "country": country,
        "row_count": len(df),
        "date_range": (df.index.min(), df.index.max()) if len(df) else (None, None),
        "present_cols": [],
        "missing_optional": [],
        "missing_cols": [],          # backward-compat alias for older pages
        "missing_values": {},
        "date_gaps": 0,
        "issues": [],
    }

    for col in COLUMN_LABELS:
        if col in df.columns:
            report["present_cols"].append(col)
        else:
            report["missing_optional"].append(col)
            report["missing_cols"].append(col)

    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            report["missing_values"][col] = {
                "count": n, "pct": round(n / len(df) * 100, 1)}

    if len(df) > 1:
        expected = pd.date_range(df.index.min(), df.index.max(), freq="MS")
        report["date_gaps"] = len(expected.difference(df.index))

    missing_core = [c for c in CORE_COLUMNS if c != "date" and c not in df.columns]
    if missing_core:
        report["issues"].append(f"Missing core columns: {', '.join(missing_core)}")
    if len(df) < 60:
        report["issues"].append("Fewer than 60 observations — models may be unreliable.")
    if report["date_gaps"] > 0:
        report["issues"].append(f"{report['date_gaps']} missing month(s) in the date range.")
    high_missing = [c for c, v in report["missing_values"].items() if v["pct"] > 20]
    if high_missing:
        report["issues"].append(f"High missing values (>20%): {', '.join(high_missing)}")

    return report


def compute_returns(df, col: str):
    """Log returns for a column (in %)."""
    import numpy as np
    return (np.log(df[col] / df[col].shift(1)) * 100)


def sidebar_country_selector(default: str = "Morocco") -> str:
    """Returns 'Morocco' silently. Kept for compatibility with all pages —
    renders nothing in the sidebar since this is a single-country project."""
    return "Morocco"
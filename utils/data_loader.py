import pandas as pd
import streamlit as st
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

COUNTRIES = {
    "Morocco": "morocco.csv",
    "Chile": "chile.csv",
    "Turkey": "turkey.csv",
    "Romania": "romania.csv",
    "Peru": "peru.csv",
    "Czech Republic": "czech_republic.csv",
}

REQUIRED_COLUMNS = [
    "date",
    "cpi",
    "policy_rate",
    "neer",
    "reer",
    "fx_usd",
    "fx_eur",
    "reserves",
    "oil_brent",
    "output_gap",
]

COLUMN_LABELS = {
    "cpi": "CPI (Inflation Index)",
    "policy_rate": "Policy Interest Rate (%)",
    "neer": "NEER Index",
    "reer": "REER Index",
    "fx_usd": "MAD / USD",
    "fx_eur": "MAD / EUR",
    "reserves": "FX Reserves (USD billions)",
    "oil_brent": "Brent Oil Price (USD)",
    "output_gap": "Output Gap (% of GDP)",
}


@st.cache_data(show_spinner=False)
def load_country(country: str) -> pd.DataFrame | None:
    """Load and validate data for a single country."""
    filename = COUNTRIES.get(country)
    if not filename:
        return None

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").set_index("date")
        df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
        return df
    except Exception as e:
        st.error(f"Error loading {country}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_all_countries() -> dict[str, pd.DataFrame]:
    """Load all available country datasets."""
    result = {}
    for country in COUNTRIES:
        df = load_country(country)
        if df is not None:
            result[country] = df
    return result


def get_available_countries() -> list[str]:
    """Return list of countries that have a CSV file in /data."""
    available = []
    for country, filename in COUNTRIES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            available.append(country)
    return available


def validate_dataframe(df: pd.DataFrame, country: str) -> dict:
    """
    Validate a dataframe and return a report.
    Returns dict with: missing_cols, missing_values, date_range, row_count, issues
    """
    report = {
        "country": country,
        "row_count": len(df),
        "date_range": (df.index.min(), df.index.max()) if len(df) > 0 else (None, None),
        "missing_cols": [],
        "missing_values": {},
        "issues": [],
    }

    # Check for required columns
    for col in REQUIRED_COLUMNS:
        if col != "date" and col not in df.columns:
            report["missing_cols"].append(col)

    # Check for missing values in existing columns
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            pct = round(n_missing / len(df) * 100, 1)
            report["missing_values"][col] = {"count": n_missing, "pct": pct}

    # Flag issues
    if report["missing_cols"]:
        report["issues"].append(f"Missing columns: {', '.join(report['missing_cols'])}")
    if len(df) < 60:
        report["issues"].append("Less than 60 observations — models may be unreliable")
    if report["missing_values"]:
        high_missing = [c for c, v in report["missing_values"].items() if v["pct"] > 20]
        if high_missing:
            report["issues"].append(f"High missing values (>20%): {', '.join(high_missing)}")

    return report


def compute_returns(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute log returns for a given column."""
    return (df[col] / df[col].shift(1)).apply(lambda x: x if pd.isna(x) else __import__("math").log(x))


def sidebar_country_selector(default: str = "Morocco") -> str:
    """Render a country selector in the sidebar and return the selected country."""
    available = get_available_countries()
    if not available:
        st.sidebar.warning("No country data found. Please add CSV files to /data.")
        return default
    idx = available.index(default) if default in available else 0
    return st.sidebar.selectbox("🌍 Select Country", available, index=idx)

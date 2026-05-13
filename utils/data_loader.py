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
    "date", "cpi", "policy_rate", "neer", "reer",
    "fx_usd", "fx_eur", "reserves", "oil_brent", "output_gap",
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
def load_country(country: str):
    """Load and validate data for a single country with robust date parsing."""
    filename = COUNTRIES.get(country)
    if not filename:
        return None

    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)

        # Find date column case-insensitively
        date_col = None
        for col in df.columns:
            if col.lower().strip() == "date":
                date_col = col
                break

        if date_col is None:
            st.error(f"No 'date' column found in {filename}. "
                     f"Columns found: {list(df.columns)}")
            return None

        # Parse dates flexibly
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")

        # Rename to lowercase 'date' if needed
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})

        # Drop rows where date couldn't be parsed
        n_bad = df["date"].isna().sum()
        if n_bad > 0:
            st.warning(f"{country}: {n_bad} rows with unparseable dates dropped.")
            df = df.dropna(subset=["date"])

        if len(df) == 0:
            st.error(f"No valid dates in {filename}. Check format — expected YYYY-MM-DD.")
            return None

        # Normalize to first of month
        df = df.sort_values("date")
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df = df.set_index("date")

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Error loading {country}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_all_countries():
    """Load all available country datasets."""
    result = {}
    for country in COUNTRIES:
        df = load_country(country)
        if df is not None:
            result[country] = df
    return result


def get_available_countries():
    """Return list of countries that have a CSV file in /data."""
    available = []
    for country, filename in COUNTRIES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            available.append(country)
    return available


def validate_dataframe(df, country):
    """Validate a dataframe and return a report."""
    report = {
        "country": country,
        "row_count": len(df),
        "date_range": (df.index.min(), df.index.max()) if len(df) > 0 else (None, None),
        "missing_cols": [],
        "missing_values": {},
        "issues": [],
    }

    for col in REQUIRED_COLUMNS:
        if col != "date" and col not in df.columns:
            report["missing_cols"].append(col)

    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            pct = round(n_missing / len(df) * 100, 1)
            report["missing_values"][col] = {"count": n_missing, "pct": pct}

    if report["missing_cols"]:
        report["issues"].append(f"Missing columns: {', '.join(report['missing_cols'])}")
    if len(df) < 60:
        report["issues"].append("Less than 60 observations — models may be unreliable")
    if report["missing_values"]:
        high_missing = [c for c, v in report["missing_values"].items() if v["pct"] > 20]
        if high_missing:
            report["issues"].append(f"High missing values (>20%): {', '.join(high_missing)}")

    return report


def compute_returns(df, col):
    """Compute log returns for a given column."""
    import math
    return (df[col] / df[col].shift(1)).apply(
        lambda x: x if pd.isna(x) else math.log(x))


def sidebar_country_selector(default="Morocco"):
    """Render a country selector in the sidebar."""
    available = get_available_countries()
    if not available:
        st.sidebar.warning("No country data found. Please add CSV files to /data.")
        return default
    idx = available.index(default) if default in available else 0
    return st.sidebar.selectbox("🌍 Select Country", available, index=idx)
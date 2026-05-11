# PFE App — Morocco FX & IT Readiness

A Decision Support Application for Analyzing Morocco's Exchange Rate Dynamics
and Evaluating BAM's Readiness for Inflation Targeting.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your data

Place your country CSV files in the `/data` folder:

```
data/
  morocco.csv        ← required
  chile.csv          ← optional (comparison)
  turkey.csv         ← optional (comparison)
  romania.csv        ← optional (comparison)
  peru.csv           ← optional (comparison)
  czech_republic.csv ← optional (comparison)
```

Each CSV must have these columns (monthly frequency):

| Column       | Description                  | Example   |
|--------------|------------------------------|-----------|
| date         | YYYY-MM-DD                   | 2010-01-01|
| cpi          | CPI index                    | 112.4     |
| policy_rate  | Central bank rate (%)        | 3.00      |
| neer         | Nominal effective exch. rate | 97.2      |
| reer         | Real effective exch. rate    | 98.5      |
| fx_usd       | Local currency per USD       | 10.05     |
| fx_eur       | Local currency per EUR       | 10.89     |
| reserves     | FX reserves (USD billions)   | 31.2      |
| oil_brent    | Brent crude price (USD)      | 82.4      |
| output_gap   | Output gap (% of potential)  | -0.3      |

A sample `morocco.csv` with placeholder data is included.
**Replace it with your real data before running analysis.**

### 3. Run the app

```bash
streamlit run app.py
```

The app will open automatically at http://localhost:8501

---

## Pages

| Page | Name | Description |
|------|------|-------------|
| 0 | Data Hub | Load and validate datasets |
| 1 | Exchange Rate Diagnostics | Historical dynamics, structural breaks |
| 2 | Volatility Modeling | GARCH framework |
| 3 | Risk & Simulation | Monte Carlo, VaR, CVaR |
| 4 | ERPT Analysis | VAR/BVAR, IRF, FEVD |
| 5 | IT Readiness | Composite index, benchmarking |
| 6 | Executive Summary | Key findings, PDF export |

---

## Data Sources

| Variable | Recommended Source |
|----------|--------------------|
| CPI | IMF IFS / World Bank |
| Policy rate | BIS / Central bank websites |
| NEER / REER | BIS (bis.org/statistics) |
| FX rates | FRED / investing.com |
| Reserves | IMF IFS |
| Oil price | FRED (DCOILBRENTEU) |
| Output gap | IMF WEO |

---

## Project structure

```
pfe_app/
├── app.py                  ← Main entry point
├── requirements.txt
├── README.md
├── data/
│   ├── morocco.csv         ← Sample data (replace with real)
│   └── ...                 ← Add other countries here
├── pages/
│   ├── 0_Data_Hub.py
│   ├── 1_Exchange_Rate_Diagnostics.py
│   ├── 2_Volatility_Modeling.py
│   ├── 3_Risk_Simulation.py
│   ├── 4_ERPT_Analysis.py
│   ├── 5_IT_Readiness.py
│   └── 6_Executive_Summary.py
└── utils/
    ├── __init__.py
    └── data_loader.py      ← Shared data loading functions
```

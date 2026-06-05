"""
Shared visual design system for the PFE Streamlit application.
Import and call apply_global_style() at the top of every page for a
consistent, professional look across the whole app.
"""
import streamlit as st


# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2563EB",   # blue   — exchange rate / main accent
    "danger":    "#dc2626",   # red    — CPI / risk
    "success":   "#16a34a",   # green  — positive / ready
    "warning":   "#f59e0b",   # amber  — caution
    "purple":    "#7c3aed",
    "morocco":   "#C1272D",   # Moroccan flag red
    "morocco_g": "#006233",   # Moroccan flag green
    "ink":       "#0B1120",   # deep background
    "surface":   "#151B2B",   # card background
    "border":    "rgba(255,255,255,0.08)",
    "muted":     "#94A3B8",
}

# Plotly template to keep every chart visually consistent
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#E2E8F0", size=12),
    margin=dict(l=10, r=10, t=50, b=10),
    xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    colorway=["#2563EB", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#0891b2"],
)


def apply_global_style():
    """Inject the global CSS. Call once at the top of every page."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* ── App background: subtle gradient ─────────────────────────────── */
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 600px at 80% -10%, rgba(37,99,235,0.08), transparent 60%),
                radial-gradient(900px 500px at 0% 10%, rgba(124,58,237,0.06), transparent 55%),
                #0B1120;
            min-height: 100vh;
        }

        /* ── Headings ─────────────────────────────────────────────────────── */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.02em !important;
            background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em !important; }

        /* ── Metric cards ─────────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(30,38,59,0.85), rgba(18,24,40,0.85));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(37,99,235,0.4);
        }
        [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 500 !important; }
        [data-testid="stMetricValue"] { font-weight: 700 !important; letter-spacing: -0.02em; }

        /* ── Tabs ─────────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            font-weight: 600;
            transition: background 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { background: rgba(37,99,235,0.08); }
        .stTabs [aria-selected="true"] {
            background: rgba(37,99,235,0.12);
            border-bottom: 2px solid #2563EB !important;
        }

        /* ── Dataframes ───────────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.06);
        }

        /* ── Buttons ──────────────────────────────────────────────────────── */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(37,99,235,0.25);
        }

        /* ── Alerts / info boxes ──────────────────────────────────────────── */
        [data-testid="stAlert"] { border-radius: 12px; }

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1422 0%, #0B1120 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* ── Sidebar navigation links (all blocks, cool colors) ───────────── */
[data-testid="stSidebarNav"] {
    padding-top: 4px;
}
[data-testid="stSidebarNav"] ul {
    gap: 4px;
    margin: 0;
}
[data-testid="stSidebarNav"] li {
    margin: 0 !important;
}
/* Every nav item gets a block background */
[data-testid="stSidebarNav"] a {
    border-radius: 10px;
    padding: 8px 14px !important;
    margin: 2px 8px !important;
    min-height: 0 !important;
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.10);
    transition: all 0.18s ease;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(37, 99, 235, 0.18) !important;
    border-color: rgba(96, 165, 250, 0.4);
    transform: translateX(3px);
}
/* Cool slate-blue text for inactive items */
[data-testid="stSidebarNav"] a span {
    font-weight: 500;
    font-size: 0.9rem;
    line-height: 1.2 !important;
    color: #A5B4CB !important;
}
[data-testid="stSidebarNav"] a:hover span {
    color: #DBEAFE !important;
}
/* Active page — vibrant gradient + glow */
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(135deg, #2563EB 0%, #6D28D9 100%) !important;
    border: 1px solid rgba(147, 197, 253, 0.5);
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
    font-weight: 700;
    color: #FFFFFF !important;
}
[data-testid="stSidebarNav"]::before {
    content: "NAVIGATION";
    display: block;
    padding: 4px 16px 6px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #60A5FA;
}

        /* ── Hide Streamlit chrome & bottom band artifacts ────────────────── */
        #MainMenu { display: none !important; }
        footer { display: none !important; }
        .stApp > footer { display: none !important; }
        [data-testid="stBottom"] { display: none !important; }
        [data-testid="stBottomBlockContainer"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }

        /* ── Custom hero/section components ───────────────────────────────── */
        .hero {
            background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 32px 36px;
            margin-bottom: 24px;
        }
        .hero-title { font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
        .hero-sub   { color: #94A3B8; font-size: 1.05rem; margin-top: 8px; }

        .pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .pill-blue  { background: rgba(37,99,235,0.15);  color: #93C5FD; border:1px solid rgba(37,99,235,0.3);}
        .pill-green { background: rgba(22,163,74,0.15);  color: #86EFAC; border:1px solid rgba(22,163,74,0.3);}
        .pill-amber { background: rgba(245,158,11,0.15); color: #FCD34D; border:1px solid rgba(245,158,11,0.3);}

        .feature-card {
            background: linear-gradient(145deg, rgba(30,38,59,0.6), rgba(18,24,40,0.6));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 20px 22px;
            height: 100%;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }
        .feature-card:hover { transform: translateY(-3px); border-color: rgba(37,99,235,0.35); }
        .feature-icon { font-size: 1.6rem; margin-bottom: 8px; }
        .feature-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
        .feature-desc  { color: #94A3B8; font-size: 0.9rem; line-height: 1.4; }
    </style>
    """, unsafe_allow_html=True)


def style_plotly(fig, height=None):
    """Apply the consistent Plotly template to any figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    if height:
        fig.update_layout(height=height)
    return fig


def section_header(title, subtitle=None):
    """Render a styled section header."""
    sub = f'<div class="hero-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin: 8px 0 20px 0;">
        <div style="font-size:1.4rem;font-weight:700;letter-spacing:-0.01em;">{title}</div>
        {sub}
    </div>
    """, unsafe_allow_html=True)
"""
Retail Performance Intelligence Dashboard
==========================================
Portfolio project — Data Analyst / Business Insights Analyst.

Author : Varun Mehta
GitHub : github.com/yourname

Demonstrates:
  - Data wrangling & pipeline design  (Pandas, NumPy)
  - KPI framework design              (Revenue, Margin, Variance, Returns)
  - Statistical forecasting           (Linear Regression / sklearn)
  - Executive data storytelling       (Plotly, Streamlit)
  - Performance-optimized Python      (vectorized ops, Streamlit caching)

Folder structure:
  project-root/
  ├── data/
  │   └── sales_data.csv
  └── App/
      └── app.py

Run:  streamlit run App/app.py
"""

import os
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  AUTHOR — update this
# ══════════════════════════════════════════════════════════════
AUTHOR_NAME  = "Varun Mehta"
AUTHOR_TITLE = "Data Analyst"

# ══════════════════════════════════════════════════════════════
#  DESIGN TOKENS — deep navy + vivid accents, high contrast
# ══════════════════════════════════════════════════════════════
C_BG       = "#05080F"   # near-black page bg
C_PANEL    = "#080D18"   # sidebar bg
C_CARD     = "#0C1526"   # KPI & insight card bg
C_CARD2    = "#0F1C32"   # slightly lighter card
C_BORDER   = "#1A2E4A"   # all borders
C_TEXT     = "#EEF4FF"   # primary text — almost white
C_SUB      = "#93B4D4"   # secondary text — clear blue-grey
C_MUTED    = "#4D6E8F"   # labels / muted
C_BLUE     = "#4D9FFF"   # primary accent — bright sky blue
C_AMBER    = "#FFB020"   # highlight — warm amber
C_GREEN    = "#2ECC97"   # on-track / positive
C_RED      = "#FF6B6B"   # alert / negative
C_TEAL     = "#18D4E8"   # tertiary
C_PURPLE   = "#9B7FFF"   # quaternary
C_DIVIDER  = "#162844"   # section rule

PALETTE    = [C_BLUE, C_AMBER, C_TEAL, C_PURPLE, C_GREEN, "#FF7EB3"]

# ── Plotly chart defaults ──────────────────────────────────────
CHART_BASE = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter, sans-serif", color=C_SUB, size=12),
    title_font    = dict(family="Inter, sans-serif", color=C_TEXT, size=15, ),
    xaxis         = dict(
        gridcolor=C_DIVIDER, linecolor=C_BORDER,
        tickcolor=C_BORDER,  tickfont=dict(color=C_SUB, size=11),
        zeroline=False, showgrid=True,
    ),
    yaxis         = dict(
        gridcolor=C_DIVIDER, linecolor=C_BORDER,
        tickcolor=C_BORDER,  tickfont=dict(color=C_SUB, size=11),
        zeroline=False, showgrid=True,
    ),
    legend        = dict(
        bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER,
        font=dict(color=C_SUB, size=12), orientation="h",
        yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    margin        = dict(l=10, r=10, t=52, b=10),
    colorway      = PALETTE,
    hovermode     = "x unified",
    hoverlabel    = dict(
        bgcolor=C_CARD2, bordercolor=C_BORDER,
        font=dict(color=C_TEXT, size=12),
    ),
)

def T(fig, height: int = 380):
    """Apply dark theme + consistent height to any Plotly figure."""
    fig.update_layout(**CHART_BASE, height=height)
    return fig

GROWTH_TARGETS = {
    "Laptops": 0.08, "Accessories": 0.05, "Appliances": 0.10,
    "Mobile Phones": 0.06, "TVs": 0.07, "Gaming": 0.09,
}

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline section-heading helper — avoids CSS class injection ──
def sec(label: str) -> None:
    """
    Render a styled section heading using only inline CSS.
    This avoids the 'undefined' bug caused by custom CSS classes
    failing to inject inside Streamlit tab iframes.
    """
    st.markdown(
        f"""<div style="
            display:flex;align-items:center;gap:12px;
            margin:28px 0 14px;
            font-size:11px;font-weight:700;
            text-transform:uppercase;letter-spacing:0.15em;
            color:{C_MUTED};font-family:'Inter',sans-serif;">
          {label}
          <div style="flex:1;height:1px;background:{C_DIVIDER};"></div>
        </div>""",
        unsafe_allow_html=True,
    )

def insight_card(text: str, tone: str = "info") -> None:
    """Render a bordered insight card — tone: pos | neg | warn | info"""
    border = {
        "pos":  C_GREEN,
        "neg":  C_RED,
        "warn": C_AMBER,
        "info": C_BLUE,
    }.get(tone, C_BLUE)
    st.markdown(
        f"""<div style="
            background:{C_CARD};border:1px solid {C_BORDER};
            border-left:3px solid {border};border-radius:6px;
            padding:12px 16px;margin-bottom:8px;
            font-size:13px;line-height:1.72;color:{C_SUB};
            font-family:'Inter',sans-serif;">
          {text}
        </div>""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ───────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background: {C_BG} !important;
    color: {C_TEXT} !important;
}}
.stApp {{ background: {C_BG} !important; }}

/* Streamlit Cloud adds a fixed ~50px toolbar at the top.
   4.5rem clears it on both local and deployed. */
.block-container {{
    padding-top: 4.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}}

/* Style the top toolbar to match dark theme */
header[data-testid="stHeader"] {{
    background: {C_BG} !important;
    border-bottom: 1px solid {C_BORDER} !important;
}}

/* Hide "Made with Streamlit" footer */
footer {{ visibility: hidden !important; }}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background: {C_PANEL} !important;
    border-right: 1px solid {C_BORDER} !important;
    min-width: 260px !important;
    max-width: 260px !important;
    width: 260px !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    width: 260px !important;
    padding: 1rem 1rem 2rem 1rem !important;
    box-sizing: border-box !important;
    overflow-x: hidden !important;
}}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{ color: {C_TEXT} !important; }}

[data-testid="stSidebar"] label p {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: {C_MUTED} !important;
    margin-bottom: 5px !important;
}}

/* Selectbox — constrain fully inside sidebar, wrap long labels */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {{
    width: 100% !important;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {{
    width: 100% !important;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {{
    width: 100% !important;
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 6px !important;
    color: {C_TEXT} !important;
    font-size: 12px !important;
    box-sizing: border-box !important;
}}
/* The selected value text inside the box */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] span {{
    font-size: 12px !important;
    color: {C_TEXT} !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 180px !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] {{
    width: 100% !important;
    background: {C_CARD} !important;
    border-color: {C_BORDER} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] * {{
    background: {C_CARD} !important;
    color: {C_TEXT} !important;
    font-size: 12px !important;
    box-sizing: border-box !important;
}}

/* Multiselect tags — full names, no truncation */
[data-testid="stSidebar"] [data-baseweb="tag"] {{
    background: {C_CARD2} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 4px !important;
    max-width: 100% !important;
    height: auto !important;
    padding: 3px 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="tag"] span {{
    font-size: 11px !important;
    color: {C_TEXT} !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    line-height: 1.5 !important;
}}

/* ── KPI Metric Cards ───────────────────────────────────────── */
[data-testid="metric-container"] {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
    padding: 20px 18px 16px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: border-color 0.18s ease !important;
}}
[data-testid="metric-container"]:hover {{ border-color: {C_BLUE} !important; }}
[data-testid="metric-container"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {C_BLUE} 0%, {C_AMBER} 60%, {C_TEAL} 100%);
}}
[data-testid="stMetricLabel"] p {{
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    color: {C_MUTED} !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 0 6px 0 !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: {C_TEXT} !important;
    line-height: 1.2 !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    margin-top: 3px !important;
}}

/* ── Tabs ───────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {C_BORDER} !important;
    gap: 2px !important;
}}
button[data-baseweb="tab"] {{
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 9px 16px !important;
    color: {C_MUTED} !important;
    background: transparent !important;
    border-radius: 0 !important;
}}
button[data-baseweb="tab"]:hover {{ color: {C_SUB} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {C_BLUE} !important;
    border-bottom: 2px solid {C_BLUE} !important;
}}

/* ── Status strips ──────────────────────────────────────────── */
.st-strip-ok  {{
    padding:10px 16px; border-radius:6px; font-size:13px; font-weight:500;
    background:rgba(46,204,151,.09); border:1px solid {C_GREEN}; color:{C_GREEN};
    display:flex; align-items:center; gap:8px; margin:8px 0 20px;
}}
.st-strip-bad {{
    padding:10px 16px; border-radius:6px; font-size:13px; font-weight:500;
    background:rgba(255,107,107,.09); border:1px solid {C_RED}; color:{C_RED};
    display:flex; align-items:center; gap:8px; margin:8px 0 20px;
}}

/* ── Misc ───────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:{C_BG}; }}
::-webkit-scrollbar-thumb {{ background:{C_BORDER}; border-radius:2px; }}
hr {{ border-color:{C_DIVIDER} !important; margin:18px 0 !important; }}
.stRadio label   {{ color:{C_SUB} !important; font-size:13px !important; }}
[data-testid="stDataFrame"] {{ border:1px solid {C_BORDER} !important; border-radius:6px !important; }}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {{ color:{C_MUTED} !important; font-size:11px !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA LOADING  — cached, vectorized
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    here = os.path.abspath(__file__)
    path = None
    for _ in range(6):
        here = os.path.dirname(here)
        candidate = os.path.join(here, "data", "sales_data.csv")
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        st.error("❌ Cannot find `data/sales_data.csv`.")
        st.stop()

    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    df.columns = (df.columns.str.strip()
                             .str.lower()
                             .str.replace(r"[\s\-]+", "_", regex=True))

    required = {"date","revenue","cost","units_sold","category","store_id","channel","product_name"}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"❌ Missing columns: {missing}  |  Found: {list(df.columns)}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "return_units" not in df.columns:
        df["return_units"] = 0
    df["return_units"] = pd.to_numeric(df["return_units"], errors="coerce").fillna(0)

    # Vectorized growth target — no row-by-row apply()
    df["target_revenue"] = df["revenue"] * (1 + df["category"].map(GROWTH_TARGETS).fillna(0.05))
    df["profit"]         = df["revenue"] - df["cost"]
    df["margin"]         = np.where(df["revenue"] > 0, df["profit"] / df["revenue"], np.nan)
    df["return_rate"]    = np.where(df["units_sold"] > 0,
                                    df["return_units"] / df["units_sold"], 0)
    df["variance"]       = df["revenue"] - df["target_revenue"]
    df["variance_pct"]   = np.where(df["target_revenue"] > 0,
                                    df["variance"] / df["target_revenue"], 0)

    period_col      = df["date"].dt.to_period("M")
    df["month"]     = period_col.astype(str)
    df["month_num"] = period_col.apply(lambda p: p.ordinal)
    df["quarter"]   = "Q" + df["date"].dt.quarter.astype(str) + " " + df["date"].dt.year.astype(str)
    df["year"]      = df["date"].dt.year

    return df


df       = load_data()
DATA_MIN = df["date"].min().normalize()
DATA_MAX = df["date"].max().normalize()


# ══════════════════════════════════════════════════════════════
#  CACHED FILTER + AGGREGATIONS
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_filtered(sd, ed, stores, cats, channels):
    # FIX 1b: include entire end day — add 1 day to ed boundary
    mask = (
        (df["date"] >= pd.Timestamp(sd)) &
        (df["date"] <  pd.Timestamp(ed) + pd.Timedelta(days=1)) &
        (df["store_id"].isin(stores)) &
        (df["category"].isin(cats)) &
        (df["channel"].isin(channels))
    )
    fdf = df.loc[mask].copy()
    if fdf.empty:
        return fdf, {}, {}

    rev    = fdf["revenue"].sum()
    profit = fdf["profit"].sum()
    units  = fdf["units_sold"].sum()
    target = fdf["target_revenue"].sum()
    ret    = fdf["return_units"].sum()

    K = dict(
        revenue      = rev,
        profit       = profit,
        units        = int(units),
        margin       = profit / rev if rev else 0,
        target       = target,
        variance     = rev - target,
        variance_pct = (rev - target) / target if target else 0,
        returns      = int(ret),
        return_rate  = ret / units if units else 0,
    )

    # ── Pre-compute all aggregations once ──────────────────────
    cat_df = (fdf.groupby("category")
                 .agg(revenue=("revenue","sum"), profit=("profit","sum"),
                      units=("units_sold","sum"), returns=("return_units","sum"),
                      target=("target_revenue","sum"))
                 .reset_index())
    cat_df["margin"]       = cat_df["profit"]  / cat_df["revenue"]
    cat_df["return_rate"]  = cat_df["returns"] / cat_df["units"]
    cat_df["variance_pct"] = (cat_df["revenue"] - cat_df["target"]) / cat_df["target"]

    store_df = (fdf.groupby("store_id")
                   .agg(revenue=("revenue","sum"), profit=("profit","sum"),
                        units=("units_sold","sum"), target=("target_revenue","sum"),
                        returns=("return_units","sum"))
                   .reset_index()
                   .sort_values("revenue", ascending=False))
    store_df["margin"]       = store_df["profit"]   / store_df["revenue"]
    store_df["variance_pct"] = (store_df["revenue"] - store_df["target"]) / store_df["target"]
    store_df["return_rate"]  = store_df["returns"]  / store_df["units"]

    ch_df = (fdf.groupby("channel")
                .agg(revenue=("revenue","sum"), profit=("profit","sum"),
                     units=("units_sold","sum"), returns=("return_units","sum"))
                .reset_index())
    ch_df["margin"]      = ch_df["profit"]  / ch_df["revenue"]
    ch_df["return_rate"] = ch_df["returns"] / ch_df["units"]

    trend_df = (fdf.groupby("month")
                   .agg(revenue=("revenue","sum"), profit=("profit","sum"))
                   .reset_index().sort_values("month"))
    trend_df["rolling_3mo"] = trend_df["revenue"].rolling(3, min_periods=1).mean()

    monthly_store = (fdf.groupby(["month","store_id"])["revenue"]
                        .sum().reset_index().sort_values("month"))
    monthly_chan  = (fdf.groupby(["month","channel"])["revenue"]
                        .sum().reset_index().sort_values("month"))

    aggs = dict(
        cat_df=cat_df, store_df=store_df, ch_df=ch_df,
        trend_df=trend_df, monthly_store=monthly_store, monthly_chan=monthly_chan,
    )
    return fdf, K, aggs


@st.cache_data(show_spinner=False)
def to_csv_bytes(sd, ed, stores, cats, channels):
    fdf, _, _ = get_filtered(sd, ed, stores, cats, channels)
    return fdf.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════
#  STRATEGIC PERIOD HELPER
# ══════════════════════════════════════════════════════════════
def period_bounds(period: str, dmin: date, dmax: date):
    today = dmax
    if period == "Last 7 Days":
        return today - timedelta(days=6), today
    if period == "Month to Date":
        return today.replace(day=1), today
    if period == "Last Full Month":
        lp = today.replace(day=1) - timedelta(days=1)
        return lp.replace(day=1), lp
    if period == "Quarter to Date":
        qsm = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=qsm, day=1), today
    if period == "Last Full Quarter":
        qsm = ((today.month - 1) // 3) * 3 + 1
        if qsm == 1:
            return date(today.year-1, 10, 1), date(today.year-1, 12, 31)
        lq_end   = today.replace(month=qsm, day=1) - timedelta(days=1)
        lq_start = lq_end.replace(month=lq_end.month - 2, day=1)
        return lq_start, lq_end
    if period == "Year to Date":
        return today.replace(month=1, day=1), today
    if period == "Last 12 Months":
        return today.replace(year=today.year-1) + timedelta(days=1), today
    if period == "Last Full Year":
        return date(today.year-1, 1, 1), date(today.year-1, 12, 31)
    return dmin, dmax  # Full Range

PERIODS = [
    "Full Range",           # FIX 1a: default is now Full Range (index=0)
    "Year to Date",
    "Last 12 Months",
    "Last Full Year",
    "Quarter to Date",
    "Last Full Quarter",
    "Month to Date",
    "Last Full Month",
    "Last 7 Days",
    "Custom Range",
]


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Branding block ───────────────────────────────────────
    st.markdown(f"""
    <div style="padding:12px 4px 20px;border-bottom:1px solid {C_BORDER};margin-bottom:16px">
      <div style="font-size:17px;font-weight:700;color:{C_TEXT};
                  letter-spacing:0.01em;line-height:1.25;font-family:'Inter',sans-serif">
        Retail Intelligence
      </div>
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.16em;
                  color:{C_MUTED};margin-top:3px;font-family:'Inter',sans-serif">
        Performance Dashboard
      </div>
      <div style="margin-top:12px;padding-top:10px;border-top:1px solid {C_BORDER}">
        <div style="font-size:12px;font-weight:600;color:{C_BLUE};
                    font-family:'Inter',sans-serif;letter-spacing:0.01em">
          {AUTHOR_NAME}
        </div>
        <div style="font-size:10px;color:{C_MUTED};margin-top:2px;
                    font-family:'Inter',sans-serif;letter-spacing:0.04em">
          {AUTHOR_TITLE}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Time Period ──────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{C_MUTED};margin-bottom:6px;'
        f'font-family:Inter,sans-serif">Time Period</div>',
        unsafe_allow_html=True,
    )
    sel_period = st.selectbox("Period", PERIODS, index=0,   # FIX 1a: default Full Range
                              label_visibility="collapsed")

    dmin = DATA_MIN.date()
    dmax = DATA_MAX.date()

    if sel_period == "Custom Range":
        cr = st.date_input("Custom dates", value=(dmin, dmax),
                           min_value=dmin, max_value=dmax,
                           label_visibility="collapsed")
        sd, ed = (cr if isinstance(cr, (list, tuple)) and len(cr) == 2
                  else (dmin, dmax))
    else:
        sd, ed = period_bounds(sel_period, dmin, dmax)
        sd, ed = max(sd, dmin), min(ed, dmax)
        st.caption(f"📅  {sd}  →  {ed}")

    st.markdown("<div style='margin:14px 0 0'></div>", unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{C_MUTED};margin-bottom:6px;'
        f'font-family:Inter,sans-serif">Filters</div>',
        unsafe_allow_html=True,
    )
    all_stores = sorted(df["store_id"].unique())
    all_cats   = sorted(df["category"].unique())
    all_chs    = sorted(df["channel"].unique())

    sel_stores = st.multiselect("Stores",     all_stores, default=all_stores)
    sel_cats   = st.multiselect("Categories", all_cats,   default=all_cats)
    sel_chs    = st.multiselect("Channels",   all_chs,    default=all_chs)

    st.markdown("<div style='margin:14px 0 0'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{C_MUTED};margin-bottom:6px;'
        f'font-family:Inter,sans-serif">Display</div>',
        unsafe_allow_html=True,
    )
    top_n = st.slider("Top N Products", 5, 20, 10)

    st.markdown("---")
    st.caption(f"Data range: {dmin} → {dmax}")
    st.caption(f"{len(df):,} total records loaded")
    st.markdown(f"""
    <div style="margin-top:12px;padding:11px 13px;background:{C_CARD};
                border:1px solid {C_BORDER};border-radius:7px;
                font-size:11px;color:{C_MUTED};line-height:1.95;
                font-family:'Inter',sans-serif">
      <div style="font-size:9px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.13em;color:{C_SUB};margin-bottom:5px">Tech Stack</div>
      Python &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; NumPy<br>
      Plotly &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; sklearn
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FILTER + AGGREGATE (single cached call)
# ══════════════════════════════════════════════════════════════
_stores = tuple(sel_stores or all_stores)
_cats   = tuple(sel_cats   or all_cats)
_chs    = tuple(sel_chs    or all_chs)

fdf, K, aggs = get_filtered(sd, ed, _stores, _cats, _chs)

if fdf.empty:
    st.warning("⚠️ No records match the selected filters. Adjust the sidebar.")
    st.stop()

cat_df        = aggs["cat_df"]
store_df      = aggs["store_df"]
ch_df         = aggs["ch_df"]
trend_df      = aggs["trend_df"]
monthly_store = aggs["monthly_store"]
monthly_chan  = aggs["monthly_chan"]


# ══════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════
period_label = f"{sel_period}  ·  {sd} → {ed}"
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;
            padding-bottom:16px;border-bottom:1px solid {C_BORDER};margin-bottom:22px">
  <div>
    <div style="font-size:25px;font-weight:700;color:{C_TEXT};
                line-height:1.15;font-family:'Inter',sans-serif;letter-spacing:-0.01em">
      Retail Performance Intelligence
    </div>
    <div style="font-size:12px;color:{C_MUTED};margin-top:5px;
                font-family:'Inter',sans-serif;letter-spacing:0.02em">
      {period_label}
      &nbsp;·&nbsp; {fdf["store_id"].nunique()} Stores
      &nbsp;·&nbsp; {fdf["category"].nunique()} Categories
      &nbsp;·&nbsp; {fdf["channel"].nunique()} Channels
    </div>
  </div>
  <div style="text-align:right;font-family:'JetBrains Mono',monospace;
              font-size:11px;color:{C_MUTED};line-height:2.1">
    <span style="color:{C_TEXT};font-weight:600">{len(fdf):,}</span> records<br>
    <span style="color:{C_TEXT};font-weight:600">{fdf["product_name"].nunique()}</span> products
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📋  Overview",
    "📦  Category · Product",
    "🏪  Store Analytics",
    "🔀  Channel Split",
    "🔥  Profitability",
    "📈  Forecast",
    "🧠  Insights",
])


# ──────────────────────────────────────────────────────────────
#  TAB 0 — EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    sec("Key Performance Indicators")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Revenue",      f"${K['revenue']:,.0f}")
    c2.metric("Gross Profit", f"${K['profit']:,.0f}")
    c3.metric("Gross Margin", f"{K['margin']:.1%}")
    c4.metric("Units Sold",   f"{K['units']:,}")
    c5.metric("vs Target",    f"{K['variance_pct']:+.1%}",
              delta=f"${K['variance']:+,.0f}", delta_color="normal")
    c6.metric("Return Rate",  f"{K['return_rate']:.1%}")

    ok      = K["variance_pct"] >= 0
    cls_str = "st-strip-ok" if ok else "st-strip-bad"
    icon    = "✅" if ok else "⚠️"
    msg     = "On Track — Exceeding Growth Target" if ok else "Below Target — Action Required"
    st.markdown(
        f'<div class="{cls_str}">{icon}&nbsp;'
        f'<strong>{msg}</strong>&nbsp;·&nbsp;'
        f'Variance: <strong>${K["variance"]:+,.0f}</strong>'
        f' ({K["variance_pct"]:+.1%} vs plan)</div>',
        unsafe_allow_html=True,
    )

    sec("Monthly Revenue & Profit Trend")
    fig_trend = go.Figure()
    fig_trend.add_bar(x=trend_df["month"], y=trend_df["revenue"],
                      name="Revenue", marker_color=C_BLUE, opacity=0.72)
    fig_trend.add_scatter(x=trend_df["month"], y=trend_df["profit"],
                          name="Profit", mode="lines+markers",
                          line=dict(color=C_AMBER, width=2.5), marker=dict(size=6))
    fig_trend.add_scatter(x=trend_df["month"], y=trend_df["rolling_3mo"],
                          name="3-Mo Avg", mode="lines",
                          line=dict(color=C_TEAL, width=1.5, dash="dot"))
    fig_trend.update_layout(barmode="overlay", xaxis_title="",
                            yaxis_title="Amount ($)", **CHART_BASE, height=360)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        sec("Quarterly Revenue")
        qtr = fdf.groupby("quarter")[["revenue","profit"]].sum().reset_index()
        fig_q = px.bar(qtr, x="quarter", y=["revenue","profit"], barmode="group",
                       text_auto=".2s", color_discrete_sequence=[C_BLUE, C_AMBER],
                       labels={"value":"Amount ($)","variable":"","quarter":""})
        T(fig_q, 320); st.plotly_chart(fig_q, use_container_width=True)

    with col_b:
        sec("Actual vs Target by Category")
        fig_ctgt = go.Figure()
        fig_ctgt.add_bar(x=cat_df["category"], y=cat_df["revenue"],
                         name="Actual", marker_color=C_BLUE)
        fig_ctgt.add_bar(x=cat_df["category"], y=cat_df["target"],
                         name="Target", marker_color=C_AMBER, opacity=0.45)
        fig_ctgt.update_layout(barmode="overlay", xaxis_title="",
                               yaxis_title="Revenue ($)", **CHART_BASE, height=320)
        st.plotly_chart(fig_ctgt, use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  TAB 1 — CATEGORY · PRODUCT
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    sec("Category Performance")
    c1, c2 = st.columns(2)
    with c1:
        fig_cbar = px.bar(
            cat_df.sort_values("revenue"), x="revenue", y="category",
            orientation="h", color="margin",
            color_continuous_scale=[[0,C_BORDER],[0.35,C_BLUE],[1,C_AMBER]],
            text_auto=".2s",
            labels={"revenue":"Revenue ($)","category":"","margin":"Margin %"},
            title="Revenue by Category  (colour = margin %)",
        )
        T(fig_cbar, 340); st.plotly_chart(fig_cbar, use_container_width=True)

    with c2:
        fig_cbub = px.scatter(
            cat_df, x="revenue", y="margin", size="units",
            color="category", text="category",
            color_discrete_sequence=PALETTE, size_max=55,
            labels={"revenue":"Revenue ($)","margin":"Margin %","units":"Units Sold"},
            title="Revenue vs Margin  (bubble = units sold)",
        )
        fig_cbub.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig_cbub.update_yaxes(tickformat=".0%")
        T(fig_cbub, 340); st.plotly_chart(fig_cbub, use_container_width=True)

    sec("Product Drill-Down")
    sel_cat = st.selectbox("Category to explore",
                           sorted(fdf["category"].unique()), key="drill")
    drilled = fdf[fdf["category"] == sel_cat]
    prod_df = (drilled.groupby("product_name")
                      .agg(revenue=("revenue","sum"), profit=("profit","sum"),
                           units=("units_sold","sum"), returns=("return_units","sum"))
                      .reset_index()
                      .sort_values("revenue", ascending=False)
                      .head(top_n))
    prod_df["margin"]      = prod_df["profit"]  / prod_df["revenue"]
    prod_df["return_rate"] = prod_df["returns"] / prod_df["units"]

    c3, c4 = st.columns(2)
    with c3:
        fig_prod = px.bar(
            prod_df.sort_values("revenue"), x="revenue", y="product_name",
            orientation="h", color="margin",
            color_continuous_scale=[[0,C_BORDER],[0.35,C_BLUE],[1,C_AMBER]],
            text_auto=".2s",
            labels={"product_name":"","revenue":"Revenue ($)","margin":"Margin %"},
            title=f"Top {top_n} Products — {sel_cat}",
        )
        T(fig_prod, 360); st.plotly_chart(fig_prod, use_container_width=True)

    with c4:
        fig_pret = px.scatter(
            prod_df, x="revenue", y="return_rate", size="units",
            color="margin", text="product_name",
            color_continuous_scale=[[0,C_BLUE],[1,C_AMBER]],
            labels={"revenue":"Revenue ($)","return_rate":"Return Rate","units":"Units"},
            title="Revenue vs Return Rate  (bubble = units)",
        )
        fig_pret.update_traces(textposition="top center", marker=dict(opacity=0.85))
        fig_pret.update_yaxes(tickformat=".0%")
        T(fig_pret, 360); st.plotly_chart(fig_pret, use_container_width=True)

    cat_trend = drilled.groupby("month")["revenue"].sum().reset_index().sort_values("month")
    fig_ctrend = px.area(cat_trend, x="month", y="revenue",
                         color_discrete_sequence=[C_BLUE],
                         labels={"month":"","revenue":"Revenue ($)"},
                         title=f"Monthly Revenue Trend — {sel_cat}")
    fig_ctrend.update_traces(fillcolor="rgba(77,159,255,0.10)", line_color=C_BLUE,
                             line_width=2)
    T(fig_ctrend, 300); st.plotly_chart(fig_ctrend, use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  TAB 2 — STORE ANALYTICS
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    sec("Store-Level Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_srev = px.bar(store_df, x="store_id", y="revenue", color="margin",
                          color_continuous_scale=[[0,C_BORDER],[0.5,C_BLUE],[1,C_AMBER]],
                          text_auto=".2s",
                          labels={"store_id":"","revenue":"Revenue ($)","margin":"Margin"},
                          title="Revenue by Store")
        T(fig_srev, 320); st.plotly_chart(fig_srev, use_container_width=True)

    with c2:
        fig_smg = px.bar(store_df, x="store_id", y="margin", text_auto=".1%",
                         color_discrete_sequence=[C_TEAL],
                         labels={"store_id":"","margin":"Margin %"},
                         title="Gross Margin % by Store")
        fig_smg.update_yaxes(tickformat=".0%")
        T(fig_smg, 320); st.plotly_chart(fig_smg, use_container_width=True)

    with c3:
        colors = [C_GREEN if v >= 0 else C_RED for v in store_df["variance_pct"]]
        fig_sv = go.Figure(go.Bar(
            x=store_df["store_id"], y=store_df["variance_pct"],
            marker_color=colors,
            text=[f"{v:+.1%}" for v in store_df["variance_pct"]],
            textposition="outside", textfont=dict(color=C_SUB, size=11),
        ))
        fig_sv.add_hline(y=0, line_color=C_MUTED, line_dash="dot", line_width=1)
        fig_sv.update_yaxes(tickformat=".0%")
        fig_sv.update_layout(title="Revenue vs Target (%)", **CHART_BASE, height=320)
        st.plotly_chart(fig_sv, use_container_width=True)

    sec("Store × Category Revenue Mix")
    sc = fdf.groupby(["store_id","category"])["revenue"].sum().reset_index()
    fig_sc = px.bar(sc, x="store_id", y="revenue", color="category",
                    barmode="stack", color_discrete_sequence=PALETTE, text_auto=".2s",
                    labels={"store_id":"","revenue":"Revenue ($)","category":"Category"},
                    title="Stacked Revenue — Store by Category")
    T(fig_sc, 340); st.plotly_chart(fig_sc, use_container_width=True)

    sec("Store Revenue Trends Over Time")
    fig_sm = px.line(monthly_store, x="month", y="revenue", color="store_id",
                     markers=True, color_discrete_sequence=PALETTE,
                     labels={"month":"","revenue":"Revenue ($)","store_id":"Store"},
                     title="Monthly Revenue by Store")
    T(fig_sm, 340); st.plotly_chart(fig_sm, use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  TAB 3 — CHANNEL SPLIT
# ──────────────────────────────────────────────────────────────
with tabs[3]:
    sec("Channel Comparison: In-Store vs Online")
    cols = st.columns(len(ch_df))
    for col, row in zip(cols, ch_df.itertuples()):
        col.metric(f"{row.channel} Revenue", f"${row.revenue:,.0f}",
                   delta=f"Margin {row.margin:.1%}")

    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(ch_df, names="channel", values="revenue",
                         color_discrete_sequence=[C_BLUE, C_AMBER],
                         hole=0.55, title="Revenue Share by Channel")
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13,
                              textfont_color=C_TEXT)
        fig_pie.update_layout(
            annotations=[dict(text="Channel", x=0.5, y=0.5,
                              font_size=12, showarrow=False, font_color=C_MUTED)],
            **CHART_BASE, height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_chb = px.bar(ch_df, x="channel", y=["revenue","profit","units"],
                         barmode="group", text_auto=".2s",
                         color_discrete_sequence=[C_BLUE, C_AMBER, C_TEAL],
                         labels={"value":"Value","variable":"","channel":""},
                         title="Channel KPI Comparison")
        T(fig_chb, 340); st.plotly_chart(fig_chb, use_container_width=True)

    sec("Channel Revenue Over Time")
    fig_chm = px.line(monthly_chan, x="month", y="revenue", color="channel",
                      markers=True, color_discrete_sequence=[C_BLUE, C_AMBER],
                      labels={"month":"","revenue":"Revenue ($)","channel":"Channel"},
                      title="Monthly Revenue by Channel")
    T(fig_chm, 320); st.plotly_chart(fig_chm, use_container_width=True)

    sec("Category Mix by Channel")
    chcat = fdf.groupby(["channel","category"])["revenue"].sum().reset_index()
    fig_chcat = px.bar(chcat, x="category", y="revenue", color="channel",
                       barmode="group", text_auto=".2s",
                       color_discrete_sequence=[C_BLUE, C_AMBER],
                       labels={"category":"","revenue":"Revenue ($)","channel":"Channel"},
                       title="Category Revenue by Channel")
    T(fig_chcat, 320); st.plotly_chart(fig_chcat, use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  TAB 4 — PROFITABILITY HEATMAP
# ──────────────────────────────────────────────────────────────
with tabs[4]:
    sec("Profitability Intelligence Heatmap")
    hm_metric = st.radio("Metric", ["Revenue","Profit","Margin %","Return Rate"],
                         horizontal=True, key="hm")
    col_map = {"Revenue":"revenue","Profit":"profit",
               "Margin %":"margin","Return Rate":"return_rate"}
    agg_map = {"Revenue":"sum","Profit":"sum",
               "Margin %":"mean","Return Rate":"mean"}
    mc, fn  = col_map[hm_metric], agg_map[hm_metric]

    pivot = (fdf.groupby(["category","store_id"])[mc]
                .agg(fn).unstack(fill_value=0))
    fmt = ".1%" if hm_metric in ("Margin %","Return Rate") else ".2s"
    fig_hm = px.imshow(
        pivot, text_auto=fmt, aspect="auto",
        color_continuous_scale=[[0,"#030810"],[0.4,C_BLUE],[1,C_AMBER]],
        labels={"x":"Store","y":"Category","color":hm_metric},
        title=f"{hm_metric} — Category × Store Heatmap",
    )
    fig_hm.update_layout(coloraxis_colorbar_title=hm_metric, **CHART_BASE, height=360)
    st.plotly_chart(fig_hm, use_container_width=True)

    sec("Product Profitability Matrix")
    pm = (fdf.groupby(["category","product_name"])
             .agg(revenue=("revenue","sum"), profit=("profit","sum"),
                  units=("units_sold","sum"))
             .reset_index())
    pm["margin"] = pm["profit"] / pm["revenue"]
    fig_pm = px.scatter(pm, x="revenue", y="margin", size="units",
                        color="category", hover_name="product_name",
                        color_discrete_sequence=PALETTE, size_max=45,
                        labels={"revenue":"Revenue ($)","margin":"Margin %",
                                "units":"Units","category":"Category"},
                        title="Product Revenue vs Margin  (bubble = units sold)")
    fig_pm.update_yaxes(tickformat=".0%")
    T(fig_pm, 400); st.plotly_chart(fig_pm, use_container_width=True)


# ──────────────────────────────────────────────────────────────
#  TAB 5 — FORECAST
# ──────────────────────────────────────────────────────────────
with tabs[5]:
    sec("Revenue Forecasting Engine")
    fc_cat   = st.selectbox("Forecast scope",
                            ["All Categories"] + sorted(fdf["category"].unique()),
                            key="fc_cat")
    n_months = st.slider("Forecast horizon (months)", 3, 12, 6, key="fc_n")

    fc_data = fdf if fc_cat == "All Categories" else fdf[fdf["category"] == fc_cat]
    monthly = (fc_data.groupby(["month","month_num"])["revenue"]
                      .sum().reset_index().sort_values("month_num"))

    if len(monthly) < 3:
        st.warning("Need ≥ 3 months of data to run a forecast.")
    else:
        X   = monthly[["month_num"]].values
        y   = monthly["revenue"].values
        mdl = LinearRegression().fit(X, y)
        r2  = mdl.score(X, y)

        last_num    = monthly["month_num"].max()
        fut_nums    = np.arange(last_num+1, last_num+n_months+1).reshape(-1,1)
        fut_rev     = mdl.predict(fut_nums)
        last_period = pd.Period(monthly["month"].iloc[-1], freq="M")
        fut_months  = [(last_period+i).strftime("%Y-%m") for i in range(1, n_months+1)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Model R²",             f"{r2:.3f}")
        k2.metric("Next Month Est.",      f"${fut_rev[0]:,.0f}")
        k3.metric(f"{n_months}-Mo Total", f"${fut_rev.sum():,.0f}")
        mom = (fut_rev[0] - y[-1]) / y[-1] if y[-1] else 0
        k4.metric("Projected MoM",        f"{mom:+.1%}")

        fig_fc = go.Figure()
        fig_fc.add_bar(x=monthly["month"], y=monthly["revenue"],
                       name="Actual", marker_color=C_BLUE, opacity=0.65)
        fig_fc.add_scatter(x=monthly["month"], y=mdl.predict(X),
                           name="Trend Fit", mode="lines",
                           line=dict(color=C_MUTED, width=1.5, dash="dot"))
        fig_fc.add_scatter(x=fut_months, y=fut_rev,
                           name="Forecast", mode="lines+markers",
                           line=dict(color=C_AMBER, width=2.5, dash="dash"),
                           marker=dict(size=8, symbol="diamond"))
        fig_fc.add_traces([
            go.Scatter(x=fut_months, y=fut_rev*1.10, mode="lines",
                       line=dict(width=0), showlegend=False),
            go.Scatter(x=fut_months, y=fut_rev*0.90, mode="lines",
                       fill="tonexty", fillcolor="rgba(255,176,32,0.08)",
                       line=dict(width=0), name="±10% Band"),
        ])
        fig_fc.update_layout(xaxis_title="", yaxis_title="Revenue ($)",
                             title=f"Revenue Forecast — {fc_cat}  (R² = {r2:.3f})",
                             **CHART_BASE, height=380)
        st.plotly_chart(fig_fc, use_container_width=True)

        fc_tbl = pd.DataFrame({
            "Month":            fut_months,
            "Forecast Revenue": [f"${v:,.0f}"   for v in fut_rev],
            "Low  (−10%)":      [f"${v*0.9:,.0f}" for v in fut_rev],
            "High (+10%)":      [f"${v*1.1:,.0f}" for v in fut_rev],
            "MoM Δ":            ["—"] + [f"{(fut_rev[i]-fut_rev[i-1])/fut_rev[i-1]:+.1%}"
                                          for i in range(1, len(fut_rev))],
        })
        st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

        insight_card(
            f"<strong>Model Note:</strong> OLS linear regression on {len(monthly)} months "
            f"(R² = {r2:.3f}). ±10% band is a simplified confidence interval. "
            f"For production: consider ARIMA, Prophet, or Holt-Winters for seasonality.",
            "info",
        )


# ──────────────────────────────────────────────────────────────
#  TAB 6 — INSIGHTS
# ──────────────────────────────────────────────────────────────
with tabs[6]:
    sec("Automated Business Insights")

    # All insights reuse pre-computed agg frames — zero extra groupbys
    vp = K["variance_pct"]
    insight_card(
        f'{"✅ Revenue exceeding" if vp >= 0 else "⚠️ Revenue trailing"} growth target by '
        f'<strong>{abs(vp):.1%}</strong> — <strong>${abs(K["variance"]):,.0f}</strong> '
        f'{"above" if vp >= 0 else "below"} plan.',
        "pos" if vp >= 0 else "neg",
    )

    best  = cat_df.loc[cat_df["margin"].idxmax()]
    worst = cat_df.loc[cat_df["margin"].idxmin()]
    insight_card(
        f'📦 <strong>{best["category"]}</strong> leads margin at <strong>{best["margin"]:.1%}</strong>. '
        f'<strong>{worst["category"]}</strong> is lowest at <strong>{worst["margin"]:.1%}</strong> '
        f'— review pricing or supplier costs.',
        "pos",
    )

    bs = store_df.loc[store_df["variance_pct"].idxmax()]
    ws = store_df.loc[store_df["variance_pct"].idxmin()]
    insight_card(
        f'🏪 Top store: <strong>{bs["store_id"]}</strong> at '
        f'<strong>{bs["variance_pct"]:+.1%}</strong> vs target. '
        f'Lowest: <strong>{ws["store_id"]}</strong> at '
        f'<strong>{ws["variance_pct"]:+.1%}</strong>.',
        "pos" if bs["variance_pct"] >= 0 else "warn",
    )

    if len(ch_df) >= 2:
        dom   = ch_df.loc[ch_df["revenue"].idxmax(), "channel"]
        share = ch_df.loc[ch_df["channel"]==dom,"revenue"].values[0] / ch_df["revenue"].sum()
        insight_card(
            f'🔀 <strong>{dom}</strong> drives <strong>{share:.0%}</strong> of revenue. '
            f'{"Scale online CX and digital acquisition." if dom=="Store" else "Online momentum is strong — ensure logistics scale."}',
            "info",
        )

    rr = K["return_rate"]
    insight_card(
        f'📦 Return rate: <strong>{rr:.1%}</strong>. '
        f'{"⚠️ Above 5% — audit top-return SKUs." if rr > 0.05 else "✅ Within acceptable range (< 5%)."}',
        "neg" if rr > 0.05 else "pos",
    )

    catmo = fdf.groupby(["month","category"])["revenue"].sum().reset_index().sort_values("month")
    if catmo["month"].nunique() >= 2:
        last2 = sorted(catmo["month"].unique())[-2:]
        m1 = catmo[catmo["month"]==last2[0]].set_index("category")["revenue"]
        m2 = catmo[catmo["month"]==last2[1]].set_index("category")["revenue"]
        growth = ((m2 - m1) / m1.replace(0, np.nan)).dropna().sort_values(ascending=False)
        if not growth.empty:
            insight_card(
                f'📈 Fastest MoM growth: <strong>{growth.index[0]}</strong> at '
                f'<strong>{growth.iloc[0]:+.1%}</strong> ({last2[0]} → {last2[1]}).',
                "pos",
            )

    tp = fdf.groupby("product_name")["revenue"].sum()
    insight_card(
        f'⭐ Top product: <strong>{tp.idxmax()}</strong> — <strong>${tp.max():,.0f}</strong> total.',
        "pos",
    )

    sec("Strategic Recommendations")
    for tone, text in [
        ("info", "🎯 <strong>Margin Expansion:</strong> Bundle accessories with Laptops and Mobile Phones. High-margin add-ons lift basket value without adding traffic cost."),
        ("info", "🏪 <strong>Store Ops:</strong> Document and replicate top-store playbooks. Structured knowledge transfer can close 60–70% of performance gaps."),
        ("info", "🔀 <strong>Channel Mix:</strong> If online lags, invest in digital shelf, SEO, and same-day delivery to capture the omnichannel demand shift."),
        ("warn", "📦 <strong>Returns:</strong> Root-cause high-return SKUs — improve product descriptions, add demo stations, or refine spec guidance."),
        ("info", "📈 <strong>Demand Planning:</strong> Align replenishment and staffing to the 6-month forecast to reduce stockouts and excess labour cost."),
        ("info", "📊 <strong>Next Steps:</strong> Layer CLV, NPS, and web-analytics data to extend the insight funnel beyond transactional metrics."),
    ]:
        insight_card(text, tone)


# ══════════════════════════════════════════════════════════════
#  FOOTER — EXPORT
# ══════════════════════════════════════════════════════════════
st.markdown("---")
col_l, col_r1, col_r2 = st.columns([3, 1, 1])

with col_l:
    st.caption(
        f"Retail Intelligence Dashboard  ·  {AUTHOR_NAME}  ·  "
        f"{len(fdf):,} records  ·  {sd} → {ed}  ·  "
        f"Python · Pandas · Plotly · Streamlit"
    )

with col_r1:
    st.download_button(
        "⬇  Filtered Data (CSV)",
        data=to_csv_bytes(sd, ed, _stores, _cats, _chs),
        file_name="retail_filtered_data.csv",
        mime="text/csv", use_container_width=True,
    )

with col_r2:
    summary = pd.DataFrame([
        {"Metric":"Total Revenue",    "Value": f"${K['revenue']:,.0f}"},
        {"Metric":"Gross Profit",     "Value": f"${K['profit']:,.0f}"},
        {"Metric":"Gross Margin",     "Value": f"{K['margin']:.1%}"},
        {"Metric":"Units Sold",       "Value": f"{K['units']:,}"},
        {"Metric":"Return Rate",      "Value": f"{K['return_rate']:.1%}"},
        {"Metric":"vs Target",        "Value": f"{K['variance_pct']:+.1%}"},
        {"Metric":"Revenue Variance", "Value": f"${K['variance']:+,.0f}"},
    ])
    st.download_button(
        "⬇  KPI Summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="retail_kpi_summary.csv",
        mime="text/csv", use_container_width=True,
    )
# trading_dashboard_optimized.py
# -*- coding: utf-8 -*-
import io
import json

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ===================== THEME (NeoTrader palette) =====================
# Backgrounds
BG_START = "#121522"  # NeoTrader deep navy
BG_END   = "#071a4a"  # subtle blue base

# Surfaces / borders / text
SURFACE_RGBA = "rgba(13, 18, 28, 0.78)"          # glassy card
SURFACE_ALT  = "rgba(17, 24, 39, 0.65)"
BORDER_RGBA  = "rgba(255, 255, 255, 0.08)"
TEXT_LIGHT   = "#e5e7eb"                          # light gray text
TEXT_MUTED   = "#9ca3af"                          # muted gray

# Brand accents
PRIMARY_A    = "#2457D6"  # deeper Neo blue
PRIMARY_B    = "#1FB8A6"  # quieter Neo teal
ACCENT_Y     = "#FFD91A"  # Neo yellow
ACCENT_M     = "#C026D3"  # Neo logo magenta

# Chart colorway (kept for defaults where no explicit colors are set)
COLORWAY = [ACCENT_Y, PRIMARY_B, PRIMARY_A, ACCENT_M, "#22c55e", "#f59e0b", "#ef4444", "#14b8a6"]

# Natural colors for "Advanced Analytics"
WIN_GREEN      = "#22c55e"
WIN_DARK_GREEN = "#166534"
LOSS_RED       = "#ef4444"
NEUTRAL_GRAY   = "#9ca3af"
POSITIVE_BLUE  = PRIMARY_A

# NEW: Enhanced color palettes for Advanced Analytics
ANALYTICS_GRADIENT_GREEN = ["#064e3b", "#065f46", "#047857", "#059669", "#10b981", "#34d399", "#6ee7b7"]
ANALYTICS_GRADIENT_BLUE = ["#071a4a", "#0b2a74", "#15358f", "#1d4ed8", "#2457D6", "#1f6fbc", "#188f9e"]
ANALYTICS_GRADIENT_PURPLE = ["#4a164f", "#6d1b76", "#92278f", "#C026D3", "#d946ef", "#e879f9", "#f0abfc"]
ANALYTICS_GRADIENT_ORANGE = ["#713f12", "#854d0e", "#a16207", "#ca8a04", "#eab308", "#FFD91A", "#fde047"]
ANALYTICS_PIE_COLORS = [ACCENT_Y, PRIMARY_B, PRIMARY_A, ACCENT_M, "#22c55e", "#f59e0b", "#ef4444", "#14b8a6"]

PNL_COL = "TOTAL_PNL"

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="NeoTrader Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- Global Plotly Defaults ---------------------------
go.layout.template.default = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_LIGHT),
        colorway=COLORWAY
    )
)

# --------------------------- Custom CSS (colors only) ---------------------------
st.markdown(f"""
<style>
    .main {{
        background: linear-gradient(135deg, {BG_START} 0%, {BG_END} 100%);
    }}

    /* CENTER THE PAGE CONTENT + CAP WIDTH */
    .main .block-container {{
        max-width: 1200px;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    .stMetric {{
        background: {SURFACE_RGBA};
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        border: 1px solid {BORDER_RGBA};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .stMetric:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
    }}
    .stMetric label {{
        color: {TEXT_MUTED} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: .6px;
    }}
    .stMetric [data-testid="stMetricValue"] {{
        font-size: 26px !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        color: {TEXT_LIGHT} !important;
        background: none !important;
        -webkit-text-fill-color: {TEXT_LIGHT} !important;
    }}
    .stMetric [data-testid="stMetricValue"] > div,
    .stMetric [data-testid="stMetricValue"] p {{
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }}

    /* Expanders / cards */
    div[data-testid="stExpander"] {{
        background: {SURFACE_ALT};
        backdrop-filter: blur(8px);
        border-radius: 14px;
        border: 1px solid {BORDER_RGBA};
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_END} 0%, {BG_START} 100%);
        border-right: 1px solid {BORDER_RGBA};
    }}

    /* Primary button = NeoTrader brand gradient */
    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY_A} 0%, {PRIMARY_B} 100%);
        color: #001018;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 700;
        transition: all .25s ease;
        box-shadow: 0 6px 18px rgba(36, 87, 214, 0.30);
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(45, 212, 191, 0.30);
        filter: brightness(1.02);
    }}

    /* Download button = teal ring */
    .stDownloadButton>button {{
        background: transparent;
        color: {TEXT_LIGHT};
        border: 1px solid {PRIMARY_B};
        border-radius: 10px;
        font-weight: 700;
    }}
    .stDownloadButton>button:hover {{
        background: rgba(45, 212, 191, 0.08);
        transform: translateY(-2px);
    }}

    /* Radios / labels */
    .stRadio > label {{
        color: {ACCENT_Y};
        background: none;
        -webkit-text-fill-color: {ACCENT_Y};
        font-weight: 800;
        font-size: 15px;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(16, 24, 39, 0.45);
        border-radius: 10px;
        padding: 5px;
        border: 1px solid {BORDER_RGBA};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 10px 18px;
        color: {TEXT_MUTED};
        font-weight: 700;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {PRIMARY_A} 0%, {PRIMARY_B} 100%);
        color: #031015;
    }}

    /* Dataframe wrapper */
    .stDataFrame {{
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
        border: 1px solid {BORDER_RGBA};
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: linear-gradient(180deg, {BG_START} 0%, {BG_END} 100%) !important;
        color: {TEXT_LIGHT};
    }}
    .main .block-container {{
        max-width: 1320px;
        padding-top: 24px;
        padding-bottom: 28px;
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        letter-spacing: 0 !important;
    }}
    .main h2 {{
        font-size: 24px !important;
        line-height: 1.2 !important;
        margin-bottom: 0 !important;
    }}
    .main h3 {{
        font-size: 18px !important;
        line-height: 1.25 !important;
    }}
    hr {{
        margin: 24px 0 !important;
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.18), transparent) !important;
    }}
    .stMetric {{
        background: rgba(15, 23, 42, 0.72) !important;
        padding: 16px 18px !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER_RGBA} !important;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22) !important;
        min-height: 124px;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }}
    .stMetric:hover {{
        transform: translateY(-2px);
        border-color: rgba(45, 212, 191, 0.28) !important;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.28) !important;
    }}
    .stMetric label {{
        font-size: 12px !important;
        letter-spacing: .4px !important;
    }}
    .stMetric [data-testid="stMetricValue"] {{
        font-size: 25px !important;
        color: {TEXT_LIGHT} !important;
        background: none !important;
        -webkit-text-fill-color: {TEXT_LIGHT} !important;
    }}
    .stMetric [data-testid="stMetricDelta"] {{
        font-size: 13px !important;
        font-weight: 700 !important;
        margin-top: 8px !important;
    }}
    div[data-testid="stExpander"] {{
        background: rgba(15, 23, 42, 0.62) !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER_RGBA} !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20) !important;
    }}
    div[data-testid="stExpander"] details summary {{
        color: {TEXT_LIGHT} !important;
        font-weight: 750 !important;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {BG_END} 0%, {BG_START} 100%) !important;
        border-right: 1px solid {BORDER_RGBA};
    }}
    [data-testid="stSidebar"] * {{
        color: {TEXT_LIGHT};
    }}
    .stButton>button,
    .stDownloadButton>button {{
        border-radius: 8px !important;
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }}
    .stButton>button {{
        box-shadow: 0 8px 18px rgba(36, 87, 214, 0.25) !important;
    }}
    .stButton>button:hover,
    .stDownloadButton>button:hover {{
        transform: translateY(-2px);
    }}
    .stRadio > label {{
        color: {ACCENT_Y} !important;
        -webkit-text-fill-color: {ACCENT_Y} !important;
        background: none !important;
        font-weight: 800;
    }}
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: rgba(15, 23, 42, 0.82) !important;
        border-color: rgba(148, 163, 184, 0.16) !important;
        border-radius: 8px !important;
        min-height: 42px;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {{
        color: {TEXT_LIGHT} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(15, 23, 42, 0.55) !important;
        border-radius: 8px !important;
        padding: 6px !important;
        border: 1px solid {BORDER_RGBA};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        padding: 10px 16px !important;
    }}
    .stDataFrame {{
        border-radius: 8px !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20) !important;
    }}
    div[data-testid="stPlotlyChart"] {{
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid {BORDER_RGBA};
        border-radius: 8px;
        padding: 10px 12px 4px 12px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        margin-bottom: 18px;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.14);
    }}
</style>
""", unsafe_allow_html=True)

# ---- Header with centered text (simplified) ----
with st.container():
    st.markdown(f"""
    <div style='
        width:100%;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        text-align:center;
        padding: 4px 0 18px 0;
    '>
        <h1 style='
            font-size: 40px;
            font-weight: 900;
            color:{TEXT_LIGHT};
            margin: 8px 0 6px 0;
        '>
            Ab Smart Trading <span style="color:{ACCENT_Y};">Hoga Easy</span>
        </h1>
        <div style='width: 118px; height: 2px; border-radius: 999px; background: linear-gradient(90deg, {ACCENT_Y}, {PRIMARY_B}, {PRIMARY_A}, {ACCENT_M});'></div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Google Drive Setup ---------------------------
def _read_secrets_or_fail():
    import streamlit as st
    present = list(st.secrets.keys())
    if "FOLDER_IDS" not in st.secrets:
        st.error(
            "st.secrets is missing [FOLDER_IDS]. Found sections: "
            + (", ".join(present) if present else "(none)")
            + "\n\nFix: create .streamlit/secrets.toml with [FOLDER_IDS] and [SERVICE_ACCOUNT_INFO] "
              "or add them in Streamlit Cloud > App settings > Secrets."
        )
        st.stop()
    if "SERVICE_ACCOUNT_INFO" not in st.secrets:
        st.error("st.secrets is missing [SERVICE_ACCOUNT_INFO]. Add that section too.")
        st.stop()

_read_secrets_or_fail()

FOLDER_IDS = {
    "Archive Data": st.secrets["FOLDER_IDS"].get("Archive Data"),
    "PnL Data": st.secrets["FOLDER_IDS"].get("PnL Data"),
}
missing = [k for k,v in FOLDER_IDS.items() if not v]
if missing:
    st.error(f"[FOLDER_IDS] is present but missing keys: {', '.join(missing)}")
    st.stop()

def _load_service_account_info():
    try:
        return dict(st.secrets["SERVICE_ACCOUNT_INFO"])
    except KeyError:
        st.error("SERVICE_ACCOUNT_INFO not found in secrets.")
        st.stop()

# --------------------------- Optimized Functions (PnL now uses capital-aware equity) ---------------------------
def build_pnl_equity_drawdown_df(
    df: pd.DataFrame,
    date_col: Optional[str],
    col: str = PNL_COL,
    initial_capital: float = 0.0
) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()

    required_cols = [col]
    if date_col and date_col in df.columns:
        required_cols.append(date_col)

    df_equity = df.dropna(subset=required_cols).copy()
    if df_equity.empty:
        return df_equity

    if date_col and date_col in df_equity.columns:
        df_equity[date_col] = pd.to_datetime(df_equity[date_col], errors='coerce')
        df_equity = df_equity.dropna(subset=[date_col]).sort_values(date_col, kind='mergesort')

    df_equity = df_equity.reset_index(drop=True)

    df_equity['PNL_num'] = pd.to_numeric(df_equity[col], errors='coerce').fillna(0.0)
    df_equity['Cumulative_PNL'] = df_equity['PNL_num'].cumsum()

    initial_capital = float(initial_capital or 0.0)
    if initial_capital > 0:
        df_equity['Equity'] = initial_capital + df_equity['Cumulative_PNL']
    else:
        df_equity['Equity'] = df_equity['Cumulative_PNL']

    df_equity['Running_Max_PNL'] = df_equity['Cumulative_PNL'].cummax()
    df_equity['Drawdown_PNL'] = df_equity['Cumulative_PNL'] - df_equity['Running_Max_PNL']
    df_equity['Running_Max_Equity'] = df_equity['Equity'].cummax()
    df_equity['Drawdown_Pct'] = 0.0

    positive_peak = df_equity['Running_Max_Equity'] > 0
    df_equity.loc[positive_peak, 'Drawdown_Pct'] = (
        df_equity.loc[positive_peak, 'Drawdown_PNL']
        / df_equity.loc[positive_peak, 'Running_Max_Equity']
        * 100.0
    )

    return df_equity


def calculate_metrics_pnl(df: pd.DataFrame, col: str = PNL_COL, date_col: Optional[str] = None) -> dict:
    metrics = {
        'total_trades': len(df),
        'total_pnl': 0.0,
        'winning_trades': 0,
        'losing_trades': 0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'max_drawdown': 0.0,
        'max_drawdown_pct': 0.0,
        'sharpe_ratio': 0.0,
        # Margin & capital metrics
        'total_margin': 0.0,
        'roi': 0.0,
        'initial_capital': 0.0,  # 👈 treated as Peak Margin Required for selected filters
    }
    if df is None or df.empty or col not in df.columns:
        return metrics

    # --- P&L series ---
    pnl_series = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    metrics['total_pnl'] = float(pnl_series.sum())
    metrics['winning_trades'] = int((pnl_series > 0).sum())
    metrics['losing_trades']  = int((pnl_series < 0).sum())
    tt = metrics['total_trades']
    metrics['win_rate'] = (metrics['winning_trades'] / tt * 100) if tt else 0.0

    wins   = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    if len(wins)   > 0: metrics['avg_win']  = float(wins.mean())
    if len(losses) > 0: metrics['avg_loss'] = float(abs(losses.mean()))

    gross_profit = float(wins.sum())
    gross_loss   = float(abs(losses.sum()))
    if gross_loss != 0:
        metrics['profit_factor'] = gross_profit / gross_loss

    # --- Peak Margin = max sum of margin across concurrently active trades ---
    # Use MARGIN_REQ for options (SCAN_NAME == "OPT 1"), intra_margin for others
    has_margin_req = 'MARGIN_REQ' in df.columns
    has_intra = 'intra_margin' in df.columns or 'INTRA_MARGIN' in df.columns
    intra_col = 'intra_margin' if 'intra_margin' in df.columns else ('INTRA_MARGIN' if 'INTRA_MARGIN' in df.columns else None)

    if has_margin_req or has_intra:
        if has_margin_req and has_intra and 'SCAN_NAME' in df.columns:
            # Per-row: OPT 1 (ROS) uses MARGIN_REQ, others use intra_margin
            is_opt = df['SCAN_NAME'].astype(str).str.strip().str.upper().isin(['OPT 1', 'ROS'])
            margin_vals = pd.Series(0.0, index=df.index)
            margin_vals[is_opt] = pd.to_numeric(df.loc[is_opt, 'MARGIN_REQ'], errors='coerce').fillna(0.0)
            margin_vals[~is_opt] = pd.to_numeric(df.loc[~is_opt, intra_col], errors='coerce').fillna(0.0)
        elif has_margin_req:
            margin_vals = pd.to_numeric(df['MARGIN_REQ'], errors='coerce').fillna(0.0)
        else:
            margin_vals = pd.to_numeric(df[intra_col], errors='coerce').fillna(0.0)

        # Determine signal/update date columns for overlap detection
        sig_col = next((c for c in df.columns if c.upper() in ('SIGNAL_DT', 'SIGNAL_DATE')), None)
        upd_col = next((c for c in df.columns if c.upper() in ('UPDATE_DT', 'UPDATE_DATE')), None)

        if sig_col and upd_col:
            sig_dt = pd.to_datetime(df[sig_col], errors='coerce')
            upd_dt = pd.to_datetime(df[upd_col], errors='coerce')

            # Build event list: +margin at signal, -margin at update
            events = []
            for margin, s, u in zip(margin_vals, sig_dt, upd_dt):
                if pd.notna(s) and pd.notna(u) and margin > 0:
                    events.append((s, margin))
                    events.append((u, -margin))

            if events:
                events.sort(key=lambda x: x[0])
                running_margin = 0.0
                peak_margin = 0.0
                for _, m in events:
                    running_margin += m
                    if running_margin > peak_margin:
                        peak_margin = running_margin
            else:
                peak_margin = 0.0
        else:
            # Fallback: no date columns found, use simple max
            peak_margin = float(margin_vals.max())

        metrics['total_margin'] = peak_margin
        metrics['initial_capital'] = peak_margin

        if peak_margin > 0:
            metrics['roi'] = (metrics['total_pnl'] / peak_margin) * 100.0

    # --- Max drawdown based on the same date-sorted curve used in the chart ---
    equity_df = build_pnl_equity_drawdown_df(
        df,
        date_col=date_col,
        col=col,
        initial_capital=metrics.get('initial_capital', 0.0)
    )
    if not equity_df.empty and 'Drawdown_PNL' in equity_df.columns:
        max_dd_idx = equity_df['Drawdown_PNL'].idxmin()
        max_dd_value = float(equity_df.loc[max_dd_idx, 'Drawdown_PNL'])
        metrics['max_drawdown'] = abs(max_dd_value)

        peak_equity_at_dd = float(equity_df.loc[max_dd_idx, 'Running_Max_Equity'])
        if peak_equity_at_dd > 0:
            metrics['max_drawdown_pct'] = abs(max_dd_value) / peak_equity_at_dd * 100.0

    # --- Sharpe (still based on raw P&L series) ---
    if len(pnl_series) > 1 and float(pnl_series.std()) != 0:
        metrics['sharpe_ratio'] = float((pnl_series.mean() / pnl_series.std()) * np.sqrt(252))

    return metrics


def style_chart(fig, title: Optional[str] = None, height: Optional[int] = None, showlegend: Optional[bool] = None):
    title_text = title
    if title_text is None and getattr(fig.layout, "title", None) and fig.layout.title.text:
        title_text = fig.layout.title.text

    layout_updates = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": dict(color=TEXT_LIGHT, size=12, family="Segoe UI, Arial, sans-serif"),
        "margin": dict(l=58, r=34, t=66, b=42),
        "hoverlabel": dict(
            bgcolor="rgba(15, 23, 42, 0.96)",
            bordercolor="rgba(148, 163, 184, 0.20)",
            font=dict(color=TEXT_LIGHT, size=12)
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=TEXT_MUTED)
        )
    }
    if height is not None:
        layout_updates["height"] = height
    if showlegend is not None:
        layout_updates["showlegend"] = showlegend
    if title_text:
        layout_updates["title"] = {
            "text": title_text,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 17, "color": TEXT_LIGHT}
        }

    fig.update_layout(**layout_updates)
    fig.update_xaxes(
        gridcolor="rgba(55, 65, 81, 0.35)",
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.16)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED, size=12)
    )
    fig.update_yaxes(
        gridcolor="rgba(55, 65, 81, 0.40)",
        zeroline=False,
        linecolor="rgba(148, 163, 184, 0.16)",
        tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED, size=12)
    )
    fig.update_traces(
        textfont=dict(color=TEXT_LIGHT, size=11),
        cliponaxis=False,
        selector=dict(type="bar")
    )
    fig.update_annotations(font=dict(color=TEXT_MUTED, size=12))
    return fig

@st.cache_resource
def get_drive_service():
    try:
        info = _load_service_account_info()
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"❌ Failed to authenticate with Google Drive: {str(e)}")
        return None

@st.cache_data(ttl=300, show_spinner=False)
def list_files_in_folder(folder_id: str) -> List[Tuple[str, str, str, str]]:
    service = get_drive_service()
    if not service:
        return []
    try:
        query = (
            f"'{folder_id}' in parents and trashed=false and ("
            "mimeType='text/csv' or "
            "mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or "
            "mimeType='application/vnd.ms-excel' or "
            "mimeType='application/vnd.google-apps.spreadsheet')"
        )
        fields = "nextPageToken, files(id, name, mimeType, modifiedTime)"
        files, page_token = [], None
        while True:
            res = service.files().list(
                q=query, fields=fields, orderBy="modifiedTime desc",
                pageToken=page_token, pageSize=1000,
                includeItemsFromAllDrives=True, supportsAllDrives=True,
            ).execute()
            files.extend(res.get('files', []))
            page_token = res.get('nextPageToken')
            if not page_token:
                break
        return [(f['id'], f['name'], f.get('modifiedTime', 'Unknown'), f.get('mimeType', '')) for f in files]
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
        return []

@st.cache_data(ttl=600, show_spinner=False)
def download_file_from_drive(file_id: str, file_name: str) -> Optional[bytes]:
    service = get_drive_service()
    if not service:
        return None
    try:
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
        mime = meta.get("mimeType", "")
        buf = io.BytesIO()

        if mime == "application/vnd.google-apps.spreadsheet":
            request = service.files().export(fileId=file_id, mimeType="text/csv")
        else:
            request = service.files().get_media(fileId=file_id)

        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.error(f"Error downloading file '{file_name}': {str(e)}")
        return None

def fix_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            name = str(col).lower().strip()
            if 'timeframe' in name:
                continue
            sample = df[col].dropna().head(100)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().mean() > 0.5:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process_file(file_id: str, filename: str) -> Optional[pd.DataFrame]:
    raw = download_file_from_drive(file_id, filename)
    if not raw:
        return None
    bio = io.BytesIO(raw)
    if filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(bio)
    else:
        try:
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio)

    df.columns = df.columns.str.strip()
    date_candidates: List[str] = []
    for col in df.columns:
        name = str(col).lower().strip()
        if 'timeframe' in name:
            continue
        if (
            name in ['date', 'time', 'timestamp'] or
            any(k in name for k in ['datetime', '_dt', 'signal_dt', 'update_dt', 'ts']) or
            name.endswith('_date') or name.endswith('_time')
        ):
            date_candidates.append(col)

    for col in date_candidates:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            pass

    df = fix_datetime_columns(df)
    return df

def precompute_filter_options_archive(df: pd.DataFrame) -> Dict:
    opts = {}
    opts['scan_names'] = ['All'] + (sorted(df['SCAN_NAME'].dropna().astype(str).unique()) if 'SCAN_NAME' in df.columns else [])
    opts['timeframes'] = ['All'] + (sorted(df['TIMEFRAME'].dropna().astype(str).unique()) if 'TIMEFRAME' in df.columns else [])
    opts['statuses'] = ['All'] + (sorted(df['STATUS'].dropna().astype(str).unique()) if 'STATUS' in df.columns else [])
    opts['alerts'] = ['All'] + (sorted(df['ALERT'].dropna().astype(str).unique()) if 'ALERT' in df.columns else [])
    opts['curated'] = ['All'] + (sorted(df['CURATED_SIGNAL'].dropna().astype(str).unique()) if 'CURATED_SIGNAL' in df.columns else [])

    date_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
    if not date_cols:
        for c in df.columns:
            if any(k in c.lower() for k in ['date', 'time', '_dt']) and pd.api.types.is_datetime64_any_dtype(df[c]):
                date_cols = [c]
                break
    if date_cols:
        dc = date_cols[0]
        vd = df[dc].dropna()
        if not vd.empty:
            opts['date_col'] = dc
            opts['min_date'] = vd.min().date()
            opts['max_date'] = vd.max().date()
    return opts

def precompute_filter_options_pnl(df: pd.DataFrame) -> Dict:
    opts = {}
    symcol = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in df.columns else 'SYMBOL'
    opts['symbols'] = ['All'] + (sorted(df[symcol].dropna().astype(str).unique()) if symcol in df.columns else [])
    opts['trades'] = ['All'] + (sorted(df['TRADE'].dropna().astype(str).unique()) if 'TRADE' in df.columns else [])
    opts['statuses'] = ['All'] + (sorted(df['STATUS'].dropna().astype(str).unique()) if 'STATUS' in df.columns else [])
    opts['scan_names'] = ['All'] + (sorted(df['SCAN_NAME'].dropna().astype(str).unique()) if 'SCAN_NAME' in df.columns else [])
    opts['curated'] = ['All'] + (sorted(df['CURATED_SIGNAL'].dropna().astype(str).unique()) if 'CURATED_SIGNAL' in df.columns else [])
    opts['fno'] = ['All'] + (sorted(df['FNO'].dropna().astype(str).unique()) if 'FNO' in df.columns else [])
    date_cols_named = [c for c in df.columns if any(k in c.upper() for k in ['SIGNAL_DT', 'UPDATE_DT', 'DATE'])]
    date_cols_named = [c for c in date_cols_named if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols_named:
        dc = date_cols_named[0]
        vd = df[dc].dropna()
        if not vd.empty:
            opts['date_col'] = dc
            opts['min_date'] = vd.min().date()
            opts['max_date'] = vd.max().date()
    return opts

def _norm_vals(vals):
    if vals is None:
        return []
    if not isinstance(vals, (list, tuple, set, pd.Series, np.ndarray)):
        vals = [vals]
    return [str(v).strip() for v in vals]

def _isin_str(df: pd.DataFrame, col: str, selected) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    left = df[col].astype(str).str.strip()
    right = set(_norm_vals(selected))
    return left.isin(right)

def apply_filters_archive(df: pd.DataFrame, scan, tf, status, alert, curated, start_date, end_date, date_col):
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    scan    = _norm_vals(scan)
    tf      = _norm_vals(tf)
    status  = _norm_vals(status)
    alert   = _norm_vals(alert)
    curated = _norm_vals(curated)

    if 'SCAN_NAME' in df.columns and scan and 'All' not in scan:
        mask &= _isin_str(df, 'SCAN_NAME', scan)
    if 'TIMEFRAME' in df.columns and tf and 'All' not in tf:
        mask &= _isin_str(df, 'TIMEFRAME', tf)
    if 'STATUS' in df.columns and status and 'All' not in status:
        col_norm = df['STATUS'].astype(str).str.upper().str.strip()
        mask &= col_norm.isin({s.upper() for s in status})
    if 'ALERT' in df.columns and alert and 'All' not in alert:
        mask &= _isin_str(df, 'ALERT', alert)
    if 'CURATED_SIGNAL' in df.columns and curated and 'All' not in curated:
        mask &= _isin_str(df, 'CURATED_SIGNAL', curated)

    if date_col and start_date and end_date and date_col in df.columns:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        col_dt = pd.to_datetime(df[date_col], errors='coerce')
        mask &= (col_dt >= start_dt) & (col_dt <= end_dt)
    return df[mask]

def apply_filters_pnl(df: pd.DataFrame, symbols, trades, statuses, scan_names, curated, fno, start_date, end_date, date_col):
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    symbols    = _norm_vals(symbols)
    trades     = _norm_vals(trades)
    statuses   = _norm_vals(statuses)
    scan_names = _norm_vals(scan_names)
    curated    = _norm_vals(curated)
    fno        = _norm_vals(fno)

    symcol = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in df.columns else 'SYMBOL'
    if symcol in df.columns and symbols and 'All' not in symbols:
        mask &= _isin_str(df, symcol, symbols)
    if 'TRADE' in df.columns and trades and 'All' not in trades:
        mask &= _isin_str(df, 'TRADE', trades)
    if 'STATUS' in df.columns and statuses and 'All' not in statuses:
        col_norm = df['STATUS'].astype(str).str.upper().str.strip()
        mask &= col_norm.isin({s.upper() for s in statuses})
    if 'SCAN_NAME' in df.columns and scan_names and 'All' not in scan_names:
        mask &= _isin_str(df, 'SCAN_NAME', scan_names)
    if 'CURATED_SIGNAL' in df.columns and curated and 'All' not in curated:
        mask &= _isin_str(df, 'CURATED_SIGNAL', curated)
    if 'FNO' in df.columns and fno and 'All' not in fno:
        mask &= _isin_str(df, 'FNO', fno)

    if date_col and start_date and end_date and date_col in df.columns:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        col_dt = pd.to_datetime(df[date_col], errors='coerce')
        mask &= (col_dt >= start_dt) & (col_dt <= end_dt)
    return df[mask]

def calculate_metrics_archive(df: pd.DataFrame) -> dict:
    metrics = {'total_trades': len(df)}
    if df is None or df.empty or 'STATUS' not in df.columns:
        metrics.update({
            'winning_trades': 0, 'losing_trades': 0, 'breakeven_trades': 0, 'hit_rate': 0,
            'target1_met': 0, 'target1_rate': 0, 'target2_met': 0, 'target2_rate': 0,
            'target3_met': 0, 'target3_rate': 0
        })
        return metrics
    s = df['STATUS'].astype(str).str.upper().str.strip()
    wins = s.str.contains('T1 MET|T2 MET|T3 MET', regex=True, na=False)
    losses = s.str.contains('SL MET', na=False)
    be = s.str.contains('LAPSED', na=False)
    metrics['winning_trades'] = int(wins.sum())
    metrics['losing_trades'] = int(losses.sum())
    metrics['breakeven_trades'] = int(be.sum())
    tt = metrics['total_trades'] or 0
    metrics['hit_rate'] = (metrics['winning_trades'] / tt * 100) if tt else 0.0
    for i in [1, 2, 3]:
        m = int(s.str.contains(f'T{i} MET', regex=False, na=False).sum())
        metrics[f'target{i}_met'] = m
        metrics[f'target{i}_rate'] = (m / tt * 100) if tt else 0.0
    return metrics

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 10px;
        background: linear-gradient(135deg, {PRIMARY_A} 0%, {PRIMARY_B} 100%);
        border-radius: 8px; margin-bottom: 16px;'>
        <h3 style='color: #031015; margin: 0;'>📁 Data Source</h3>
    </div>
    """, unsafe_allow_html=True)

    data_mode = st.radio(
        "Select Data Type",
        options=["Archive Data", "PnL Data"],
        index=0,
        help="Choose between Archive Data (trade signals) or PnL Data (profit/loss analysis)"
    )

    selected_folder_id = FOLDER_IDS[data_mode]

    service = get_drive_service()
    if service:
        st.success("✅ Connected to Google Drive", icon="✅")
    else:
        st.error("❌ Connection Failed", icon="❌")
        st.stop()

    with st.spinner("🔄 Loading files..."):
        files = list_files_in_folder(selected_folder_id)

    if not files:
        st.warning("⚠️ No files found", icon="⚠️")
        st.info("Upload CSV/XLSX files to the Drive folder")
        st.stop()

    st.info(f"📊 {len(files)} file(s) available in {data_mode}", icon="📊")

    file_names = [f[1] for f in files]
    selected_file_name = st.selectbox("📂 Select File", file_names, label_visibility="collapsed")
    selected_file_id = next(f[0] for f in files if f[1] == selected_file_name)

    st.markdown("---")

    if st.button("🔄 Refresh Files", key="refresh_files", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style='padding: 14px; background: rgba(255, 217, 26, 0.08); border-radius: 8px; border-left: 3px solid {ACCENT_Y};'>
        <h4 style='color: {ACCENT_Y}; margin: 0 0 6px 0;'>ℹ️ Current Mode</h4>
        <p style='color: {TEXT_MUTED}; font-size: 13px; margin: 0;'>
            <strong style="color:{TEXT_LIGHT}">{data_mode}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Main App ---------------------------
if selected_file_name:
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        with st.spinner(f"⚡ Loading {selected_file_name}..."):
            df = load_and_process_file(selected_file_id, selected_file_name)
    progress_placeholder.empty()

    if df is None or df.empty:
        st.error("❌ Failed to load file or file is empty")
        st.stop()

    st.success(f"✅ Loaded {len(df):,} records from {selected_file_name} (cached)", icon="🚀")

    # ================== Rename Strategy / SCAN_NAME labels (no filtering) ==================
    if 'SCAN_NAME' in df.columns:
        df = df.copy()

        # Normalise original names to make matching robust
        norm = (
            df['SCAN_NAME']
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(' ', '-', regex=False)  # "BB SQUEEZE TRIDENT" → "BB-SQUEEZE-TRIDENT"
        )

        rename_map = {
            "ROS": "OPT 1",
            "BBR": "INDEX 3",
            "RBB": "INDEX 4",
            "RD": "INDEX 2",
            "OPENING-GAP": "BREAKOUT 1",
            "TRIDENT": "MOMENTUM 1",
            "SWING": "SWING 1",
            "RANGE": "BREAKOUT 2",
            "BB-SQUEEZE": "BREAKOUT 3",
            "BB-MACD": "MOMENTUM 2",
            "ADX-EXHAUSTION": "REVERSAL 2",
            "BB-EXHAUSTION": "REVERSAL 1",
            "CANDLE-EXHAUSTION": "REVERSAL 3",
            "BTST": "BTST 1",
            "BB-SQUEEZE-TRIDENT": "INVEST",
        }

        mapped = norm.map(rename_map)

        # Only replace where we have a mapping; everything else stays as-is
        df.loc[mapped.notna(), 'SCAN_NAME'] = mapped[mapped.notna()].values
    # ======================================================================

    # ============================= ARCHIVE DATA MODE =============================
    if data_mode == "Archive Data":
        filter_options = precompute_filter_options_archive(df)

        with st.expander("🔍 Advanced Filters", expanded=True):
            filter_cols = st.columns(5)
            with filter_cols[0]:
                selected_scan = st.multiselect("📊 Strategy", filter_options['scan_names'], default=['All'])
            with filter_cols[1]:
                selected_tf = st.multiselect("⏰ Timeframe", filter_options['timeframes'], default=['All'])
            with filter_cols[2]:
                selected_status = st.multiselect("🎯 Status", filter_options['statuses'], default=['All'])
            with filter_cols[3]:
                selected_alert = st.multiselect("🔔 Alert", filter_options['alerts'], default=['All'])
            with filter_cols[4]:
                selected_curated = st.multiselect("✨ Curated Signal", filter_options['curated'], default=['All'])

            if 'date_col' in filter_options:
                st.markdown("---")
                st.markdown("##### 📅 Date Range Filter")
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=filter_options['min_date'],
                        min_value=filter_options['min_date'],
                        max_value=filter_options['max_date']
                    )
                with date_col2:
                    end_date = st.date_input(
                        "End Date",
                        value=filter_options['max_date'],
                        min_value=filter_options['min_date'],
                        max_value=filter_options['max_date']
                    )
                st.caption(f"📊 Using date column: {filter_options['date_col']}")
            else:
                start_date = None
                end_date = None

        filtered_df = apply_filters_archive(
            df, selected_scan, selected_tf, selected_status,
            selected_alert, selected_curated, start_date, end_date,
            filter_options.get('date_col')
        )

        filter_pct = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        st.info(f"📌 Showing {len(filtered_df):,} / {len(df):,} trades ({filter_pct:.1f}%)", icon="📌")

        metrics = calculate_metrics_archive(filtered_df)

        st.markdown(f"""
        <div style='text-align: center; margin: 16px 0;'>
            <h2 style='
                color:{ACCENT_Y};
                font-size: 30px;
                font-weight: 900;
            '>📊 Performance Metrics</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", f"{metrics['total_trades']:,}")
        with col2:
            st.metric("Winning Trades", f"{metrics.get('winning_trades', 0):,}",
                      delta=f"{metrics.get('hit_rate', 0):.1f}%", help="Trades that hit any target (T1/T2/T3)")
        with col3:
            losing = metrics.get('losing_trades', 0)
            total = max(metrics['total_trades'], 1)
            st.metric("Losing Trades", f"{losing:,}",
                      delta=f"-{(losing/total*100):.1f}%", delta_color="inverse")
        with col4:
            st.metric("Breakeven", f"{metrics.get('breakeven_trades', 0):,}")
        with col5:
            win_loss_ratio = metrics.get('winning_trades', 0) / max(metrics.get('losing_trades', 1), 1)
            st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")

        st.markdown(f"""
        <div style='text-align: center; margin: 24px 0 10px 0;'>
            <h3 style='
                color:{TEXT_LIGHT};
                font-size: 22px;
                font-weight: 800;
            '>🎯 Target Achievement Rates</h3>
        </div>
        """, unsafe_allow_html=True)

        target_cols = st.columns(3)
        for i, col in enumerate(target_cols, 1):
            with col:
                rate = metrics.get(f'target{i}_rate', 0)
                st.metric(
                    f"Target {i} Hit Rate",
                    f"{metrics.get(f'target{i}_met', 0):,} trades",
                    delta=f"{rate:.1f}%"
                )

        st.markdown(f"""
        <div style='text-align: center; margin: 32px 0 16px 0;'>
            <h2 style='
                color:{ACCENT_Y};
                background:none;
                font-size: 30px;
                font-weight: 900;
            '>📈 Advanced Analytics</h2>
        </div>
        """, unsafe_allow_html=True)

        viz_tabs = st.tabs(["📊 Overview", "🎯 Performance", "📈 Time Series", "🔥 Heatmaps"])

        # ----- Overview Tab (Enhanced Colors)
        with viz_tabs[0]:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    status_counts = filtered_df['STATUS'].value_counts()
                    labels = status_counts.index.astype(str)
                    # Enhanced color mapping for pie chart
                    colors = []
                    for lbl in labels:
                        L = lbl.upper()
                        if 'T3 MET' in L:
                            colors.append(ANALYTICS_GRADIENT_GREEN[6])  # Brightest green for T3
                        elif 'T2 MET' in L:
                            colors.append(ANALYTICS_GRADIENT_GREEN[4])  # Medium green for T2
                        elif 'T1 MET' in L:
                            colors.append(ANALYTICS_GRADIENT_GREEN[2])  # Darker green for T1
                        elif 'SL MET' in L:
                            colors.append(ANALYTICS_GRADIENT_ORANGE[5])  # Orange-red for SL
                        else:
                            colors.append(ANALYTICS_GRADIENT_BLUE[3])  # Blue for others
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=status_counts.values,
                        hole=0.55,
                        marker=dict(
                            colors=colors, 
                            line=dict(color=BG_END, width=3)
                        ),
                        textposition='inside',
                        textinfo='percent',
                        textfont=dict(size=14, weight='bold'),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    fig_pie.update_layout(
                        title={'text': "📊 Trade Status Distribution", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_A}},
                        showlegend=True, height=450,
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                                    bgcolor=SURFACE_ALT, bordercolor=BORDER_RGBA, borderwidth=1, font=dict(size=11))
                    )
                    style_chart(fig_pie)
                    st.plotly_chart(fig_pie, use_container_width=True)

            with viz_col2:
                if 'SCAN_NAME' in filtered_df.columns and 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    try:
                        status_by_strategy = pd.crosstab(filtered_df['SCAN_NAME'], filtered_df['STATUS'])
                        win_cols = [c for c in status_by_strategy.columns
                                    if any(x in str(c).upper() for x in ['T1 MET', 'T2 MET', 'T3 MET'])]
                        if win_cols:
                            total_per_strategy = status_by_strategy.sum(axis=1)
                            win_per_strategy = status_by_strategy[win_cols].sum(axis=1)
                            win_rate = (win_per_strategy / total_per_strategy * 100).sort_values(ascending=True)
                            
                            # Gradient colors based on win rate
                            bar_colors = []
                            for rate in win_rate.values:
                                if rate >= 75:
                                    bar_colors.append(ANALYTICS_GRADIENT_GREEN[6])
                                elif rate >= 60:
                                    bar_colors.append(ANALYTICS_GRADIENT_GREEN[4])
                                elif rate >= 50:
                                    bar_colors.append(ANALYTICS_GRADIENT_BLUE[4])
                                elif rate >= 40:
                                    bar_colors.append(ANALYTICS_GRADIENT_ORANGE[3])
                                else:
                                    bar_colors.append(ANALYTICS_GRADIENT_ORANGE[5])
                            
                            fig_winrate = go.Figure(go.Bar(
                                x=win_rate.values, y=win_rate.index, orientation='h',
                                text=[f"{val:.1f}%" for val in win_rate.values], textposition='outside',
                                marker=dict(
                                    color=bar_colors,
                                    line=dict(color=BG_END, width=1)
                                ),
                                hovertemplate='<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>'
                            ))
                            fig_winrate.update_layout(
                                title={'text': "🏆 Win Rate by Strategy", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_B}},
                                xaxis_title="Win Rate (%)", yaxis_title="", height=450,
                                xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                            )
                            style_chart(fig_winrate)
                            st.plotly_chart(fig_winrate, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create win rate chart: {e}")

            viz_col3, viz_col4 = st.columns(2)
            with viz_col3:
                if 'TIMEFRAME' in filtered_df.columns and not filtered_df.empty:
                    tf_counts = filtered_df['TIMEFRAME'].astype(str).value_counts().sort_values(ascending=True)
                    # Use purple gradient for timeframes
                    n_bars = len(tf_counts)
                    bar_colors = [ANALYTICS_GRADIENT_PURPLE[min(i, len(ANALYTICS_GRADIENT_PURPLE)-1)] for i in range(n_bars)]
                    
                    fig_tf = go.Figure(go.Bar(
                        x=tf_counts.values, y=tf_counts.index, orientation='h',
                        text=tf_counts.values, textposition='outside',
                        marker=dict(
                            color=bar_colors,
                            line=dict(color=BG_END, width=1)
                        ),
                        hovertemplate='<b>%{y}</b><br>Trades: %{x}<extra></extra>'
                    ))
                    fig_tf.update_layout(
                        title={'text': "⏰ Trades by Timeframe", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': ACCENT_Y}},
                        xaxis_title="Number of Trades", yaxis_title="", height=450,
                        xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                    )
                    style_chart(fig_tf)
                    st.plotly_chart(fig_tf, use_container_width=True)

            with viz_col4:
                if 'ALERT' in filtered_df.columns and not filtered_df.empty:
                    alert_counts = filtered_df['ALERT'].value_counts()
                    fig_alert = go.Figure(data=[go.Pie(
                        labels=alert_counts.index, values=alert_counts.values, hole=0.6,
                        marker=dict(
                            colors=ANALYTICS_PIE_COLORS[:len(alert_counts)],
                            line=dict(color=BG_END, width=3)
                        ),
                        textposition='inside', textinfo='percent',
                        textfont=dict(size=14, weight='bold'),
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    fig_alert.update_layout(
                        title={'text': "🔔 Alert Type Distribution", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_A}},
                        showlegend=True, height=450,
                        legend=dict(bgcolor=SURFACE_ALT, bordercolor=BORDER_RGBA, borderwidth=1, font=dict(size=11))
                    )
                    style_chart(fig_alert)
                    st.plotly_chart(fig_alert, use_container_width=True)

        # ----- Performance Tab (Enhanced Colors)
        with viz_tabs[1]:
            viz_col5, viz_col6 = st.columns(2)
            with viz_col5:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    breakdown_data = {
                        'Category': ['Wins', 'Losses', 'Breakeven'],
                        'Count': [metrics.get('winning_trades', 0),
                                  metrics.get('losing_trades', 0),
                                  metrics.get('breakeven_trades', 0)]
                    }
                    colors = [ANALYTICS_GRADIENT_GREEN[5], ANALYTICS_GRADIENT_ORANGE[5], ANALYTICS_GRADIENT_BLUE[3]]
                    fig_breakdown = go.Figure(go.Bar(
                        x=breakdown_data['Category'], y=breakdown_data['Count'],
                        text=breakdown_data['Count'], textposition='outside',
                        marker=dict(
                            color=colors,
                            line=dict(color=BG_END, width=2)
                        ),
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    ))
                    fig_breakdown.update_layout(
                        title={'text': "📊 Win/Loss/Breakeven Breakdown", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_B}},
                        xaxis_title="", yaxis_title="Number of Trades", height=450,
                        xaxis=dict(gridcolor='#374151'), yaxis=dict(gridcolor='#374151', showgrid=True)
                    )
                    style_chart(fig_breakdown)
                    st.plotly_chart(fig_breakdown, use_container_width=True)

            with viz_col6:
                if 'SCAN_NAME' in filtered_df.columns and not filtered_df.empty:
                    try:
                        trades_by_scan = filtered_df['SCAN_NAME'].value_counts().sort_values(ascending=True)
                        # Use blue gradient
                        n_bars = len(trades_by_scan)
                        bar_colors = [ANALYTICS_GRADIENT_BLUE[min(i % 7, 6)] for i in range(n_bars)]
                        
                        fig_scan = go.Figure(go.Bar(
                            x=trades_by_scan.values, y=trades_by_scan.index, orientation='h',
                            text=trades_by_scan.values, textposition='outside',
                            marker=dict(
                                color=bar_colors,
                                line=dict(color=BG_END, width=1)
                            ),
                            hovertemplate='<b>%{y}</b><br>Trades: %{x}<extra></extra>'
                        ))
                        fig_scan.update_layout(
                            title={'text': "📌 Trades by Strategy", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': ACCENT_Y}},
                            xaxis_title="Trades", yaxis_title="", height=450,
                            xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                        )
                        style_chart(fig_scan)
                        st.plotly_chart(fig_scan, use_container_width=True)
                    except Exception:
                        pass

        # ----- Time Series Tab (Enhanced Colors)
        with viz_tabs[2]:
            if 'date_col' in filter_options and filter_options['date_col'] in filtered_df.columns:
                date_col = filter_options['date_col']
                df_time = filtered_df.dropna(subset=[date_col]).copy().sort_values(date_col)
                if not df_time.empty:
                    df_time['cumulative_trades'] = range(1, len(df_time) + 1)
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=df_time[date_col], y=df_time['cumulative_trades'], mode='lines',
                        name='Cumulative Trades', 
                        line=dict(width=4, color=ANALYTICS_GRADIENT_BLUE[4], shape='spline'),
                        fill='tozeroy', fillcolor='rgba(36, 87, 214, 0.18)',
                        hovertemplate='<b>Date</b>: %{x}<br><b>Total Trades</b>: %{y}<extra></extra>'
                    ))
                    fig_timeline.update_layout(
                        title={'text': "📈 Cumulative Trades Over Time", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_A}},
                        xaxis_title="Date", yaxis_title="Total Trades", height=500,
                        xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151', showgrid=True),
                        hovermode='x unified'
                    )
                    style_chart(fig_timeline)
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    if 'STATUS' in df_time.columns:
                        df_time['is_win'] = df_time['STATUS'].astype(str).str.upper().str.contains(
                            'T1 MET|T2 MET|T3 MET', regex=True, na=False).astype(int)
                        window_size = min(20, len(df_time))
                        df_time['rolling_win_rate'] = df_time['is_win'].rolling(window=window_size, min_periods=1).mean() * 100
                        fig_winrate_time = go.Figure()
                        fig_winrate_time.add_trace(go.Scatter(
                            x=df_time[date_col], y=df_time['rolling_win_rate'], mode='lines',
                            name=f'Win Rate ({window_size}-trade MA)',
                            line=dict(width=4, color=ANALYTICS_GRADIENT_GREEN[5], shape='spline'),
                            fill='tozeroy', fillcolor='rgba(16,185,129,0.20)',
                            hovertemplate='<b>Date</b>: %{x}<br><b>Win Rate</b>: %{y:.1f}%<extra></extra>'
                        ))
                        fig_winrate_time.add_hline(y=50, line_dash="dash", line_color=ACCENT_Y, line_width=2,
                                                   annotation_text="50% Baseline", annotation_position="right")
                        fig_winrate_time.update_layout(
                            title={'text': f"🎯 Rolling Win Rate ({window_size}-Trade MA)", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': PRIMARY_B}},
                            xaxis_title="Date", yaxis_title="Win Rate (%)", height=450,
                            xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151', showgrid=True),
                            hovermode='x unified'
                        )
                        style_chart(fig_winrate_time)
                        st.plotly_chart(fig_winrate_time, use_container_width=True)

        # ----- Heatmaps Tab (Enhanced Colors)
        with viz_tabs[3]:
            if 'SCAN_NAME' in filtered_df.columns and 'TIMEFRAME' in filtered_df.columns:
                try:
                    heatmap_data = pd.crosstab(filtered_df['SCAN_NAME'], filtered_df['TIMEFRAME'])
                    # Custom colorscale with NeoTrader colors
                    custom_colorscale = [
                        [0.0, ANALYTICS_GRADIENT_BLUE[0]],
                        [0.25, ANALYTICS_GRADIENT_BLUE[2]],
                        [0.5, ANALYTICS_GRADIENT_PURPLE[3]],
                        [0.75, ANALYTICS_GRADIENT_GREEN[3]],
                        [1.0, ANALYTICS_GRADIENT_GREEN[6]]
                    ]
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                        colorscale=custom_colorscale,
                        text=heatmap_data.values, texttemplate='%{text}',
                        textfont={"size": 14, "color": "white", "family": "Arial Black"},
                        hovertemplate='<b>Strategy</b>: %{y}<br><b>Timeframe</b>: %{x}<br><b>Trades</b>: %{z}<extra></extra>',
                        colorbar=dict(title="Trades", thickness=18, bgcolor=SURFACE_ALT, bordercolor=BORDER_RGBA, borderwidth=1)
                    ))
                    fig_heatmap.update_layout(
                        title={'text': "🔥 Strategy × Timeframe Heatmap", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'color': ACCENT_Y}},
                        xaxis_title="Timeframe", yaxis_title="Strategy", height=600,
                        xaxis=dict(side='bottom'), yaxis=dict(side='left')
                    )
                    style_chart(fig_heatmap)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create heatmap: {e}")

    # ============================= PNL DATA MODE =============================
    elif data_mode == "PnL Data":
        filter_options = precompute_filter_options_pnl(df)

        with st.expander("🔍 Advanced Filters", expanded=True):
            filter_cols = st.columns(6)
            with filter_cols[0]:
                selected_symbols = st.multiselect("💹 Symbol", filter_options['symbols'], default=['All'])
            with filter_cols[1]:
                selected_trades = st.multiselect("📈 Trade Type", filter_options['trades'], default=['All'])
            with filter_cols[2]:
                selected_statuses = st.multiselect("🎯 Status", filter_options['statuses'], default=['All'])
            with filter_cols[3]:
                selected_scan_names = st.multiselect("📊 Strategy", filter_options['scan_names'], default=['All'])
            with filter_cols[4]:
                selected_curated_pnl = st.multiselect("✨ Curated", filter_options['curated'], default=['All'])
            with filter_cols[5]:
                selected_fno = st.multiselect("🔄 FNO", filter_options['fno'], default=['All'])

            if 'date_col' in filter_options:
                st.markdown("---")
                st.markdown("##### 📅 Date Range Filter")
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    start_date_pnl = st.date_input(
                        "Start Date",
                        value=filter_options['min_date'],
                        min_value=filter_options['min_date'],
                        max_value=filter_options['max_date'],
                        key="pnl_start_date"
                    )
                with date_col2:
                    end_date_pnl = st.date_input(
                        "End Date",
                        value=filter_options['max_date'],
                        min_value=filter_options['min_date'],
                        max_value=filter_options['max_date'],
                        key="pnl_end_date"
                    )
                st.caption(f"📊 Using date column: {filter_options['date_col']}")
            else:
                start_date_pnl = None
                end_date_pnl = None

        filtered_df = apply_filters_pnl(
            df, selected_symbols, selected_trades, selected_statuses,
            selected_scan_names, selected_curated_pnl, selected_fno,
            start_date_pnl, end_date_pnl, filter_options.get('date_col')
        )

        filter_pct = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        st.info(f"📌 Showing {len(filtered_df):,} / {len(df):,} trades ({filter_pct:.1f}%)", icon="📌")

        date_col_for_pnl = filter_options.get('date_col')
        pnl_metrics = calculate_metrics_pnl(filtered_df, date_col=date_col_for_pnl)

        st.markdown(f"""
        <div style='text-align: center; margin: 16px 0;'>
            <h2 style='
                color:{ACCENT_Y};
                font-size: 30px;
                font-weight: 900;
            '>💰 Profit & Loss Metrics</h2>
        </div>
        """, unsafe_allow_html=True)

        # 1st row: Win Rate – Winning Trades – Losing Trades – Total Trades – Profit Factor
        pnl_col1, pnl_col2, pnl_col3, pnl_col4, pnl_col5 = st.columns(5)
        with pnl_col1:
            st.metric("Win Rate", f"{pnl_metrics['win_rate']:.1f}%",
                      delta=f"{pnl_metrics['winning_trades']:,} wins")
        with pnl_col2:
            st.metric("Winning Trades", f"{pnl_metrics['winning_trades']:,}")
        with pnl_col3:
            st.metric("Losing Trades", f"{pnl_metrics['losing_trades']:,}",
                      delta=f"-{pnl_metrics['losing_trades']:,}", delta_color="inverse")
        with pnl_col4:
            st.metric("Total Trades", f"{pnl_metrics['total_trades']:,}")
        with pnl_col5:
            st.metric("Profit Factor", f"{pnl_metrics['profit_factor']:.2f}")

        # 2nd row: Total P&L, Avg Winner, Avg Loser, Sharpe Ratio
        pnl_col6, pnl_col7, pnl_col8, pnl_col9 = st.columns(4)
        with pnl_col6:
            pnl_val = pnl_metrics['total_pnl']
            st.metric("Total P&L", f"₹{pnl_val:,.0f}",
                      delta="Profit" if pnl_val > 0 else "Loss",
                      delta_color="normal" if pnl_val > 0 else "inverse")
        with pnl_col7:
            st.metric("Avg Winner", f"₹{pnl_metrics['avg_win']:,.0f}")
        with pnl_col8:
            st.metric("Avg Loser", f"₹{pnl_metrics['avg_loss']:,.0f}",
                      delta=f"-₹{pnl_metrics['avg_loss']:,.0f}", delta_color="inverse")
        with pnl_col9:
            st.metric("Sharpe Ratio", f"{pnl_metrics['sharpe_ratio']:.2f}")

        # NEW 3rd row: Margin & ROI (keeps all old metrics intact) 
        pnl_col11, pnl_col12 = st.columns(2)
        with pnl_col11:
            st.metric(
                "Peak Margin Required",
                f"₹{pnl_metrics['total_margin']:,.0f}",
                help="Maximum combined margin used at any point in the selected period (treated as Initial Capital)"
            )
        with pnl_col12:
            st.metric(
                "ROI on Margin",
                f"{pnl_metrics['roi']:.1f}%",
                help="Total P&L ÷ Peak Margin Required × 100 for the filtered period"
            )

        st.markdown(f"""
        <div style='text-align: center; margin: 28px 0 12px 0;'>
            <h2 style='
                color:{ACCENT_Y};
                background:none;
                font-size: 30px;
                font-weight: 900;
            '>📊 P&L Analytics Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)

        pnl_tabs = st.tabs(["💹 Equity Curve", "📊 Performance Analysis", "🎯 Trade Distribution", "📈 Time-Based Analysis"])

        # TAB 1: Equity Curve (capital-aware)
        with pnl_tabs[0]:
            if PNL_COL in filtered_df.columns:
                date_col_pnl = filter_options.get('date_col')
                if date_col_pnl:
                    initial_capital = pnl_metrics.get('initial_capital', 0.0)
                    df_equity = build_pnl_equity_drawdown_df(
                        filtered_df,
                        date_col=date_col_pnl,
                        col=PNL_COL,
                        initial_capital=initial_capital
                    )

                    max_dd_value = float(df_equity['Drawdown_PNL'].min())
                    max_dd_idx = int(df_equity['Drawdown_PNL'].idxmin())
                    max_dd_date = df_equity.loc[max_dd_idx, date_col_pnl]
                    max_dd_abs = abs(max_dd_value)
                    max_dd_pct = abs(float(df_equity.loc[max_dd_idx, 'Drawdown_Pct']))
                    peak_equity_at_dd = float(df_equity.loc[max_dd_idx, 'Running_Max_Equity'])
                    peak_pnl_at_dd = float(df_equity.loc[max_dd_idx, 'Running_Max_PNL'])
                    trough_equity = float(df_equity.loc[max_dd_idx, 'Equity'])
                    peak_window = df_equity.iloc[:max_dd_idx + 1]
                    peak_dd_idx = int((peak_window['Cumulative_PNL'] - peak_pnl_at_dd).abs().idxmin())
                    peak_dd_date = df_equity.loc[peak_dd_idx, date_col_pnl]
                    peak_dd_equity = float(df_equity.loc[peak_dd_idx, 'Equity'])

                    equity_hover = np.column_stack((
                        df_equity['Cumulative_PNL'],
                        df_equity['Running_Max_Equity']
                    ))
                    drawdown_hover = np.column_stack((
                        df_equity['Drawdown_PNL'].abs(),
                        df_equity['Drawdown_Pct'].abs()
                    ))

                    fig_equity = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.66, 0.34],
                        subplot_titles=("Equity Curve", "Drawdown"),
                        vertical_spacing=0.07,
                        shared_xaxes=True
                    )

                    if peak_dd_date != max_dd_date:
                        for chart_row in (1, 2):
                            fig_equity.add_vrect(
                                x0=peak_dd_date, x1=max_dd_date,
                                fillcolor='rgba(239, 68, 68, 0.06)',
                                line_width=0,
                                layer='below',
                                row=chart_row, col=1
                            )

                    fig_equity.add_trace(
                        go.Scatter(
                            x=df_equity[date_col_pnl],
                            y=df_equity['Equity'],
                            mode='lines',
                            name='Equity',
                            customdata=equity_hover,
                            line=dict(width=2.8, color=PRIMARY_B, shape='spline'),
                            fill='tozeroy',
                            fillcolor='rgba(45, 212, 191, 0.08)',
                            hovertemplate=(
                                '<b>%{x|%d %b %Y}</b><br>'
                                'Equity: ₹%{y:,.0f}<br>'
                                'Cumulative P&L: ₹%{customdata[0]:,.0f}<extra></extra>'
                            )
                        ),
                        row=1, col=1
                    )
                    fig_equity.add_trace(
                        go.Scatter(
                            x=df_equity[date_col_pnl],
                            y=df_equity['Running_Max_Equity'],
                            mode='lines',
                            name='Peak',
                            line=dict(width=1.4, color='rgba(250, 204, 21, 0.55)', dash='dot'),
                            hovertemplate='<b>%{x|%d %b %Y}</b><br>Peak: ₹%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    fig_equity.add_hline(
                        y=initial_capital if initial_capital > 0 else 0,
                        line_dash="dash",
                        line_color='rgba(156, 163, 175, 0.35)',
                        line_width=1,
                        row=1, col=1
                    )
                    fig_equity.add_trace(
                        go.Scatter(
                            x=[peak_dd_date],
                            y=[peak_dd_equity],
                            mode='markers',
                            name='Peak before DD',
                            showlegend=False,
                            marker=dict(size=8, color=ACCENT_Y, symbol='circle', line=dict(width=1, color=BG_END)),
                            hovertemplate='<b>Peak</b><br>₹%{y:,.0f}<br>%{x|%d %b %Y}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    fig_equity.add_trace(
                        go.Scatter(
                            x=[max_dd_date],
                            y=[trough_equity],
                            mode='markers',
                            name='DD trough',
                            showlegend=False,
                            marker=dict(size=9, color=LOSS_RED, symbol='circle', line=dict(width=1, color=BG_END)),
                            hovertemplate='<b>Trough</b><br>₹%{y:,.0f}<br>%{x|%d %b %Y}<extra></extra>'
                        ),
                        row=1, col=1
                    )

                    fig_equity.add_trace(
                        go.Scatter(
                            x=df_equity[date_col_pnl],
                            y=df_equity['Drawdown_PNL'],
                            mode='lines',
                            name='Drawdown',
                            customdata=drawdown_hover,
                            line=dict(color='#f97316', width=1.8, shape='spline'),
                            fill='tozeroy',
                            fillcolor='rgba(249, 115, 22, 0.18)',
                            hovertemplate=(
                                '<b>%{x|%d %b %Y}</b><br>'
                                'Down: ₹%{customdata[0]:,.0f}<br>'
                                'DD: %{customdata[1]:.2f}%<extra></extra>'
                            )
                        ),
                        row=2, col=1
                    )
                    fig_equity.add_hline(
                        y=0,
                        line_color='rgba(156, 163, 175, 0.35)',
                        line_width=1,
                        row=2, col=1
                    )
                    fig_equity.add_trace(
                        go.Scatter(
                            x=[max_dd_date],
                            y=[max_dd_value],
                            mode='markers',
                            name='Max DD',
                            showlegend=False,
                            marker=dict(size=8, color=LOSS_RED, symbol='triangle-down', line=dict(width=1, color=BG_END)),
                            hoverinfo='skip'
                        ),
                        row=2, col=1
                    )
                    fig_equity.add_annotation(
                        x=max_dd_date,
                        y=max_dd_value,
                        text=(
                            f"<b>Peak</b> ₹{peak_equity_at_dd:,.0f}<br>"
                            f"<b>Down</b> ₹{max_dd_abs:,.0f}<br>"
                            f"<b>DD</b> {max_dd_pct:.1f}%"
                        ),
                        showarrow=True,
                        arrowhead=0,
                        arrowwidth=1.2,
                        arrowcolor='rgba(249, 115, 22, 0.7)',
                        ax=48,
                        ay=-34,
                        bgcolor='rgba(15, 23, 42, 0.86)',
                        bordercolor='rgba(249, 115, 22, 0.45)',
                        borderwidth=1,
                        borderpad=6,
                        font=dict(size=11, color=TEXT_LIGHT),
                        row=2, col=1
                    )

                    fig_equity.update_layout(
                        title={'text': "Equity & Drawdown", 'x': 0.01, 'xanchor': 'left', 'font': {'size': 18, 'color': TEXT_LIGHT}},
                        height=720,
                        showlegend=True,
                        hovermode='x unified',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=1.02,
                            xanchor='right',
                            x=1,
                            bgcolor='rgba(0,0,0,0)',
                            font=dict(size=11, color=TEXT_MUTED)
                        ),
                        margin=dict(l=64, r=34, t=72, b=42),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    fig_equity.update_xaxes(
                        gridcolor='rgba(55, 65, 81, 0.35)',
                        showgrid=True,
                        zeroline=False,
                        showline=False,
                        row=1, col=1
                    )
                    fig_equity.update_xaxes(
                        gridcolor='rgba(55, 65, 81, 0.35)',
                        showgrid=True,
                        zeroline=False,
                        showline=False,
                        row=2, col=1
                    )
                    fig_equity.update_yaxes(
                        title_text='Equity (₹)',
                        tickformat=',.0f',
                        gridcolor='rgba(55, 65, 81, 0.4)',
                        showgrid=True,
                        zeroline=False,
                        row=1, col=1
                    )
                    fig_equity.update_yaxes(
                        title_text='Drawdown (₹)',
                        tickformat=',.0f',
                        gridcolor='rgba(55, 65, 81, 0.4)',
                        showgrid=True,
                        zeroline=False,
                        row=2, col=1
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                else:
                    st.warning("📅 No date column found for equity curve visualization")
            else:
                st.error("❌ PNL column not found in data")

        # TAB 2: Performance Analysis
        with pnl_tabs[1]:
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                if PNL_COL in filtered_df.columns and not filtered_df.empty:
                    pnl_series = pd.to_numeric(filtered_df[PNL_COL], errors='coerce').fillna(0)
                    win_loss_data = pd.DataFrame({
                        'Category': ['Wins', 'Losses'],
                        'Count': [(pnl_series > 0).sum(), (pnl_series < 0).sum()],
                        'Amount': [pnl_series[pnl_series > 0].sum(), abs(pnl_series[pnl_series < 0]).sum()]
                    })
                    fig_winloss = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Trade Count", "Total Amount (₹)"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    fig_winloss.add_trace(
                        go.Bar(x=win_loss_data['Category'], y=win_loss_data['Count'],
                               text=win_loss_data['Count'], textposition='outside', name='Count',
                               marker=dict(color=[ANALYTICS_GRADIENT_GREEN[5], ANALYTICS_GRADIENT_ORANGE[5]])),
                        row=1, col=1
                    )
                    fig_winloss.add_trace(
                        go.Bar(x=win_loss_data['Category'], y=win_loss_data['Amount'],
                               text=[f"₹{v:,.0f}" for v in win_loss_data['Amount']],
                               textposition='outside', name='Amount',
                               marker=dict(color=[ANALYTICS_GRADIENT_GREEN[4], ANALYTICS_GRADIENT_ORANGE[4]])),
                        row=1, col=2
                    )
                    fig_winloss.update_layout(
                        title={'text': "🎯 Win/Loss Breakdown", 'x': 0.5, 'xanchor': 'center'},
                        height=450, showlegend=False
                    )
                    fig_winloss.update_xaxes(gridcolor='#374151')
                    fig_winloss.update_yaxes(gridcolor='#374151', showgrid=True)
                    style_chart(fig_winloss)
                    st.plotly_chart(fig_winloss, use_container_width=True)

            with perf_col2:
                if PNL_COL in filtered_df.columns and not filtered_df.empty:
                    pnl_data = pd.to_numeric(filtered_df[PNL_COL], errors='coerce').dropna()
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=pnl_data, nbinsx=50,
                        marker=dict(line=dict(color=BG_END, width=1)),
                        hovertemplate='<b>P&L Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
                    ))
                    fig_dist.add_vline(x=0, line_dash="dash", line_color=TEXT_MUTED, line_width=2,
                                       annotation_text="Breakeven", annotation_position="top")
                    fig_dist.update_layout(
                        title={'text': "📊 P&L Distribution", 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title="P&L (₹)", yaxis_title="Frequency", height=450,
                        xaxis=dict(gridcolor='#374151', showgrid=True),
                        yaxis=dict(gridcolor='#374151', showgrid=True)
                    )
                    style_chart(fig_dist)
                    st.plotly_chart(fig_dist, use_container_width=True)

            if 'SCAN_NAME' in filtered_df.columns and PNL_COL in filtered_df.columns and not filtered_df.empty:
                st.markdown("---")
                pnl_by_strategy = filtered_df.groupby('SCAN_NAME')[PNL_COL].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').sum()
                ).sort_values(ascending=True)
                fig_strategy_pnl = go.Figure(go.Bar(
                    x=pnl_by_strategy.values, y=pnl_by_strategy.index, orientation='h',
                    text=[f"₹{val:,.0f}" for val in pnl_by_strategy.values], textposition='outside',
                    marker=dict(
                        color=[ANALYTICS_GRADIENT_GREEN[5] if v >= 0 else ANALYTICS_GRADIENT_ORANGE[5] for v in pnl_by_strategy.values]
                    ),
                    hovertemplate='<b>%{y}</b><br>P&L: ₹%{x:,.2f}<extra></extra>'
                ))
                fig_strategy_pnl.update_layout(
                    title={'text': "💼 P&L by Strategy/Scan", 'x': 0.5, 'xanchor': 'center'},
                    xaxis_title="Total P&L (₹)", yaxis_title="", height=500,
                    xaxis=dict(gridcolor='#374151', showgrid=True),
                    yaxis=dict(gridcolor='#374151')
                )
                style_chart(fig_strategy_pnl)
                st.plotly_chart(fig_strategy_pnl, use_container_width=True)

        # TAB 3: Trade Distribution
        with pnl_tabs[2]:
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    status_series = filtered_df['STATUS'].astype(str).str.upper().str.strip()

                    # New logic:
                    # T1 = T1 + T2 + T3
                    # T2 = T2 + T3
                    # T3 = T3 only
                    # SL = SL only
                    t1_all = status_series.str.contains('T1 MET|T2 MET|T3 MET', regex=True, na=False).sum()
                    t2_plus = status_series.str.contains('T2 MET|T3 MET', regex=True, na=False).sum()
                    t3_only = status_series.str.contains('T3 MET', regex=False, na=False).sum()
                    sl_met  = status_series.str.contains('SL MET', regex=False, na=False).sum()

                    categories = ["T1 & Above (T1+T2+T3)", "T2 & Above (T2+T3)", "T3 Only", "SL Met"]
                    counts = [t1_all, t2_plus, t3_only, sl_met]

                    colors = [
                        ANALYTICS_GRADIENT_GREEN[3],  # T1 & Above - medium green
                        ANALYTICS_GRADIENT_GREEN[5],  # T2 & Above - brighter green
                        ANALYTICS_GRADIENT_GREEN[6],  # T3 Only - brightest green
                        ANALYTICS_GRADIENT_ORANGE[5]  # SL Met - orange-red
                    ]

                    fig_trade_dist = go.Figure(
                        data=[
                            go.Bar(
                                x=categories,
                                y=counts,
                                marker=dict(
                                    color=colors,
                                    line=dict(color=BG_END, width=2)
                                ),
                                text=counts,
                                textposition='outside',
                                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                            )
                        ]
                    )

                    fig_trade_dist.update_layout(
                        title={'text': "🎯 Trade Distribution by Target (Histogram View)", 'x': 0.5, 'xanchor': 'center'},
                        xaxis_title="Target Buckets",
                        yaxis_title="Number of Trades",
                        height=450,
                        xaxis=dict(gridcolor='#374151', showgrid=False),
                        yaxis=dict(gridcolor='#374151', showgrid=True)
                    )

                    style_chart(fig_trade_dist)
                    st.plotly_chart(fig_trade_dist, use_container_width=True)

            with dist_col2:
                if 'TRADE' in filtered_df.columns and not filtered_df.empty:
                    trade_counts = filtered_df['TRADE'].value_counts()
                    fig_trade_type = go.Figure(data=[go.Pie(
                        labels=trade_counts.index, values=trade_counts.values, hole=0.5,
                        marker=dict(
                            colors=ANALYTICS_PIE_COLORS[:len(trade_counts)],
                            line=dict(color=BG_END, width=3)
                        ),
                        textposition='inside', textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
                    )])
                    fig_trade_type.update_layout(
                        title={'text': "📈 Trade Type Distribution", 'x': 0.5, 'xanchor': 'center'},
                        height=450, showlegend=True
                    )
                    style_chart(fig_trade_type)
                    st.plotly_chart(fig_trade_type, use_container_width=True)

            if PNL_COL in filtered_df.columns and not filtered_df.empty:
                st.markdown("---")
                st.markdown("### 🏆 Top Performers & 📉 Worst Performers")
                symbol_col = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in filtered_df.columns else 'SYMBOL'
                if symbol_col in filtered_df.columns:
                    top_col1, top_col2 = st.columns(2)
                    with top_col1:
                        top_winners = filtered_df.copy()
                        top_winners['PNL_num'] = pd.to_numeric(top_winners[PNL_COL], errors='coerce')
                        top_winners = top_winners.nlargest(10, 'PNL_num')[[symbol_col, PNL_COL, 'TRADE' if 'TRADE' in filtered_df.columns else symbol_col]]
                        st.markdown("#### 🥇 Top 10 Winning Trades")
                        st.dataframe(
                            top_winners.style.background_gradient(subset=[PNL_COL], cmap='Greens'),
                            use_container_width=True, hide_index=True, height=400
                        )
                    with top_col2:
                        top_losers = filtered_df.copy()
                        top_losers['PNL_num'] = pd.to_numeric(top_losers[PNL_COL], errors='coerce')
                        top_losers = top_losers.nsmallest(10, 'PNL_num')[[symbol_col, PNL_COL, 'TRADE' if 'TRADE' in filtered_df.columns else symbol_col]]
                        st.markdown("#### 📉 Top 10 Losing Trades")
                        st.dataframe(
                            top_losers.style.background_gradient(subset=[PNL_COL], cmap='Reds_r'),
                            use_container_width=True, hide_index=True, height=400
                        )

        # TAB 4: Time-Based Analysis
        with pnl_tabs[3]:
            date_col_pnl = filter_options.get('date_col')
            if date_col_pnl and PNL_COL in filtered_df.columns and not filtered_df.empty:
                df_time_pnl = filtered_df.dropna(subset=[date_col_pnl, PNL_COL]).copy()
                df_time_pnl['PNL_num'] = pd.to_numeric(df_time_pnl[PNL_COL], errors='coerce')
                df_time_pnl = df_time_pnl.sort_values(date_col_pnl)

                # Monthly Performance & Trade Count
                df_time_pnl['Month'] = pd.to_datetime(df_time_pnl[date_col_pnl]).dt.to_period('M').astype(str)
                monthly_pnl = df_time_pnl.groupby('Month').agg({'PNL_num': ['sum', 'count', 'mean']}).reset_index()
                monthly_pnl.columns = ['Month', 'Total_PNL', 'Trade_Count', 'Avg_PNL']

                fig_monthly = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("💰 Monthly Performance", "📊 Trade Count"),
                    row_heights=[0.6, 0.4], vertical_spacing=0.15
                )
                colors_monthly = [ANALYTICS_GRADIENT_GREEN[5] if x >= 0 else ANALYTICS_GRADIENT_ORANGE[5] for x in monthly_pnl['Total_PNL']]
                fig_monthly.add_trace(
                    go.Bar(x=monthly_pnl['Month'], y=monthly_pnl['Total_PNL'],
                           marker=dict(color=colors_monthly), name='Monthly P&L',
                           text=[f"₹{v:,.0f}" for v in monthly_pnl['Total_PNL']],
                           textposition='outside'),
                    row=1, col=1
                )
                fig_monthly.add_trace(
                    go.Bar(x=monthly_pnl['Month'], y=monthly_pnl['Trade_Count'],
                           marker=dict(color=ANALYTICS_GRADIENT_BLUE[4]), name='Trade Count',
                           text=monthly_pnl['Trade_Count'], textposition='outside'),
                    row=2, col=1
                )
                fig_monthly.update_layout(
                    title={'text': "📈 Monthly Performance & Trade Count", 'x': 0.5, 'xanchor': 'center'},
                    height=700, showlegend=False
                )
                fig_monthly.update_xaxes(gridcolor='#374151', showgrid=True)
                fig_monthly.update_yaxes(gridcolor='#374151', showgrid=True)
                style_chart(fig_monthly)
                st.plotly_chart(fig_monthly, use_container_width=True)

                # Daily P&L (Optional)
                st.markdown("---")
                df_time_pnl['Date'] = pd.to_datetime(df_time_pnl[date_col_pnl]).dt.date
                daily_pnl = df_time_pnl.groupby('Date')['PNL_num'].sum().reset_index()
                colors_daily = [ANALYTICS_GRADIENT_GREEN[5] if x >= 0 else ANALYTICS_GRADIENT_ORANGE[5] for x in daily_pnl['PNL_num']]
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=daily_pnl['Date'], y=daily_pnl['PNL_num'],
                    marker=dict(color=colors_daily, line=dict(color=BG_END, width=1)),
                    hovertemplate='<b>Date</b>: %{x}<br><b>P&L</b>: ₹%{y:,.2f}<extra></extra>'
                ))
                fig_daily.add_hline(y=0, line_dash="dash", line_color=TEXT_MUTED, line_width=1)
                fig_daily.update_layout(
                    title={'text': "📅 Daily P&L", 'x': 0.5, 'xanchor': 'center'},
                    xaxis_title="Date", yaxis_title="P&L (₹)", height=500,
                    xaxis=dict(gridcolor='#374151', showgrid=True),
                    yaxis=dict(gridcolor='#374151', showgrid=True),
                    hovermode='x unified'
                )
                style_chart(fig_daily)
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("No date column detected for time-based P&L analysis.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='
    text-align: center;
    padding: 10px 0 4px 0;
    margin-top: 4px;
'>
    <p style='color: {TEXT_MUTED}; font-size: 11px; margin: 0;'>
        Powered by Google Drive × Streamlit × Plotly
    </p>
</div>
""", unsafe_allow_html=True)

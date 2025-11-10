# trading_dashboard_optimized.py
# -*- coding: utf-8 -*-
import io
import json
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --------------------------- Page Config ---------------------------
st.set_page_config(
    page_title="Trading Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for better styling
st.markdown("""
<style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }
    
    /* Metric cards with glassmorphism effect */
    .stMetric {
        background: rgba(30, 33, 48, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }
    
    .stMetric label {
        color: #9ca3af !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Expander styling */
    div[data-testid="stExpander"] {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0e1117 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
    }
    
    /* Radio button styling */
    .stRadio > label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 16px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 33, 48, 0.4);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background: transparent;
        color: #9ca3af;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Animated title with gradient
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    '>📈 Trading Analysis Dashboard</h1>
    <p style='color: #9ca3af; font-size: 16px; font-weight: 500;'>
        🚀 Supercharged analytics with real-time insights and stunning visualizations
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------- Google Drive Setup ---------------------------
def _read_secrets_or_fail():
    import streamlit as st
    # Show what sections are present (helps debug)
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

# --------------------------- Optimized Functions ---------------------------

@st.cache_data(show_spinner=False)
def calculate_metrics_pnl(df: pd.DataFrame) -> dict:
    """Calculate PnL metrics"""
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
        'sharpe_ratio': 0.0
    }
    
    if df is None or df.empty or 'PNL' not in df.columns:
        return metrics
    
    pnl_series = pd.to_numeric(df['PNL'], errors='coerce').fillna(0.0)
    
    metrics['total_pnl'] = float(pnl_series.sum())
    metrics['winning_trades'] = int((pnl_series > 0).sum())
    metrics['losing_trades'] = int((pnl_series < 0).sum())
    
    tt = metrics['total_trades']
    metrics['win_rate'] = (metrics['winning_trades'] / tt * 100) if tt else 0.0
    
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    
    if len(wins) > 0:
        metrics['avg_win'] = float(wins.mean())
    if len(losses) > 0:
        metrics['avg_loss'] = float(abs(losses.mean()))
    
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    if gross_loss != 0:
        metrics['profit_factor'] = gross_profit / gross_loss
    
    # Cumulative PnL for drawdown
    cumulative_pnl = pnl_series.cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    metrics['max_drawdown'] = float(abs(drawdown.min()))
    
    # Sharpe ratio (simplified; assumes daily-like sampling)
    if len(pnl_series) > 1 and float(pnl_series.std()) != 0:
        metrics['sharpe_ratio'] = float((pnl_series.mean() / pnl_series.std()) * np.sqrt(252))
    
    return metrics

# =========================== Utilities (place BEFORE sidebar) ===========================

@st.cache_resource
def get_drive_service():
    """Authenticate and return Google Drive service"""
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
    """List CSV/XLSX/Sheets in the folder"""
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
        # return tuples: (id, name, modifiedTime, mimeType)
        return [(f['id'], f['name'], f.get('modifiedTime', 'Unknown'), f.get('mimeType', '')) for f in files]
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
        return []


@st.cache_data(ttl=600, show_spinner=False)
def download_file_from_drive(file_id: str, file_name: str) -> Optional[bytes]:
    """Download file; export Google Sheets to CSV by default"""
    service = get_drive_service()
    if not service:
        return None
    try:
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
        mime = meta.get("mimeType", "")
        buf = io.BytesIO()

        if mime == "application/vnd.google-apps.spreadsheet":
            # Export Google Sheets to CSV (or XLSX if you prefer)
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
    """Convert object-like datetime columns to real datetimes (helps Streamlit Arrow)"""
    for col in df.columns:
        if df[col].dtype == 'object':
            name = str(col).lower().strip()
            # ---- IMPORTANT: never treat TIMEFRAME as a datetime ----
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
    """Load CSV/XLSX (and Google Sheets via export) into a DataFrame"""
    raw = download_file_from_drive(file_id, filename)
    if not raw:
        return None

    bio = io.BytesIO(raw)

    # Decide reader by extension (fallback to CSV for Google Sheets export)
    if filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(bio)
    else:
        # Try CSV first
        try:
            df = pd.read_csv(bio)
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio)

    # Strip column whitespace
    df.columns = df.columns.str.strip()

    # ---- SAFER date detection: NEVER parse TIMEFRAME ----
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def precompute_filter_options_pnl(df: pd.DataFrame) -> Dict:
    opts = {}
    symcol = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in df.columns else 'SYMBOL'
    opts['symbols'] = ['All'] + (sorted(df[symcol].dropna().astype(str).unique()) if symcol in df.columns else [])
    opts['trades'] = ['All'] + (sorted(df['TRADE'].dropna().astype(str).unique()) if 'TRADE' in df.columns else [])
    opts['statuses'] = ['All'] + (sorted(df['STATUS'].dropna().astype(str).unique()) if 'STATUS' in df.columns else [])
    opts['scan_names'] = ['All'] + (sorted(df['SCAN_NAME'].dropna().astype(str).unique()) if 'SCAN_NAME' in df.columns else [])
    opts['curated'] = ['All'] + (sorted(df['CURATED_SIGNAL'].dropna().astype(str).unique()) if 'CURATED_SIGNAL' in df.columns else [])
    opts['fno'] = ['All'] + (sorted(df['FNO'].dropna().astype(str).unique()) if 'FNO' in df.columns else [])
    # date
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


def apply_filters_archive(df: pd.DataFrame, scan, tf, status, alert, curated, start_date, end_date, date_col):
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    if 'SCAN_NAME' in df.columns and scan and 'All' not in scan:
        mask &= df['SCAN_NAME'].isin(scan)
    if 'TIMEFRAME' in df.columns and tf and 'All' not in tf:
        mask &= df['TIMEFRAME'].isin(tf)
    if 'STATUS' in df.columns and status and 'All' not in status:
        mask &= df['STATUS'].isin(status)
    if 'ALERT' in df.columns and alert and 'All' not in alert:
        mask &= df['ALERT'].isin(alert)
    if 'CURATED_SIGNAL' in df.columns and curated and 'All' not in curated:
        mask &= df['CURATED_SIGNAL'].isin(curated)
    if date_col and start_date and end_date and date_col in df.columns:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask &= (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df[mask]


def apply_filters_pnl(df: pd.DataFrame, symbols, trades, statuses, scan_names, curated, fno, start_date, end_date, date_col):
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    symcol = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in df.columns else 'SYMBOL'
    if symcol in df.columns and symbols and 'All' not in symbols:
        mask &= df[symcol].isin(symbols)
    if 'TRADE' in df.columns and trades and 'All' not in trades:
        mask &= df['TRADE'].isin(trades)
    if 'STATUS' in df.columns and statuses and 'All' not in statuses:
        mask &= df['STATUS'].isin(statuses)
    if 'SCAN_NAME' in df.columns and scan_names and 'All' not in scan_names:
        mask &= df['SCAN_NAME'].isin(scan_names)
    if 'CURATED_SIGNAL' in df.columns and curated and 'All' not in curated:
        mask &= df['CURATED_SIGNAL'].isin(curated)
    if 'FNO' in df.columns and fno and 'All' not in fno:
        mask &= df['FNO'].isin(fno)
    if date_col and start_date and end_date and date_col in df.columns:
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask &= (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df[mask]


@st.cache_data(show_spinner=False)
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
    st.markdown("""
    <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin: 0;'>📁 Data Source</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Folder/data selection
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
    <div style='padding: 15px; background: rgba(102, 126, 234, 0.1); border-radius: 10px; border-left: 4px solid #667eea;'>
        <h4 style='color: #667eea; margin: 0 0 10px 0;'>ℹ️ Current Mode</h4>
        <p style='color: #9ca3af; font-size: 13px; margin: 0;'>
            📂 <strong>{data_mode}</strong><br>
            ⚡ Lightning-fast analytics<br>
            🔄 Real-time sync<br>
            📊 Advanced visualizations<br>
            💾 Smart caching
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Main App ---------------------------
if selected_file_name:
    # Progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        with st.spinner(f"⚡ Loading {selected_file_name}..."):
            df = load_and_process_file(selected_file_id, selected_file_name)
    progress_placeholder.empty()
    
    if df is None or df.empty:
        st.error("❌ Failed to load file or file is empty")
        st.stop()
    
    st.success(f"✅ Loaded {len(df):,} records from {selected_file_name} (cached)", icon="🚀")
    
    # ============================= ARCHIVE DATA MODE =============================
    if data_mode == "Archive Data":
        filter_options = precompute_filter_options_archive(df)
        
        # --------------------------- Filters ---------------------------
        with st.expander("🔍 Advanced Filters", expanded=True):
            filter_cols = st.columns(5)
            
            with filter_cols[0]:
                selected_scan = st.multiselect("📊 Strategy", filter_options['scan_names'], default=['All'])
            
            with filter_cols[1]:
                # This will now populate since TIMEFRAME stays as text
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
        
        # --------------------------- Key Metrics ---------------------------
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h2 style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 900;
            '>📊 Performance Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Trades", f"{metrics['total_trades']:,}",
                      help="Total number of trades in filtered dataset")
        with col2:
            st.metric("Winning Trades", f"{metrics.get('winning_trades', 0):,}",
                      delta=f"{metrics.get('hit_rate', 0):.1f}%",
                      help="Trades that hit any target (T1/T2/T3)")
        with col3:
            losing = metrics.get('losing_trades', 0)
            total = max(metrics['total_trades'], 1)
            st.metric("Losing Trades", f"{losing:,}",
                      delta=f"-{(losing/total*100):.1f}%",
                      delta_color="inverse",
                      help="Trades that hit stop loss")
        with col4:
            st.metric("Breakeven", f"{metrics.get('breakeven_trades', 0):,}",
                      help="Trades that lapsed without hitting targets or SL")
        with col5:
            win_loss_ratio = metrics.get('winning_trades', 0) / max(metrics.get('losing_trades', 1), 1)
            st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}",
                      help="Ratio of winning to losing trades")
        
        st.markdown("""
        <div style='text-align: center; margin: 30px 0 10px 0;'>
            <h3 style='
                background: linear-gradient(135deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 24px;
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
                    delta=f"{rate:.1f}%",
                    help=f"Percentage of trades that reached Target {i}"
                )
        
        # --------------------------- Visualizations ---------------------------
        st.markdown("""
        <div style='text-align: center; margin: 40px 0 20px 0;'>
            <h2 style='
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 900;
            '>📈 Advanced Analytics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        viz_tabs = st.tabs(["📊 Overview", "🎯 Performance", "📈 Time Series", "🔥 Heatmaps"])
        
        with viz_tabs[0]:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    status_counts = filtered_df['STATUS'].value_counts()
                    colors_status = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=status_counts.index,
                        values=status_counts.values,
                        hole=0.5,
                        marker=dict(colors=colors_status, line=dict(color='#1a1d29', width=3)),
                        textposition='inside',
                        textinfo='percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    fig_pie.update_layout(
                        title={'text': "📊 Trade Status Distribution", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        showlegend=True, height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05,
                                    bgcolor='rgba(30, 33, 48, 0.8)', bordercolor='rgba(255, 255, 255, 0.1)', borderwidth=1)
                    )
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
                            fig_winrate = go.Figure(go.Bar(
                                x=win_rate.values, y=win_rate.index, orientation='h',
                                marker=dict(color=win_rate.values, colorscale='RdYlGn', showscale=True,
                                            colorbar=dict(title="Win Rate %", thickness=15,
                                                          bgcolor='rgba(30, 33, 48, 0.8)')),
                                text=[f"{val:.1f}%" for val in win_rate.values], textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>'
                            ))
                            fig_winrate.update_layout(
                                title={'text': "🏆 Win Rate by Strategy", 'x': 0.5, 'xanchor': 'center',
                                       'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                                xaxis_title="Win Rate (%)", yaxis_title="", height=450,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                                xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                            )
                            st.plotly_chart(fig_winrate, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create win rate chart: {e}")
            
            viz_col3, viz_col4 = st.columns(2)
            with viz_col3:
                if 'TIMEFRAME' in filtered_df.columns and not filtered_df.empty:
                    tf_counts = filtered_df['TIMEFRAME'].astype(str).value_counts().sort_values(ascending=True)
                    fig_tf = go.Figure(go.Bar(
                        x=tf_counts.values, y=tf_counts.index, orientation='h',
                        marker=dict(color=tf_counts.values, colorscale='Viridis', showscale=False,
                                    line=dict(color='#1e40af', width=1.5)),
                        text=tf_counts.values, textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Trades: %{x}<extra></extra>'
                    ))
                    fig_tf.update_layout(
                        title={'text': "⏰ Trades by Timeframe", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="Number of Trades", yaxis_title="", height=450,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                        xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                    )
                    st.plotly_chart(fig_tf, use_container_width=True)
            with viz_col4:
                if 'ALERT' in filtered_df.columns and not filtered_df.empty:
                    alert_counts = filtered_df['ALERT'].value_counts()
                    colors_alert = ['#8b5cf6', '#ec4899', '#f97316', '#14b8a6', '#eab308']
                    fig_alert = go.Figure(data=[go.Pie(
                        labels=alert_counts.index, values=alert_counts.values, hole=0.6,
                        marker=dict(colors=colors_alert, line=dict(color='#1a1d29', width=3)),
                        textposition='inside', textinfo='percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    fig_alert.update_layout(
                        title={'text': "🔔 Alert Type Distribution", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        showlegend=True, height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        legend=dict(bgcolor='rgba(30, 33, 48, 0.8)', bordercolor='rgba(255, 255, 255, 0.1)', borderwidth=1)
                    )
                    st.plotly_chart(fig_alert, use_container_width=True)
        
        with viz_tabs[1]:
            viz_col5, viz_col6 = st.columns(2)
            with viz_col5:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    breakdown_data = {
                        'Category': ['Wins', 'Losses', 'Breakeven'],
                        'Count': [metrics.get('winning_trades', 0),
                                  metrics.get('losing_trades', 0),
                                  metrics.get('breakeven_trades', 0)],
                        'Color': ['#10b981', '#ef4444', '#6b7280']
                    }
                    fig_breakdown = go.Figure(go.Bar(
                        x=breakdown_data['Category'], y=breakdown_data['Count'],
                        marker=dict(color=breakdown_data['Color'], line=dict(color='#1a1d29', width=2)),
                        text=breakdown_data['Count'], textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    ))
                    fig_breakdown.update_layout(
                        title={'text': "📊 Win/Loss/Breakeven Breakdown", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="", yaxis_title="Number of Trades", height=450,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                        xaxis=dict(gridcolor='#374151'), yaxis=dict(gridcolor='#374151', showgrid=True)
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)
            with viz_col6:
                if 'SCAN_NAME' in filtered_df.columns and not filtered_df.empty:
                    try:
                        trades_by_scan = filtered_df['SCAN_NAME'].value_counts().sort_values(ascending=True)
                        fig_scan = go.Figure(go.Bar(
                            x=trades_by_scan.values, y=trades_by_scan.index, orientation='h',
                            marker=dict(color='#4f46e5'),
                            text=trades_by_scan.values, textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Trades: %{x}<extra></extra>'
                        ))
                        fig_scan.update_layout(
                            title={'text': "📌 Trades by Strategy", 'x': 0.5, 'xanchor': 'center',
                                   'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                            xaxis_title="Trades", yaxis_title="", height=450,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                            xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151')
                        )
                        st.plotly_chart(fig_scan, use_container_width=True)
                    except Exception:
                        pass
        
        with viz_tabs[2]:
            if 'date_col' in filter_options and filter_options['date_col'] in filtered_df.columns:
                date_col = filter_options['date_col']
                df_time = filtered_df.dropna(subset=[date_col]).copy()
                df_time = df_time.sort_values(date_col)
                if not df_time.empty:
                    df_time['cumulative_trades'] = range(1, len(df_time) + 1)
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=df_time[date_col], y=df_time['cumulative_trades'], mode='lines',
                        name='Cumulative Trades', line=dict(color='#3b82f6', width=4),
                        fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.3)',
                        hovertemplate='<b>Date</b>: %{x}<br><b>Total Trades</b>: %{y}<extra></extra>'
                    ))
                    fig_timeline.update_layout(
                        title={'text': "📈 Cumulative Trades Over Time", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 22, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="Date", yaxis_title="Total Trades", height=500,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                        xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151', showgrid=True),
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    if 'STATUS' in df_time.columns:
                        df_time['is_win'] = df_time['STATUS'].astype(str).str.upper().str.contains(
                            'T1 MET|T2 MET|T3 MET', regex=True, na=False).astype(int)
                        window_size = min(20, len(df_time))
                        df_time['rolling_win_rate'] = df_time['is_win'].rolling(window=window_size, min_periods=1).mean() * 100
                        fig_winrate_time = go.Figure()
                        fig_winrate_time.add_trace(go.Scatter(
                            x=df_time[date_col], y=df_time['rolling_win_rate'], mode='lines',
                            name=f'Win Rate ({window_size}-trade MA)', line=dict(color='#10b981', width=4),
                            fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.2)',
                            hovertemplate='<b>Date</b>: %{x}<br><b>Win Rate</b>: %{y:.1f}%<extra></extra>'
                        ))
                        fig_winrate_time.add_hline(y=50, line_dash="dash", line_color="#9ca3af", line_width=2,
                                                   annotation_text="50% Baseline", annotation_position="right")
                        fig_winrate_time.update_layout(
                            title={'text': f"🎯 Rolling Win Rate ({window_size}-Trade MA)", 'x': 0.5, 'xanchor': 'center',
                                   'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                            xaxis_title="Date", yaxis_title="Win Rate (%)", height=450,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                            xaxis=dict(gridcolor='#374151', showgrid=True), yaxis=dict(gridcolor='#374151', showgrid=True),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_winrate_time, use_container_width=True)
        
        with viz_tabs[3]:
            if 'SCAN_NAME' in filtered_df.columns and 'TIMEFRAME' in filtered_df.columns:
                try:
                    heatmap_data = pd.crosstab(filtered_df['SCAN_NAME'], filtered_df['TIMEFRAME'])
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                        colorscale='YlOrRd', text=heatmap_data.values, texttemplate='%{text}',
                        textfont={"size": 14, "color": "white"},
                        hovertemplate='<b>Strategy</b>: %{y}<br><b>Timeframe</b>: %{x}<br><b>Trades</b>: %{z}<extra></extra>',
                        colorbar=dict(title="Trades", thickness=20, bgcolor='rgba(30, 33, 48, 0.8)')
                    ))
                    fig_heatmap.update_layout(
                        title={'text': "🔥 Strategy × Timeframe Heatmap", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 22, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="Timeframe", yaxis_title="Strategy", height=600,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12)
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create heatmap: {e}")
        
        # --------------------------- Data Table & Export ---------------------------
        st.markdown("---")
        st.markdown("### 🧾 Filtered Trades")
        st.dataframe(filtered_df, width="stretch", height=420)
        
        export_cols = st.columns(3)
        with export_cols[0]:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Filtered Data (CSV)", data=csv_data,
                               file_name=f"archive_filtered_{selected_file_name.split('.')[0]}.csv",
                               mime="text/csv", use_container_width=True)
        with export_cols[1]:
            metrics_df = pd.DataFrame([metrics]).T
            metrics_df.columns = ['Value']
            st.download_button("⬇️ Download Metrics (CSV)",
                               data=metrics_df.to_csv().encode('utf-8'),
                               file_name=f"archive_metrics_{selected_file_name.split('.')[0]}.csv",
                               mime="text/csv", use_container_width=True)
        with export_cols[2]:
            summary_report = f"""
ARCHIVE PERFORMANCE SUMMARY
{'='*50}
File: {selected_file_name}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall
{'-'*20}
Total Trades: {metrics['total_trades']:,}
Wins: {metrics['winning_trades']:,} ({metrics.get('hit_rate', 0):.2f}%)
Losses: {metrics['losing_trades']:,}
Breakeven: {metrics['breakeven_trades']:,}

Targets
{'-'*20}
T1 Hit: {metrics['target1_met']:,} ({metrics['target1_rate']:.2f}%)
T2 Hit: {metrics['target2_met']:,} ({metrics['target2_rate']:.2f}%)
T3 Hit: {metrics['target3_met']:,} ({metrics['target3_rate']:.2f}%)
"""
            st.download_button("⬇️ Download Summary (TXT)", data=summary_report,
                               file_name=f"archive_summary_{selected_file_name.split('.')[0]}.txt",
                               mime="text/plain", use_container_width=True)
    
    # ============================= PNL DATA MODE =============================
    elif data_mode == "PnL Data":
        filter_options = precompute_filter_options_pnl(df)
        
        # --------------------------- PnL Filters ---------------------------
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
        
        pnl_metrics = calculate_metrics_pnl(filtered_df)
        
        # --------------------------- PnL Key Metrics ---------------------------
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h2 style='
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 900;
            '>💰 Profit & Loss Metrics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        pnl_col1, pnl_col2, pnl_col3, pnl_col4, pnl_col5 = st.columns(5)
        with pnl_col1:
            pnl_val = pnl_metrics['total_pnl']
            st.metric("Total P&L", f"₹{pnl_val:,.2f}", 
                      delta="Profit" if pnl_val > 0 else "Loss",
                      delta_color="normal" if pnl_val > 0 else "inverse",
                      help="Total profit/loss across all trades")
        with pnl_col2:
            st.metric("Total Trades", f"{pnl_metrics['total_trades']:,}",
                      help="Total number of trades executed")
        with pnl_col3:
            st.metric("Win Rate", f"{pnl_metrics['win_rate']:.1f}%",
                      delta=f"{pnl_metrics['winning_trades']:,} wins",
                      help="Percentage of profitable trades")
        with pnl_col4:
            st.metric("Profit Factor", f"{pnl_metrics['profit_factor']:.2f}",
                      help="Gross profit divided by gross loss")
        with pnl_col5:
            st.metric("Sharpe Ratio", f"{pnl_metrics['sharpe_ratio']:.2f}",
                      help="Risk-adjusted return metric")
        
        pnl_col6, pnl_col7, pnl_col8, pnl_col9, pnl_col10 = st.columns(5)
        with pnl_col6:
            st.metric("Winning Trades", f"{pnl_metrics['winning_trades']:,}",
                      help="Number of profitable trades")
        with pnl_col7:
            st.metric("Losing Trades", f"{pnl_metrics['losing_trades']:,}",
                      delta=f"-{pnl_metrics['losing_trades']:,}",
                      delta_color="inverse",
                      help="Number of loss-making trades")
        with pnl_col8:
            st.metric("Avg Win", f"₹{pnl_metrics['avg_win']:,.2f}",
                      help="Average profit per winning trade")
        with pnl_col9:
            st.metric("Avg Loss", f"₹{pnl_metrics['avg_loss']:,.2f}",
                      delta=f"-₹{pnl_metrics['avg_loss']:,.2f}",
                      delta_color="inverse",
                      help="Average loss per losing trade")
        with pnl_col10:
            st.metric("Max Drawdown", f"₹{pnl_metrics['max_drawdown']:,.2f}",
                      delta="Risk", delta_color="inverse",
                      help="Maximum peak-to-trough decline")
        
        # --------------------------- PnL Visualizations ---------------------------
        st.markdown("""
        <div style='text-align: center; margin: 40px 0 20px 0;'>
            <h2 style='
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 900;
            '>📊 P&L Analytics Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        pnl_tabs = st.tabs(["💹 Equity Curve", "📊 Performance Analysis", "🎯 Trade Distribution", "📈 Time-Based Analysis"])
        
        # TAB 1: Equity Curve
        with pnl_tabs[0]:
            if 'PNL' in filtered_df.columns:
                df_equity = filtered_df.copy()
                date_col_pnl = filter_options.get('date_col')
                if date_col_pnl:
                    df_equity = df_equity.dropna(subset=[date_col_pnl, 'PNL'])
                    df_equity = df_equity.sort_values(date_col_pnl)
                    df_equity['PNL_num'] = pd.to_numeric(df_equity['PNL'], errors='coerce').fillna(0)
                    df_equity['Cumulative_PNL'] = df_equity['PNL_num'].cumsum()
                    df_equity['Running_Max'] = df_equity['Cumulative_PNL'].cummax()
                    df_equity['Drawdown'] = df_equity['Cumulative_PNL'] - df_equity['Running_Max']
                    
                    fig_equity = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("💰 Cumulative P&L Equity Curve", "📉 Drawdown"),
                        vertical_spacing=0.1
                    )
                    fig_equity.add_trace(
                        go.Scatter(
                            x=df_equity[date_col_pnl],
                            y=df_equity['Cumulative_PNL'],
                            mode='lines',
                            name='Cumulative P&L',
                            line=dict(color='#10b981', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(16, 185, 129, 0.2)',
                            hovertemplate='<b>Date</b>: %{x}<br><b>P&L</b>: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    fig_equity.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1, row=1, col=1)
                    fig_equity.add_trace(
                        go.Scatter(
                            x=df_equity[date_col_pnl],
                            y=df_equity['Drawdown'],
                            mode='lines',
                            name='Drawdown',
                            line=dict(color='#ef4444', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(239, 68, 68, 0.3)',
                            hovertemplate='<b>Date</b>: %{x}<br><b>Drawdown</b>: ₹%{y:,.2f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    fig_equity.update_layout(
                        title={'text': "🚀 P&L Equity Curve with Drawdown Analysis", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 24, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        height=700,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        showlegend=True,
                        hovermode='x unified'
                    )
                    fig_equity.update_xaxes(gridcolor='#374151', showgrid=True)
                    fig_equity.update_yaxes(gridcolor='#374151', showgrid=True)
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    eq_col1, eq_col2, eq_col3, eq_col4 = st.columns(4)
                    with eq_col1:
                        max_pnl = df_equity['Cumulative_PNL'].max()
                        st.metric("Peak P&L", f"₹{max_pnl:,.2f}", help="Highest cumulative profit reached")
                    with eq_col2:
                        min_pnl = df_equity['Cumulative_PNL'].min()
                        st.metric("Lowest P&L", f"₹{min_pnl:,.2f}", help="Lowest cumulative point")
                    with eq_col3:
                        final_pnl = df_equity['Cumulative_PNL'].iloc[-1]
                        st.metric("Final P&L", f"₹{final_pnl:,.2f}", help="Ending cumulative P&L")
                    with eq_col4:
                        max_dd = df_equity['Drawdown'].min()
                        peak = max(df_equity['Cumulative_PNL'].max(), 1)
                        st.metric("Max Drawdown", f"₹{max_dd:,.2f}",
                                  delta=f"{(max_dd/peak*100):.1f}%",
                                  delta_color="inverse",
                                  help="Maximum peak-to-trough decline")
                else:
                    st.warning("📅 No date column found for equity curve visualization")
            else:
                st.error("❌ PNL column not found in data")
        
        # TAB 2: Performance Analysis
        with pnl_tabs[1]:
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                if 'PNL' in filtered_df.columns and not filtered_df.empty:
                    pnl_series = pd.to_numeric(filtered_df['PNL'], errors='coerce').fillna(0)
                    win_loss_data = pd.DataFrame({
                        'Category': ['Wins', 'Losses'],
                        'Count': [(pnl_series > 0).sum(), (pnl_series < 0).sum()],
                        'Amount': [pnl_series[pnl_series > 0].sum(), abs(pnl_series[pnl_series < 0].sum())]
                    })
                    fig_winloss = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Trade Count", "Total Amount (₹)"),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )
                    colors = ['#10b981', '#ef4444']
                    fig_winloss.add_trace(
                        go.Bar(x=win_loss_data['Category'], y=win_loss_data['Count'],
                               marker=dict(color=colors), text=win_loss_data['Count'],
                               textposition='outside', name='Count'),
                        row=1, col=1
                    )
                    fig_winloss.add_trace(
                        go.Bar(x=win_loss_data['Category'], y=win_loss_data['Amount'],
                               marker=dict(color=colors), text=[f"₹{v:,.0f}" for v in win_loss_data['Amount']],
                               textposition='outside', name='Amount'),
                        row=1, col=2
                    )
                    fig_winloss.update_layout(
                        title={'text': "🎯 Win/Loss Breakdown", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        height=450,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        showlegend=False
                    )
                    fig_winloss.update_xaxes(gridcolor='#374151')
                    fig_winloss.update_yaxes(gridcolor='#374151', showgrid=True)
                    st.plotly_chart(fig_winloss, use_container_width=True)
            with perf_col2:
                if 'PNL' in filtered_df.columns and not filtered_df.empty:
                    pnl_data = pd.to_numeric(filtered_df['PNL'], errors='coerce').dropna()
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=pnl_data, nbinsx=50,
                        marker=dict(color=pnl_data, colorscale='RdYlGn',
                                    line=dict(color='#1a1d29', width=1)),
                        hovertemplate='<b>P&L Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
                    ))
                    fig_dist.add_vline(x=0, line_dash="dash", line_color="#9ca3af", line_width=2,
                                       annotation_text="Breakeven", annotation_position="top")
                    fig_dist.update_layout(
                        title={'text': "📊 P&L Distribution", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="P&L (₹)", yaxis_title="Frequency", height=450,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                        xaxis=dict(gridcolor='#374151', showgrid=True),
                        yaxis=dict(gridcolor='#374151', showgrid=True)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            if 'SCAN_NAME' in filtered_df.columns and 'PNL' in filtered_df.columns and not filtered_df.empty:
                st.markdown("---")
                pnl_by_strategy = filtered_df.groupby('SCAN_NAME')['PNL'].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').sum()
                ).sort_values(ascending=True)
                fig_strategy_pnl = go.Figure(go.Bar(
                    x=pnl_by_strategy.values,
                    y=pnl_by_strategy.index,
                    orientation='h',
                    marker=dict(
                        color=pnl_by_strategy.values,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L (₹)", thickness=15, bgcolor='rgba(30, 33, 48, 0.8)')
                    ),
                    text=[f"₹{val:,.0f}" for val in pnl_by_strategy.values],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>P&L: ₹%{x:,.2f}<extra></extra>'
                ))
                fig_strategy_pnl.update_layout(
                    title={'text': "💼 P&L by Strategy/Scan", 'x': 0.5, 'xanchor': 'center',
                           'font': {'size': 22, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                    xaxis_title="Total P&L (₹)", yaxis_title="", height=500,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                    xaxis=dict(gridcolor='#374151', showgrid=True),
                    yaxis=dict(gridcolor='#374151')
                )
                st.plotly_chart(fig_strategy_pnl, use_container_width=True)
        
        # TAB 3: Trade Distribution
        with pnl_tabs[2]:
            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                if 'TRADE' in filtered_df.columns and not filtered_df.empty:
                    trade_counts = filtered_df['TRADE'].value_counts()
                    fig_trade = go.Figure(data=[go.Pie(
                        labels=trade_counts.index,
                        values=trade_counts.values,
                        hole=0.5,
                        marker=dict(
                            colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'],
                            line=dict(color='#1a1d29', width=3)
                        ),
                        textposition='inside',
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
                    )])
                    fig_trade.update_layout(
                        title={'text': "📈 Trade Type Distribution", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        height=450,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        showlegend=True
                    )
                    st.plotly_chart(fig_trade, use_container_width=True)
            with dist_col2:
                if 'STATUS' in filtered_df.columns and not filtered_df.empty:
                    status_counts = filtered_df['STATUS'].value_counts()
                    fig_status = go.Figure(go.Bar(
                        x=status_counts.index,
                        y=status_counts.values,
                        marker=dict(
                            color=status_counts.values,
                            colorscale='Viridis',
                            showscale=False,
                            line=dict(color='#1a1d29', width=2)
                        ),
                        text=status_counts.values,
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    ))
                    fig_status.update_layout(
                        title={'text': "🎯 Status Distribution", 'x': 0.5, 'xanchor': 'center',
                               'font': {'size': 20, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                        xaxis_title="",
                        yaxis_title="Count",
                        height=450,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb', size=12),
                        xaxis=dict(gridcolor='#374151', tickangle=-45),
                        yaxis=dict(gridcolor='#374151', showgrid=True)
                    )
                    st.plotly_chart(fig_status, use_container_width=True)
            
            if 'PNL' in filtered_df.columns and not filtered_df.empty:
                st.markdown("---")
                st.markdown("### 🏆 Top Performers & 📉 Worst Performers")
                symbol_col = 'TRADING_SYMBOL' if 'TRADING_SYMBOL' in filtered_df.columns else 'SYMBOL'
                if symbol_col in filtered_df.columns:
                    top_col1, top_col2 = st.columns(2)
                    with top_col1:
                        top_winners = filtered_df.copy()
                        top_winners['PNL_num'] = pd.to_numeric(top_winners['PNL'], errors='coerce')
                        top_winners = top_winners.nlargest(10, 'PNL_num')[[symbol_col, 'PNL', 'TRADE' if 'TRADE' in filtered_df.columns else symbol_col]]
                        st.markdown("#### 🥇 Top 10 Winning Trades")
                        st.dataframe(
                            top_winners.style.background_gradient(subset=['PNL'], cmap='Greens'),
                            width="stretch",
                            hide_index=True,
                            height=400
                        )
                    with top_col2:
                        top_losers = filtered_df.copy()
                        top_losers['PNL_num'] = pd.to_numeric(top_losers['PNL'], errors='coerce')
                        top_losers = top_losers.nsmallest(10, 'PNL_num')[[symbol_col, 'PNL', 'TRADE' if 'TRADE' in filtered_df.columns else symbol_col]]
                        st.markdown("#### 📉 Top 10 Losing Trades")
                        st.dataframe(
                            top_losers.style.background_gradient(subset=['PNL'], cmap='Reds_r'),
                            width="stretch",
                            hide_index=True,
                            height=400
                        )
        
        # TAB 4: Time-Based Analysis
        with pnl_tabs[3]:
            date_col_pnl = filter_options.get('date_col')
            if date_col_pnl and 'PNL' in filtered_df.columns and not filtered_df.empty:
                df_time_pnl = filtered_df.dropna(subset=[date_col_pnl, 'PNL']).copy()
                df_time_pnl['PNL_num'] = pd.to_numeric(df_time_pnl['PNL'], errors='coerce')
                df_time_pnl = df_time_pnl.sort_values(date_col_pnl)
                
                # Daily P&L
                df_time_pnl['Date'] = pd.to_datetime(df_time_pnl[date_col_pnl]).dt.date
                daily_pnl = df_time_pnl.groupby('Date')['PNL_num'].sum().reset_index()
                fig_daily = go.Figure()
                colors = ['#10b981' if x >= 0 else '#ef4444' for x in daily_pnl['PNL_num']]
                fig_daily.add_trace(go.Bar(
                    x=daily_pnl['Date'],
                    y=daily_pnl['PNL_num'],
                    marker=dict(color=colors, line=dict(color='#1a1d29', width=1)),
                    hovertemplate='<b>Date</b>: %{x}<br><b>P&L</b>: ₹%{y:,.2f}<extra></extra>'
                ))
                fig_daily.add_hline(y=0, line_dash="dash", line_color="#9ca3af", line_width=1)
                fig_daily.update_layout(
                    title={'text': "📅 Daily P&L", 'x': 0.5, 'xanchor': 'center',
                           'font': {'size': 22, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                    xaxis_title="Date", yaxis_title="P&L (₹)", height=500,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb', size=12),
                    xaxis=dict(gridcolor='#374151', showgrid=True),
                    yaxis=dict(gridcolor='#374151', showgrid=True),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_daily, use_container_width=True)
                
                # Monthly Performance
                df_time_pnl['Month'] = pd.to_datetime(df_time_pnl[date_col_pnl]).dt.to_period('M').astype(str)
                monthly_pnl = df_time_pnl.groupby('Month').agg({
                    'PNL_num': ['sum', 'count', 'mean']
                }).reset_index()
                monthly_pnl.columns = ['Month', 'Total_PNL', 'Trade_Count', 'Avg_PNL']
                
                fig_monthly = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("💰 Monthly P&L", "📊 Trade Count"),
                    row_heights=[0.6, 0.4],
                    vertical_spacing=0.15
                )
                colors_monthly = ['#10b981' if x >= 0 else '#ef4444' for x in monthly_pnl['Total_PNL']]
                fig_monthly.add_trace(
                    go.Bar(x=monthly_pnl['Month'], y=monthly_pnl['Total_PNL'],
                           marker=dict(color=colors_monthly), name='Monthly P&L',
                           text=[f"₹{v:,.0f}" for v in monthly_pnl['Total_PNL']],
                           textposition='outside'),
                    row=1, col=1
                )
                fig_monthly.add_trace(
                    go.Bar(x=monthly_pnl['Month'], y=monthly_pnl['Trade_Count'],
                           marker=dict(color='#667eea'), name='Trade Count',
                           text=monthly_pnl['Trade_Count'], textposition='outside'),
                    row=2, col=1
                )
                fig_monthly.update_layout(
                    title={'text': "📈 Monthly Performance Summary", 'x': 0.5, 'xanchor': 'center',
                           'font': {'size': 22, 'color': '#e5e7eb', 'family': 'Arial Black'}},
                    height=700,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb', size=12),
                    showlegend=False
                )
                fig_monthly.update_xaxes(gridcolor='#374151', showgrid=True)
                fig_monthly.update_yaxes(gridcolor='#374151', showgrid=True)
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("No date column detected for time-based P&L analysis.")
        
        # --------------------------- Data Table & Export ---------------------------
        st.markdown("---")
        st.markdown("### 🧾 Filtered P&L Data")
        st.dataframe(filtered_df, width="stretch", height=420)
        
        export_cols = st.columns(3)
        with export_cols[0]:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Filtered Data (CSV)", data=csv_data,
                               file_name=f"pnl_filtered_{selected_file_name.split('.')[0]}.csv",
                               mime="text/csv", use_container_width=True)
        with export_cols[1]:
            metrics_df = pd.DataFrame([pnl_metrics]).T
            metrics_df.columns = ['Value']
            st.download_button("⬇️ Download Metrics (CSV)",
                               data=metrics_df.to_csv().encode('utf-8'),
                               file_name=f"pnl_metrics_{selected_file_name.split('.')[0]}.csv",
                               mime="text/csv", use_container_width=True)
        with export_cols[2]:
            summary_report = f"""
PNL PERFORMANCE SUMMARY
{'='*50}
File: {selected_file_name}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall
{'-'*20}
Total Trades: {pnl_metrics['total_trades']:,}
Wins: {pnl_metrics['winning_trades']:,}
Losses: {pnl_metrics['losing_trades']:,}
Win Rate: {pnl_metrics['win_rate']:.2f}%
Total P&L: ₹{pnl_metrics['total_pnl']:,.2f}
Profit Factor: {pnl_metrics['profit_factor']:.2f}
Sharpe Ratio: {pnl_metrics['sharpe_ratio']:.2f}
Max Drawdown: ₹{pnl_metrics['max_drawdown']:,.2f}
"""
            st.download_button("⬇️ Download Summary (TXT)", data=summary_report,
                               file_name=f"pnl_summary_{selected_file_name.split('.')[0]}.txt",
                               mime="text/plain", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='
    text-align: center;
    padding: 30px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border-radius: 15px;
    margin-top: 20px;
'>
    <h3 style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 10px;
    '>📈 Trading Analysis Dashboard</h3>
    <p style='color: #9ca3af; font-size: 12px; margin: 5px 0;'>
        ⚡ Lightning-fast processing • 🔄 Real-time Drive sync • 📊 Advanced analytics • 💾 Smart caching
    </p>
    <p style='color: #6b7280; font-size: 11px; margin-top: 8px;'>
        Powered by Google Drive × Streamlit × Plotly
    </p>
</div>
""", unsafe_allow_html=True)

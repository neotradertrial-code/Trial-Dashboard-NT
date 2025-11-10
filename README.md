# Trading Analysis Dashboard

A Streamlit-based dashboard for analyzing trading data from Google Drive.

## Features
- 📊 Archive Data Analysis
- 💰 P&L Performance Tracking
- 📈 Advanced Visualizations
- 🔄 Real-time Google Drive Sync

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add secrets in Streamlit Cloud dashboard

### Required Secrets
Add these in Streamlit Cloud Settings → Secrets:
- Copy contents from `.streamlit/secrets.toml`
- Paste into Streamlit Cloud secrets section

## Local Development
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` with your credentials
4. Run: `streamlit run trading_dashboard_optimized.py`
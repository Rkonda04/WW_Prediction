# Richmond Regional WWTF - Flow Prediction Dashboard

A clean, interactive dashboard for visualizing XGBoost flow prediction model performance.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

- **Period Selection**: View predictions for any validation year or all years combined
- **Performance Metrics**: Real-time R², RMSE, accuracy, and data point counts
- **Interactive Visualizations**: Time-series plots with observed vs predicted flow
- **Residual Analysis**: Distribution of prediction errors and top misestimated days
- **Flagged Days**: View screened no-rain spike days (2024 test year)
- **Responsive Design**: Works on desktop and tablet

## Dashboard Sections

1. **Period Overview** - Select which time period to analyze
2. **Performance Metrics** - Key performance indicators
3. **Time-Series Plot** - Observed vs predicted daily flow visualization
4. **Residual Analysis** (optional) - Dive into prediction errors
5. **Screened Days** (2024 only) - Days excluded from model scoring

## Deployment to Vercel

For future Vercel deployment, use one of these options:

### Option A: Python Backend (Recommended for Vercel)
Convert to FastAPI/Flask with a React frontend:
- Backend: Deploy to Railway/Render
- Frontend: Deploy to Vercel

### Option B: Streamlit Cloud
Deploy directly to Streamlit Cloud:
```bash
git push origin main
```
Then enable Streamlit Cloud in your repo settings.

## Data Requirements

The app expects these files:
- `output/13_screened_model/out_of_sample_predictions_all_years.csv`
- `output/13_screened_model/screened_days.csv`

## Styling

The dashboard uses:
- Custom Matplotlib styling for consistency with your analysis scripts
- Streamlit's native components for responsive layout
- Accessible color palette (validated for CVD compatibility)

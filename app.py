import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="WWTF Flow Prediction", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "output" / "13_screened_model"
OOS_CSV = MODEL_DIR / "out_of_sample_predictions_all_years.csv"
SCREENED_CSV = MODEL_DIR / "screened_days.csv"

OBSERVED = "#2a78d6"
PREDICTED = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e4e1"
FLAG = "#e34948"

@st.cache_data
def load_data():
    df = pd.read_csv(OOS_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    screened = pd.read_csv(SCREENED_CSV, parse_dates=["Date"])
    return df, screened

@st.cache_data
def get_metrics(sub):
    err = sub["p"] - sub["y"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((sub["y"] - sub["y"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / sub["y"]).mean() * 100)
    return {"R²": r2, "RMSE": rmse, "Accuracy": 100 - mape, "Points": len(sub)}

def plot_timeseries(sub, title, show_flags=False, screened=None):
    fig, ax = plt.subplots(figsize=(12, 4))

    if show_flags and screened is not None:
        for _, row in screened.iterrows():
            ax.axvspan(row["Date"] - pd.Timedelta(hours=14),
                       row["Date"] + pd.Timedelta(hours=14),
                       color=FLAG, alpha=0.11, linewidth=0, zorder=1)

    gap = sub["Date"].diff().dt.days.fillna(1) > 1
    for _, chunk in sub.groupby(gap.cumsum()):
        ax.plot(chunk["Date"], chunk["y"], color=OBSERVED, linewidth=2,
                solid_capstyle="round", zorder=3, label="Observed" if _ == 0 else "")
        ax.plot(chunk["Date"], chunk["p"], color=PREDICTED, linewidth=2,
                solid_capstyle="round", zorder=4, label="Predicted" if _ == 0 else "")

    ax.set_ylabel("Daily Treated Flow (MGD)", fontsize=11, color=INK_SECONDARY)
    ax.set_xlabel("Date", fontsize=11, color=INK_SECONDARY)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.5)
    ax.tick_params(colors=INK_SECONDARY)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    plt.tight_layout()

    return fig

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f8f7;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2a78d6;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #0b0b0b;
    }
    .metric-label {
        font-size: 12px;
        color: #8a8985;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Richmond Regional WWTF - Flow Prediction")
st.markdown("**XGBoost Model Predictions vs Observed Daily Flow**")
st.divider()

try:
    df, screened = load_data()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Period Overview")
        periods = ["All Years", "Fold 1 val 2020", "Fold 2 val 2021", "Fold 3 val 2022", "TEST 2024"]
        selected_period = st.selectbox("Select period:", periods, index=4)

    with col2:
        st.markdown("### View Options")
        show_residuals = st.checkbox("Show residuals", value=False)

    st.divider()

    if selected_period == "All Years":
        data_to_show = df
        title_text = "All Out-of-Sample Years"
        show_flags = False
    else:
        data_to_show = df[df["Period"] == selected_period].copy()
        title_text = f"Period: {selected_period}"
        show_flags = (selected_period == "TEST 2024")

    metrics = get_metrics(data_to_show)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metrics['R²']:.3f}", "Performance fit")
    with col2:
        st.metric("RMSE (MGD)", f"{metrics['RMSE']:.3f}", "Error magnitude")
    with col3:
        st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%", "Mean accuracy")
    with col4:
        st.metric("Data Points", f"{metrics['Points']}", "Days analyzed")

    st.divider()

    screened_2024 = screened[screened["Date"].dt.year == 2024] if show_flags else None
    fig = plot_timeseries(data_to_show, title_text, show_flags, screened_2024)
    st.pyplot(fig)

    if show_residuals and len(data_to_show) > 0:
        st.markdown("### Residual Analysis")
        data_to_show["residual"] = data_to_show["y"] - data_to_show["p"]

        col1, col2 = st.columns(2)
        with col1:
            fig_resid, ax = plt.subplots(figsize=(6, 3))
            ax.hist(data_to_show["residual"], bins=30, color=PREDICTED, alpha=0.7, edgecolor=INK)
            ax.set_xlabel("Residual (MGD)", color=INK_SECONDARY)
            ax.set_ylabel("Frequency", color=INK_SECONDARY)
            ax.set_title("Distribution of Residuals", fontweight="bold", color=INK)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(colors=INK_SECONDARY)
            fig_resid.patch.set_facecolor("#fcfcfb")
            ax.set_facecolor("#fcfcfb")
            plt.tight_layout()
            st.pyplot(fig_resid)

        with col2:
            top_errors = data_to_show.nlargest(5, "residual")[["Date", "y", "p", "residual"]]
            st.markdown("**Top 5 Underestimated Days** (highest positive residuals)")
            for idx, row in top_errors.iterrows():
                st.write(f"{row['Date']:%Y-%m-%d}: Obs={row['y']:.2f}, Pred={row['p']:.2f}, Diff=+{row['residual']:.2f} MGD")

    if show_flags and len(screened_2024) > 0:
        st.divider()
        st.markdown("### Screened No-Rain Spike Days (2024)")
        st.info(f"**{len(screened_2024)}** days excluded from scoring due to anomalous no-rain spikes")
        screened_display = screened_2024[["Date"]].copy()
        screened_display["Date"] = screened_display["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(screened_display, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Model Details:** XGBoost screened decomposition model trained on historical data with cross-validation and held-out 2024 test year evaluation.")

except FileNotFoundError as e:
    st.error(f"❌ Data files not found. Please ensure the model output files exist in `output/13_screened_model/`")
    st.stop()

"""
RRWWTF Master Dataset Visualizations — Step 2 (Visualization Only)

Reads the already-validated master dataset
(output/richmond_master_flow_rainfall_dataset.csv) and produces engineering
review figures at DAILY resolution. This script does not go back to the raw
monthly workbooks, does not recreate or modify the master dataset, and does
not perform statistical analysis, correlation/lag analysis, outlier removal,
or modeling. Two visualizations (histogram mean/median, monthly/annual
averages) compute simple values strictly for use as plot reference
lines/points, per the visualization spec.

No missing calendar date is filled with fabricated data. For line plots, the
underlying series is reindexed onto its full daily date range so that
matplotlib breaks the line (via NaN) across days/months with no plant
record, instead of drawing a straight line across a gap that would imply
continuous observations.

Usage:
    python scripts/master_data_visualizations.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_flow_rainfall_dataset.csv"

VIZ_DIR = OUTPUT_DIR / "01_visualizations"
YEARLY_DIR = VIZ_DIR / "yearly"
PDF_PATH = VIZ_DIR / "richmond_yearly_daily_flow_rainfall.pdf"

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"
MEAN_COLOR = "#c0392b"
MEDIAN_COLOR = "#27ae60"

DPI = 300


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_master_dataset() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def full_range_series(df: pd.DataFrame, col: str, start, end) -> pd.Series:
    """
    Reindex a column onto its complete daily calendar range so that plotting
    breaks the line across days with no plant record (no row in the master
    dataset), rather than connecting straight across the gap. This does not
    add data to the dataset itself -- it is used only to control how the
    line is drawn.
    """
    idx = pd.date_range(start, end, freq="D")
    s = df.drop_duplicates(subset="Date").set_index("Date")[col]
    return s.reindex(idx)


def save_fig(fig, out_path: Path, dpi: int = DPI):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization 1: Complete historical daily flow
# ---------------------------------------------------------------------------

def viz1_historical_daily_flow(df: pd.DataFrame) -> Path:
    start, end = df["Date"].min(), df["Date"].max()
    flow = full_range_series(df, "Total_Treated_MGD", start, end)

    fig, ax = plt.subplots(figsize=(28, 8))
    ax.plot(flow.index, flow.values, color=FLOW_COLOR, linewidth=0.8,
            marker="o", markersize=2.2, markerfacecolor=FLOW_COLOR,
            markeredgewidth=0, label="Daily Total Treated Flow")
    ax.set_title("Historical Daily Wastewater Flow – Richmond RRWWTF", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Treated Flow (MGD)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()

    out = VIZ_DIR / "01_historical_daily_flow.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Visualization 2: Complete historical flow + plant rainfall (dual axis)
# ---------------------------------------------------------------------------

def viz2_historical_flow_rainfall(df: pd.DataFrame) -> Path:
    start, end = df["Date"].min(), df["Date"].max()
    flow = full_range_series(df, "Total_Treated_MGD", start, end)

    fig, ax1 = plt.subplots(figsize=(28, 9))
    line, = ax1.plot(flow.index, flow.values, color=FLOW_COLOR, linewidth=0.8,
                      marker="o", markersize=2.2, markerfacecolor=FLOW_COLOR,
                      markeredgewidth=0, label="Daily Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    rain_valid = df.dropna(subset=["Rainfall_in"])
    bars = ax2.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=1.0,
                    color=RAIN_COLOR, alpha=0.6, label="Plant Rainfall")
    ax2.set_ylabel("Plant Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    ax1.set_title("Daily Wastewater Flow and Plant Rainfall – Richmond RRWWTF", fontsize=16)
    ax1.legend([line, bars], ["Daily Flow", "Plant Rainfall"], loc="upper right")
    fig.autofmt_xdate()

    out = VIZ_DIR / "02_historical_flow_rainfall.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Visualization 3: Year-by-year daily flow + rainfall (PNG per year + PDF)
# ---------------------------------------------------------------------------

def render_year_axes(ax1, year: int, df_year: pd.DataFrame):
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    flow = full_range_series(df_year, "Total_Treated_MGD", start, end)

    line, = ax1.plot(flow.index, flow.values, color=FLOW_COLOR, linewidth=1.0,
                      marker="o", markersize=3.5, markerfacecolor=FLOW_COLOR,
                      markeredgewidth=0, label="Daily Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    rain_valid = df_year.dropna(subset=["Rainfall_in"])
    bars = ax2.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=0.8,
                    color=RAIN_COLOR, alpha=0.6, label="Plant Rainfall")
    ax2.set_ylabel("Plant Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    ax1.set_title(f"Daily Wastewater Flow and Plant Rainfall – {year}", fontsize=16)
    ax1.legend([line, bars], ["Daily Flow", "Plant Rainfall"], loc="upper right")

    ticks = pd.date_range(start, end, freq="7D")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([t.strftime("%m/%d/%Y") for t in ticks], rotation=45, ha="right")
    ax1.set_xlim(start, end)


def viz3_yearly_flow_rainfall(df: pd.DataFrame, years: list) -> list:
    YEARLY_DIR.mkdir(parents=True, exist_ok=True)
    png_paths = []

    with PdfPages(PDF_PATH) as pdf:
        for year in years:
            df_year = df[df["Date"].dt.year == year]

            # Large, dense PNG for individual review
            fig, ax1 = plt.subplots(figsize=(26, 10))
            render_year_axes(ax1, year, df_year)
            fig.tight_layout()
            out = YEARLY_DIR / f"flow_rainfall_{year}.png"
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
            png_paths.append(out)

            # 11x17 landscape page for the combined PDF
            fig2, ax1b = plt.subplots(figsize=(17, 11))
            render_year_axes(ax1b, year, df_year)
            fig2.tight_layout()
            pdf.savefig(fig2)
            plt.close(fig2)

    return png_paths


# ---------------------------------------------------------------------------
# Visualization 4: Flow distribution histogram (mean/median reference only)
# ---------------------------------------------------------------------------

def viz4_flow_distribution(df: pd.DataFrame) -> Path:
    valid = df["Total_Treated_MGD"].dropna()
    mean_v = valid.mean()
    median_v = valid.median()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(valid, bins=50, color=FLOW_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(mean_v, color=MEAN_COLOR, linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_v:.3f} MGD")
    ax.axvline(median_v, color=MEDIAN_COLOR, linestyle="--", linewidth=1.5,
               label=f"Median = {median_v:.3f} MGD")
    ax.set_title("Distribution of Daily Wastewater Flow – All Years", fontsize=15)
    ax.set_xlabel("Total Treated Flow (MGD)")
    ax.set_ylabel("Number of Days")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend()

    out = VIZ_DIR / "03_flow_distribution.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Visualization 5: Monthly flow boxplot (all years combined, outliers shown)
# ---------------------------------------------------------------------------

def viz5_monthly_boxplot(df: pd.DataFrame) -> Path:
    d = df.dropna(subset=["Total_Treated_MGD"])
    data = [d.loc[d["Month"] == m, "Total_Treated_MGD"].values for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(data, tick_labels=MONTH_ABBR, showfliers=True)
    ax.set_title("Monthly Distribution of Wastewater Flow – All Years", fontsize=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Treated Flow (MGD)")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    out = VIZ_DIR / "04_monthly_flow_boxplot.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Visualization 6: Monthly average flow (all years combined)
# ---------------------------------------------------------------------------

def viz6_monthly_average_flow(df: pd.DataFrame) -> Path:
    d = df.dropna(subset=["Total_Treated_MGD"])
    monthly_avg = d.groupby("Month")["Total_Treated_MGD"].mean().reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(1, 13), monthly_avg.values, color=FLOW_COLOR, linewidth=1.5,
            marker="o", markersize=6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_title("Average Wastewater Flow by Month – All Years", fontsize=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Total Treated Flow (MGD)")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out = VIZ_DIR / "05_monthly_average_flow.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Visualization 7: Annual average flow
# ---------------------------------------------------------------------------

def viz7_annual_average_flow(df: pd.DataFrame, years: list) -> Path:
    d = df.dropna(subset=["Total_Treated_MGD"])
    annual_avg = d.groupby("Year")["Total_Treated_MGD"].mean().reindex(years)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(years, annual_avg.values, color=FLOW_COLOR, linewidth=1.5,
            marker="o", markersize=7)
    ax.set_xticks(years)
    ax.set_title("Annual Average Wastewater Flow – Richmond RRWWTF", fontsize=15)
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Total Treated Flow (MGD)")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out = VIZ_DIR / "06_annual_average_flow.png"
    save_fig(fig, out)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    YEARLY_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master_dataset()
    years = sorted(int(y) for y in df["Year"].unique())

    generated = []
    generated.append(viz1_historical_daily_flow(df))
    generated.append(viz2_historical_flow_rainfall(df))
    generated.extend(viz3_yearly_flow_rainfall(df, years))
    generated.append(viz4_flow_distribution(df))
    generated.append(viz5_monthly_boxplot(df))
    generated.append(viz6_monthly_average_flow(df))
    generated.append(viz7_annual_average_flow(df, years))

    n_flow = int(df["Total_Treated_MGD"].notna().sum())
    n_rain = int(df["Rainfall_in"].notna().sum())

    print("DATA VISUALIZATION COMPLETE")
    print()
    print("Master dataset used:")
    print(MASTER_CSV.relative_to(BASE_DIR).as_posix())
    print()
    print("Date range:")
    print(f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    print()
    print("Years visualized:")
    print(", ".join(str(y) for y in years))
    print()
    print("Number of daily flow observations plotted:")
    print(n_flow)
    print()
    print("Number of rainfall observations plotted:")
    print(n_rain)
    print()
    print("Generated files:")
    for p in generated:
        print(p.relative_to(BASE_DIR).as_posix())
    print(PDF_PATH.relative_to(BASE_DIR).as_posix())


if __name__ == "__main__":
    main()

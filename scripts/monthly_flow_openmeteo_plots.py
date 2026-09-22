"""
RRWWTF — Monthly Daily-Resolution Flow vs. Open-Meteo Rainfall Graphs

Recreates the monthly daily-resolution flow/rainfall graph structure, but
with rainfall sourced ONLY from the independent Open-Meteo download
(scripts/openmeteo_richmond_rainfall.py) instead of the `Rain` column that
lives in the original RRWWTF workbooks. The original rainfall column is
never read or referenced anywhere in this script.

Wastewater flow still comes from the existing Step 1 dataset
(output/richmond_total_treated_mgd.csv), unmodified.

Steps:
  1. Reuse the existing Open-Meteo rainfall CSV if it already covers the
     flow dataset's full date range; otherwise fetch it fresh using the
     same function from openmeteo_richmond_rainfall.py.
  2. Merge flow (reference) and Open-Meteo rainfall onto one continuous
     daily calendar index spanning the flow dataset's earliest to latest
     date, producing Date / Flow_MGD / OpenMeteo_Rainfall_in / Year / Month.
  3. Generate one daily-resolution dual-axis graph per calendar month
     (every month from the first to the last month in the dataset, even
     months with no flow coverage at all, e.g. parts of 2023).
  4. Save each monthly graph as its own PNG and combine all of them,
     chronologically, into one landscape multi-page PDF.

No correlation, lag analysis, or modeling is performed here.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

FLOW_CSV = OUTPUT_DIR / "richmond_total_treated_mgd.csv"          # flow only, no rain column
OPENMETEO_CSV = OUTPUT_DIR / "richmond_openmeteo_daily_rainfall.csv"
MERGED_CSV = OUTPUT_DIR / "richmond_flow_openmeteo_daily.csv"

MONTHLY_DIR = OUTPUT_DIR / "richmond_openmeteo_monthly_plots"
PDF_PATH = OUTPUT_DIR / "Richmond_Monthly_Flow_OpenMeteo_Rainfall.pdf"

LATITUDE = 29.5818
LONGITUDE = -95.7608
TIMEZONE = "America/Chicago"

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}


# ---------------------------------------------------------------------------
# Step 1: get Open-Meteo rainfall (reuse cached CSV if it already covers
# the required range, otherwise fetch fresh via the sibling script)
# ---------------------------------------------------------------------------

def get_openmeteo_rainfall(flow_start: pd.Timestamp, flow_end: pd.Timestamp) -> pd.DataFrame:
    if OPENMETEO_CSV.exists():
        cached = pd.read_csv(OPENMETEO_CSV, parse_dates=["Date"])
        if cached["Date"].min() <= flow_start and cached["Date"].max() >= flow_end:
            print(f"Reusing cached Open-Meteo rainfall: {OPENMETEO_CSV} "
                  f"({cached['Date'].min().date()} to {cached['Date'].max().date()})")
            return cached[(cached["Date"] >= flow_start) & (cached["Date"] <= flow_end)].reset_index(drop=True)
        print("Cached Open-Meteo rainfall does not cover the full flow date range - refetching.")
    else:
        print("No cached Open-Meteo rainfall found - fetching from the Archive API.")

    sys.path.insert(0, str(SCRIPTS_DIR))
    from openmeteo_richmond_rainfall import fetch_openmeteo_daily_rainfall  # noqa: E402

    rain_df = fetch_openmeteo_daily_rainfall(
        LATITUDE, LONGITUDE, flow_start.strftime("%Y-%m-%d"), flow_end.strftime("%Y-%m-%d"), TIMEZONE
    )
    rain_df.to_csv(OPENMETEO_CSV, index=False)
    return rain_df


# ---------------------------------------------------------------------------
# Step 2: build the merged daily plotting dataframe
# ---------------------------------------------------------------------------

def build_merged_daily(flow_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(flow_df["Date"].min(), flow_df["Date"].max(), freq="D")
    merged = pd.DataFrame({"Date": full_index})
    merged = merged.merge(flow_df[["Date", "Total_Treated_MGD"]], on="Date", how="left")
    merged = merged.merge(rain_df[["Date", "OpenMeteo_Rainfall_in"]], on="Date", how="left")
    merged = merged.rename(columns={"Total_Treated_MGD": "Flow_MGD"})
    merged["Year"] = merged["Date"].dt.year
    merged["Month"] = merged["Date"].dt.month
    return merged[["Date", "Flow_MGD", "OpenMeteo_Rainfall_in", "Year", "Month"]]


# ---------------------------------------------------------------------------
# Step 3: one dual-axis daily-resolution plot per month
# ---------------------------------------------------------------------------

def plot_month(month_df: pd.DataFrame, year: int, month: int, flow_range: tuple) -> plt.Figure:
    month_name = MONTH_NAMES[month]
    fig, ax1 = plt.subplots(figsize=(16, 8))

    ax1.plot(month_df["Date"], month_df["Flow_MGD"], color=FLOW_COLOR, linewidth=1.4,
              marker="o", markersize=4, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
              label="Wastewater Flow", zorder=3)
    ax1.set_ylabel("Wastewater Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.set_xlabel("Date")
    ax1.grid(True, linewidth=0.3, alpha=0.35, zorder=0)

    # When a month has no flow observations at all, no line is drawn (the
    # gap is genuine, nothing is fabricated) - but leave the axis anchored
    # to the dataset's overall flow range instead of matplotlib's default
    # auto-scale-to-nothing behavior, which would otherwise show a tiny,
    # misleading near-zero range for an entirely-missing month.
    if month_df["Flow_MGD"].notna().sum() == 0:
        ax1.set_ylim(*flow_range)

    ax2 = ax1.twinx()
    rain_valid = month_df.dropna(subset=["OpenMeteo_Rainfall_in"])
    bars = ax2.bar(rain_valid["Date"], rain_valid["OpenMeteo_Rainfall_in"], width=0.7,
                    color=RAIN_COLOR, alpha=0.65, label="Open-Meteo Rainfall", zorder=2)
    ax2.set_ylabel("Daily Rainfall (in)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    # Conventional hyetograph orientation: rainfall bars hang down from the
    # top of the axis so the flow line remains the visually dominant series.
    rain_max = rain_valid["OpenMeteo_Rainfall_in"].max()
    rain_max = 0.3 if pd.isna(rain_max) or rain_max <= 0 else rain_max
    ax2.set_ylim(rain_max * 3.2, 0)

    ax1.set_xlim(month_df["Date"].min(), month_df["Date"].max())
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(f"Richmond Wastewater Flow and Open-Meteo Rainfall — {month_name} {year}",
                 fontsize=14, y=0.975)
    ax1.set_title(f"Rainfall Source: Open-Meteo | Richmond, TX ({LATITUDE}, {LONGITUDE})",
                  fontsize=9, color="gray", pad=10)

    lines, line_labels = ax1.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bar_handles, line_labels + bar_labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    flow_df = pd.read_csv(FLOW_CSV, parse_dates=["Date"])  # flow only - no rain column present
    flow_start, flow_end = flow_df["Date"].min(), flow_df["Date"].max()

    rain_df = get_openmeteo_rainfall(flow_start, flow_end)

    merged = build_merged_daily(flow_df, rain_df)
    merged.to_csv(MERGED_CSV, index=False)

    year_months = sorted(merged[["Year", "Month"]].drop_duplicates().itertuples(index=False, name=None))

    valid_flow = merged["Flow_MGD"].dropna()
    pad = (valid_flow.max() - valid_flow.min()) * 0.08
    flow_range = (valid_flow.min() - pad, valid_flow.max() + pad)

    generated_files = []
    with PdfPages(PDF_PATH) as pdf:
        for year, month in year_months:
            month_df = merged[(merged["Year"] == year) & (merged["Month"] == month)]
            fig = plot_month(month_df, year, month, flow_range)

            fname = f"{year}_{month:02d}_{MONTH_NAMES[month]}_flow_openmeteo_rainfall.png"
            out_path = MONTHLY_DIR / fname
            fig.savefig(out_path, dpi=200)
            pdf.savefig(fig, orientation="landscape")
            plt.close(fig)
            generated_files.append(out_path)

    n_months_no_flow = sum(
        1 for year, month in year_months
        if merged.loc[(merged["Year"] == year) & (merged["Month"] == month), "Flow_MGD"].notna().sum() == 0
    )

    print()
    print("RICHMOND MONTHLY FLOW vs. OPEN-METEO RAINFALL - GRAPHS COMPLETE")
    print(f"Flow dataset range:               {flow_start.date()} to {flow_end.date()}")
    print(f"Rainfall source:                   Open-Meteo Archive API ({LATITUDE}, {LONGITUDE})")
    print(f"Monthly graphs generated:          {len(generated_files)}")
    print(f"Months with zero flow coverage:    {n_months_no_flow} (rainfall still plotted, flow line absent)")
    print()
    print("Output files:")
    print(f"  {MERGED_CSV}")
    print(f"  {MONTHLY_DIR}\\ ({len(generated_files)} PNGs)")
    print(f"  {PDF_PATH}")


if __name__ == "__main__":
    main()

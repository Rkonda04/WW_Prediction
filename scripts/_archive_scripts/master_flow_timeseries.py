"""
RRWWTF Master Daily Flow + Rainfall — Single Continuous Time-Series Graph

Reads the already-validated master dataset
(output/richmond_master_flow_rainfall_dataset.csv) and produces ONE
continuous dual-axis figure covering the complete historical period: daily
Total_Treated_MGD as a line (left axis) and plant-recorded Rainfall_in as
bars (right axis). No aggregation, no smoothing, no outlier removal, no
statistics, no correlation/lag analysis.

The plotted flow line is reindexed onto the full daily calendar range (min
to max date) purely so matplotlib breaks the line at missing calendar dates
instead of connecting straight across a gap. This does not modify or add
data to the master dataset -- it only controls how the existing points are
drawn. Rainfall bars are drawn only for days with an actual recorded value
(missing rainfall is never treated as zero).

Usage:
    python scripts/master_flow_timeseries.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_flow_rainfall_dataset.csv"

VIZ_DIR = OUTPUT_DIR / "01_visualizations"
OUT_PATH = VIZ_DIR / "master_daily_flow_timeseries.png"

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    earliest = df["Date"].min()
    latest = df["Date"].max()

    valid = df.dropna(subset=["Total_Treated_MGD"])
    n_obs = int(len(valid))
    min_flow = valid["Total_Treated_MGD"].min()
    max_flow = valid["Total_Treated_MGD"].max()
    mean_flow = valid["Total_Treated_MGD"].mean()
    median_flow = valid["Total_Treated_MGD"].median()

    rain_valid = df.dropna(subset=["Rainfall_in"])
    n_rain_obs = int(len(rain_valid))
    min_rain = rain_valid["Rainfall_in"].min()
    max_rain = rain_valid["Rainfall_in"].max()
    mean_rain = rain_valid["Rainfall_in"].mean()
    median_rain = rain_valid["Rainfall_in"].median()

    full_range = pd.date_range(earliest, latest, freq="D")
    present_dates = pd.DatetimeIndex(df["Date"].unique())
    n_missing_calendar_dates = int(len(full_range.difference(present_dates)))

    # Reindex onto the full calendar range so the line breaks across gaps
    # instead of connecting straight across missing calendar dates.
    flow_series = df.drop_duplicates(subset="Date").set_index("Date")["Total_Treated_MGD"]
    flow_series = flow_series.reindex(full_range)

    fig, ax1 = plt.subplots(figsize=(20, 8))
    line, = ax1.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=0.7,
                      marker="o", markersize=2.5, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
                      label="Daily Total Treated Flow")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = ax1.twinx()
    bars = ax2.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=2.5,
                    color=RAIN_COLOR, alpha=0.75, label="Plant Rainfall")
    ax2.set_ylabel("Plant Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    ax1.set_title("Historical Daily Wastewater Flow and Plant Rainfall – Richmond RRWWTF", fontsize=16)
    ax1.legend([line, bars], ["Daily Total Treated Flow", "Plant Rainfall"], loc="upper right")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)

    # Stats panel on the side (visual reference only -- not a printed
    # statistical-analysis table; values also echoed to console below).
    stats_text = (
        "FLOW (MGD)\n"
        f"  n:      {n_obs:,}\n"
        f"  Mean:   {mean_flow:.3f}\n"
        f"  Median: {median_flow:.3f}\n"
        f"  Min:    {min_flow:.3f}\n"
        f"  Max:    {max_flow:.3f}\n"
        "\n"
        "RAINFALL (in)\n"
        f"  n:      {n_rain_obs:,}\n"
        f"  Mean:   {mean_rain:.3f}\n"
        f"  Median: {median_rain:.3f}\n"
        f"  Min:    {min_rain:.3f}\n"
        f"  Max:    {max_rain:.3f}"
    )
    fig.subplots_adjust(right=0.86)
    fig.text(0.875, 0.5, stats_text, transform=fig.transFigure,
              ha="left", va="center", fontsize=11, family="monospace",
              bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                        edgecolor="#888888", linewidth=0.8))

    fig.savefig(OUT_PATH, dpi=300)

    print("MASTER FLOW + RAINFALL TIME SERIES")
    print()
    print("Date range:")
    print(f"{earliest.date()} to {latest.date()}")
    print()
    print("Number of flow observations plotted:")
    print(n_obs)
    print()
    print("Minimum plotted flow:")
    print(f"{min_flow} MGD")
    print()
    print("Maximum plotted flow:")
    print(f"{max_flow} MGD")
    print()
    print("Mean plotted flow (reference only):")
    print(f"{mean_flow:.3f} MGD")
    print()
    print("Median plotted flow (reference only):")
    print(f"{median_flow:.3f} MGD")
    print()
    print("Number of rainfall observations plotted:")
    print(n_rain_obs)
    print()
    print("Minimum plotted rainfall:")
    print(f"{min_rain} in")
    print()
    print("Maximum plotted rainfall:")
    print(f"{max_rain} in")
    print()
    print("Mean plotted rainfall (reference only):")
    print(f"{mean_rain:.3f} in")
    print()
    print("Median plotted rainfall (reference only):")
    print(f"{median_rain:.3f} in")
    print()
    print("Number of missing calendar dates within the date range:")
    print(n_missing_calendar_dates)
    print()
    print(f"Saved: {OUT_PATH.relative_to(BASE_DIR).as_posix()}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

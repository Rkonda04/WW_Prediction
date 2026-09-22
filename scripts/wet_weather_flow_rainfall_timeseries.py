"""
RRWWTF Wet-Weather (Rainfall Days) Daily Flow + Rainfall — Time-Series Graph

Reads the already-validated master dataset
(output/richmond_master_flow_rainfall_dataset.csv) and produces ONE
continuous dual-axis figure using ONLY dates with a confirmed plant-recorded
rainfall value greater than 0.00 in (wet weather / rainfall days). Also
saves the filtered wet-weather subset as its own dataset
(output/richmond_wet_weather_flow_rainfall_dataset.csv) -- the master
dataset itself is never modified.

Excluded from the plot (not from the master dataset, which is never
modified):
  - Days with Rainfall_in == 0 (dry weather days)
  - Days with Rainfall_in missing/NaN (rainfall unknown)

No aggregation, smoothing, outlier removal, or statistics is performed.

The flow line connects only the actual wet-weather observations, directly in
chronological order -- excluded days (dry, missing rainfall, or no plant
record at all) are simply not part of the series, so no break/placeholder
points are plotted for them. Rainfall bars are drawn only for the included
wet-weather dates.

Usage:
    python scripts/wet_weather_flow_rainfall_timeseries.py
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
OUT_PATH = VIZ_DIR / "wet_weather_flow_rainfall_timeseries.png"
DATASET_OUT_PATH = VIZ_DIR / "richmond_wet_weather_flow_rainfall_dataset.csv"

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"
MEAN_COLOR = "#c0392b"
MEDIAN_COLOR = "#27ae60"


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    earliest = df["Date"].min()
    latest = df["Date"].max()

    n_total = len(df)
    n_dry = int((df["Rainfall_in"] == 0).sum())
    n_missing_rain = int(df["Rainfall_in"].isna().sum())

    wet_df = df[df["Rainfall_in"] > 0].sort_values("Date").reset_index(drop=True)
    n_wet = len(wet_df)

    # Save the filtered wet-weather subset as its own dataset (does not
    # touch/modify the master dataset file).
    wet_df.to_csv(DATASET_OUT_PATH, index=False)

    valid_flow = wet_df.dropna(subset=["Total_Treated_MGD"])
    n_flow_obs = int(len(valid_flow))
    min_flow = valid_flow["Total_Treated_MGD"].min()
    max_flow = valid_flow["Total_Treated_MGD"].max()
    mean_flow = valid_flow["Total_Treated_MGD"].mean()
    median_flow = valid_flow["Total_Treated_MGD"].median()

    min_rain = wet_df["Rainfall_in"].min()
    max_rain = wet_df["Rainfall_in"].max()

    # Plot only the actual wet-weather observations, connected directly in
    # chronological order -- no placeholder points for excluded days.
    plot_df = valid_flow.sort_values("Date")

    fig, ax1 = plt.subplots(figsize=(18, 8))
    line, = ax1.plot(plot_df["Date"], plot_df["Total_Treated_MGD"], color=FLOW_COLOR, linewidth=0.7,
                      marker="o", markersize=3, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
                      label="Daily Total Treated Flow (Rainfall Days Only)")
    mean_line = ax1.axhline(mean_flow, color=MEAN_COLOR, linestyle="--", linewidth=1.3,
                             label=f"Mean = {mean_flow:.3f} MGD")
    median_line = ax1.axhline(median_flow, color=MEDIAN_COLOR, linestyle="--", linewidth=1.3,
                               label=f"Median = {median_flow:.3f} MGD")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    ax2 = ax1.twinx()
    bars = ax2.bar(wet_df["Date"], wet_df["Rainfall_in"], width=1.5,
                    color=RAIN_COLOR, alpha=0.7, label="Plant Rainfall")
    ax2.set_ylabel("Plant Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    ax1.set_title(
        "Wet-Weather Daily Wastewater Flow and Plant Rainfall – Richmond RRWWTF\n"
        "(Rainfall_in > 0 only; dry and unknown-rainfall days excluded)",
        fontsize=15)
    ax1.legend([line, bars, mean_line, median_line],
               ["Daily Total Treated Flow (Rainfall Days Only)", "Plant Rainfall",
                f"Mean = {mean_flow:.3f} MGD", f"Median = {median_flow:.3f} MGD"],
               loc="upper right")

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)

    print("WET-WEATHER (RAINFALL DAYS) FLOW + RAINFALL TIME SERIES")
    print()
    print("Master dataset date range:")
    print(f"{earliest.date()} to {latest.date()}")
    print()
    print("Total records in master dataset:")
    print(n_total)
    print()
    print("Dates excluded (rainfall = 0, dry weather):")
    print(n_dry)
    print()
    print("Dates excluded (rainfall missing/unknown):")
    print(n_missing_rain)
    print()
    print("Wet-weather dates plotted (Rainfall_in > 0):")
    print(n_wet)
    print()
    print("Number of wet-weather flow observations plotted:")
    print(n_flow_obs)
    print()
    print("Minimum plotted flow (wet weather):")
    print(f"{min_flow} MGD")
    print()
    print("Maximum plotted flow (wet weather):")
    print(f"{max_flow} MGD")
    print()
    print("Mean plotted flow (wet weather, reference line only):")
    print(f"{mean_flow:.3f} MGD")
    print()
    print("Median plotted flow (wet weather, reference line only):")
    print(f"{median_flow:.3f} MGD")
    print()
    print("Minimum plotted rainfall:")
    print(f"{min_rain} in")
    print()
    print("Maximum plotted rainfall:")
    print(f"{max_rain} in")
    print()
    print(f"Figure saved: {OUT_PATH.relative_to(BASE_DIR).as_posix()}")
    print(f"Dataset saved: {DATASET_OUT_PATH.relative_to(BASE_DIR).as_posix()}")

    plt.close(fig)


if __name__ == "__main__":
    main()

"""
RRWWTF Dry-Weather Daily Flow — Single Continuous Time-Series Graph

Reads the already-validated master dataset
(output/richmond_master_flow_rainfall_dataset.csv) and produces ONE
continuous daily Total_Treated_MGD time-series figure using ONLY dates with
a confirmed plant-recorded rainfall value of exactly 0.00 in (dry weather
days). Also saves the filtered dry-weather subset as its own dataset
(output/richmond_dry_weather_flow_dataset.csv) -- the master dataset itself
is never modified.

Excluded from the plot (not from the master dataset, which is never
modified):
  - Days with Rainfall_in > 0 (wet weather days)
  - Days with Rainfall_in missing/NaN (rainfall unknown -- cannot be
    confirmed dry, so not included as a dry-weather observation)

No aggregation, smoothing, outlier removal, or statistics is performed.

The line connects only the actual dry-weather observations, directly in
chronological order -- excluded days (wet, missing rainfall, or no plant
record at all) are simply not part of the series, so no break/placeholder
points are plotted for them.

Usage:
    python scripts/dry_weather_flow_timeseries.py
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
OUT_PATH = VIZ_DIR / "dry_weather_flow_timeseries.png"
DATASET_OUT_PATH = VIZ_DIR / "richmond_dry_weather_flow_dataset.csv"

FLOW_COLOR = "#2b6cb0"
MEAN_COLOR = "#c0392b"
MEDIAN_COLOR = "#27ae60"

#This ios the main function that runs the script/
def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    earliest = df["Date"].min()
    latest = df["Date"].max()

    n_total = len(df)
    n_wet = int((df["Rainfall_in"] > 0).sum())
    n_missing_rain = int(df["Rainfall_in"].isna().sum())

    dry_df = df[df["Rainfall_in"] == 0].sort_values("Date").reset_index(drop=True)
    n_dry = len(dry_df)

    # Save the filtered dry-weather subset as its own dataset (does not
    # touch/modify the master dataset file).
    dry_df.to_csv(DATASET_OUT_PATH, index=False)

    valid_flow = dry_df.dropna(subset=["Total_Treated_MGD"])
    n_flow_obs = int(len(valid_flow))
    min_flow = valid_flow["Total_Treated_MGD"].min()
    max_flow = valid_flow["Total_Treated_MGD"].max()
    mean_flow = valid_flow["Total_Treated_MGD"].mean()
    median_flow = valid_flow["Total_Treated_MGD"].median()

    # Plot only the actual dry-weather observations, connected directly in
    # chronological order -- no placeholder points for excluded days.
    plot_df = valid_flow.sort_values("Date")

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(plot_df["Date"], plot_df["Total_Treated_MGD"], color=FLOW_COLOR, linewidth=0.7,
            marker="o", markersize=2.5, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
            label="Daily Total Treated Flow (Dry Weather Only)")
    ax.axhline(mean_flow, color=MEAN_COLOR, linestyle="--", linewidth=1.3,
               label=f"Mean = {mean_flow:.3f} MGD")
    ax.axhline(median_flow, color=MEDIAN_COLOR, linestyle="--", linewidth=1.3,
               label=f"Median = {median_flow:.3f} MGD")

    ax.set_title("Dry-Weather Daily Wastewater Flow – Richmond RRWWTF\n(Rainfall_in = 0, wet and unknown-rainfall days excluded)",
                  fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Treated Flow (MGD)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)

    print("DRY-WEATHER FLOW TIME SERIES")
    print()
    print("Master dataset date range:")
    print(f"{earliest.date()} to {latest.date()}")
    print()
    print("Total records in master dataset:")
    print(n_total)
    print()
    print("Dates excluded (rainfall > 0, wet weather):")
    print(n_wet)
    print()
    print("Dates excluded (rainfall missing/unknown):")
    print(n_missing_rain)
    print()
    print("Dry-weather dates plotted (Rainfall_in = 0):")
    print(n_dry)
    print()
    print("Number of dry-weather flow observations plotted:")
    print(n_flow_obs)
    print()
    print("Minimum plotted flow (dry weather):")
    print(f"{min_flow} MGD")
    print()
    print("Maximum plotted flow (dry weather):")
    print(f"{max_flow} MGD")
    print()
    print("Mean plotted flow (dry weather, reference line only):")
    print(f"{mean_flow:.3f} MGD")
    print()
    print("Median plotted flow (dry weather, reference line only):")
    print(f"{median_flow:.3f} MGD")
    print()
    print(f"Figure saved: {OUT_PATH.relative_to(BASE_DIR).as_posix()}")
    print(f"Dataset saved: {DATASET_OUT_PATH.relative_to(BASE_DIR).as_posix()}")

    plt.close(fig)


if __name__ == "__main__":
    main()

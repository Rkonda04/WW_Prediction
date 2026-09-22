"""
RRWWTF Rainfall Time Series, Characterization, and Flow Alignment (Step 1 of I&I/RDII)

Visualization and QA only -- NO excess-flow or I&I/RDII calculation happens
in this script. It reads the already-validated master dataset (with the
previously computed 5-day antecedent dry-weather baseline already assigned)
and produces:

    1. A full-period daily rainfall time series with the top ~10 largest
       daily totals highlighted.
    2. Rainfall characterization: monthly totals, calendar-month seasonal
       average, and a wet-day-only rainfall histogram with dry/wet day counts.
    3. The key rainfall-vs-flow alignment figure: daily flow + the dry-weather
       baseline on the left axis, plant rainfall as inverted bars (hyetograph
       style) on the right axis -- one full-period version plus zoomed panels
       around the largest, well-separated rainfall events.
    4. A small yearly rainfall summary CSV.

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_dry_weather_baseline.csv
        Date, Total_Treated_MGD, Rainfall_in, Year, Month, Source_File,
        Source_Year, Dry_5Day, Expected_Baseline_MGD, Baseline_Source,
        Baseline_Dry_Day_Count

Rainfall is the plant-recorded Rainfall_in column only -- no OpenMeteo, NOAA,
or other external weather data is used. The dry-weather baseline methodology
is reused exactly as previously computed, not recalculated here.

Usage:
    python scripts/rainfall_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration -- paths and styling reused from the existing numbered pipeline
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_DIR = OUTPUT_DIR / "00_master_dataset"
MASTER_BASELINE_CSV = MASTER_DIR / "richmond_master_with_dry_weather_baseline.csv"

ANALYSIS_DIR = OUTPUT_DIR / "04_rainfall_analysis"
YEARLY_SUMMARY_CSV = ANALYSIS_DIR / "yearly_rainfall_summary.csv"

FIG1_PATH = ANALYSIS_DIR / "01_rainfall_daily_timeseries.png"
FIG2_PATH = ANALYSIS_DIR / "02_monthly_and_seasonal_rainfall.png"
FIG3_PATH = ANALYSIS_DIR / "03_wet_day_rainfall_histogram.png"
FIG4_PATH = ANALYSIS_DIR / "04_flow_rainfall_hyetograph_full_period.png"
FIG5_PATH = ANALYSIS_DIR / "05_flow_rainfall_hyetograph_zoomed_events.png"
FIG6_PATH = ANALYSIS_DIR / "06_flow_vs_rainfall_wet_days_only.png"

TOP_N_RAIN_DAYS = 10
N_ZOOM_EVENTS = 3
ZOOM_WINDOW_DAYS = 10          # days shown on each side of a zoomed event's peak
ZOOM_MIN_SEPARATION_DAYS = 15  # minimum gap between selected zoom events

RAINFALL_QA_THRESHOLD_IN = 5.0  # daily rainfall above this is flagged for manual QA (not removed)

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- Colors reused from the existing DWF/flow figures for visual consistency ---
FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"
BASELINE_COLOR = "#c0392b"
HIGHLIGHT_COLOR = "#d97706"

DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def style_ts_axes(ax, fig, interval_months=3):
    ax.set_xlabel("Date")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)


def collapse_to_ranges(dates) -> list:
    dates = sorted(dates)
    if not dates:
        return []
    ranges = []
    start = prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days == 1:
            prev = d
            continue
        ranges.append((start, prev))
        start = prev = d
    ranges.append((start, prev))
    return ranges


# ---------------------------------------------------------------------------
# Load validated inputs
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(MASTER_BASELINE_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Figure 1: full-period daily rainfall time series, top-N highlighted
# ---------------------------------------------------------------------------

def plot_rainfall_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    rain_valid = df.dropna(subset=["Rainfall_in"]).sort_values("Date")
    top_n = rain_valid.nlargest(TOP_N_RAIN_DAYS, "Rainfall_in").sort_values("Date")
    top_ids = set(top_n.index)

    colors = [HIGHLIGHT_COLOR if idx in top_ids else RAIN_COLOR for idx in rain_valid.index]

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=1.2, color=colors, alpha=0.85)

    for _, row in top_n.iterrows():
        ax.annotate(f"{row['Rainfall_in']:.2f} in\n{row['Date']:%m/%d/%Y}",
                    xy=(row["Date"], row["Rainfall_in"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#7c2d12")

    ax.set_title("Daily Plant Rainfall – Richmond RRWWTF (Top 10 Days Highlighted)")
    ax.set_ylabel("Daily Rainfall (in)")
    style_ts_axes(ax, fig)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=RAIN_COLOR, label="Daily Rainfall"),
                       Patch(color=HIGHLIGHT_COLOR, label=f"Top {TOP_N_RAIN_DAYS} Rainfall Days")]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG1_PATH, dpi=DPI)
    plt.close(fig)
    return top_n


# ---------------------------------------------------------------------------
# Figure 2: monthly totals + calendar-month seasonal average
# ---------------------------------------------------------------------------

def plot_monthly_and_seasonal_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    rain_valid = df.dropna(subset=["Rainfall_in"])

    monthly = rain_valid.groupby(["Year", "Month"])["Rainfall_in"].sum().reset_index()
    monthly["YM_Date"] = pd.to_datetime(dict(year=monthly["Year"], month=monthly["Month"], day=1))
    monthly = monthly.sort_values("YM_Date")

    seasonal = monthly.groupby("Month")["Rainfall_in"].mean().reindex(range(1, 13))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    ax1.bar(monthly["YM_Date"], monthly["Rainfall_in"], width=20, color=RAIN_COLOR)
    ax1.set_title("Monthly Total Rainfall – Richmond RRWWTF")
    ax1.set_ylabel("Total Rainfall (in)")
    style_ts_axes(ax1, fig)

    ax2.bar(range(1, 13), seasonal.values, color=RAIN_COLOR)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_ABBR)
    ax2.set_title("Average Monthly Rainfall by Calendar Month (All Years) – Richmond RRWWTF")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Average Monthly Total Rainfall (in)")
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG2_PATH, dpi=DPI)
    plt.close(fig)
    return monthly


# ---------------------------------------------------------------------------
# Figure 3: wet-day-only rainfall histogram, dry/wet counts noted
# ---------------------------------------------------------------------------

def plot_wet_day_histogram(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["Rainfall_in"])
    dry_days = int((valid["Rainfall_in"] == 0).sum())
    wet_days = int((valid["Rainfall_in"] > 0).sum())
    missing_days = int(df["Rainfall_in"].isna().sum())
    wet_vals = valid.loc[valid["Rainfall_in"] > 0, "Rainfall_in"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(wet_vals, bins=40, color=RAIN_COLOR, edgecolor="white", alpha=0.9)
    ax.set_title("Distribution of Daily Rainfall on Wet Days – Richmond RRWWTF")
    ax.set_xlabel("Daily Rainfall on Wet Days (in)")
    ax.set_ylabel("Number of Days")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    summary_text = (f"Dry days (rain = 0):   {dry_days:,}\n"
                     f"Wet days (rain > 0):   {wet_days:,}\n"
                     f"Missing rainfall days: {missing_days:,}\n"
                     f"Mean wet-day rainfall: {wet_vals.mean():.3f} in\n"
                     f"Median wet-day rainfall: {wet_vals.median():.3f} in")
    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888"))

    fig.tight_layout()
    fig.savefig(FIG3_PATH, dpi=DPI)
    plt.close(fig)

    return {"dry_days": dry_days, "wet_days": wet_days, "missing_days": missing_days}


# ---------------------------------------------------------------------------
# Figure 4: full-period flow + baseline + inverted rainfall hyetograph
# ---------------------------------------------------------------------------

def draw_hyetograph_panel(ax1, df: pd.DataFrame, start, end, full_range_for_break=None):
    """Draws flow (line, breaks across gaps) + baseline (line) on ax1, and
    inverted rainfall bars (hyetograph) on a twin ax2. Returns (ax2, handles, labels)."""
    window = df[(df["Date"] >= start) & (df["Date"] <= end)]

    if full_range_for_break is None:
        full_range_for_break = pd.date_range(start, end, freq="D")
    flow_series = window.drop_duplicates(subset="Date").set_index("Date")["Total_Treated_MGD"].reindex(full_range_for_break)
    baseline_series = window.drop_duplicates(subset="Date").set_index("Date")["Expected_Baseline_MGD"].reindex(full_range_for_break)

    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    rain_window = window.dropna(subset=["Rainfall_in"])
    bars = ax2.bar(rain_window["Date"], rain_window["Rainfall_in"], width=0.9,
                    color=RAIN_COLOR, alpha=0.65, label="Plant Rainfall")
    rain_max = window["Rainfall_in"].max()
    rain_ceiling = max(rain_max * 3.0, 0.5) if pd.notna(rain_max) else 1.0
    ax2.set_ylim(rain_ceiling, 0)  # inverted: bars hang from the top
    ax2.set_ylabel("Plant Rainfall (in)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    flow_line, = ax1.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=1.1,
                           marker="o", markersize=2.5, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
                           label="Daily Total Treated Flow")
    baseline_line, = ax1.plot(baseline_series.index, baseline_series.values, color=BASELINE_COLOR,
                               linewidth=1.8, linestyle="--", label="Dry-Weather Baseline")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    return ax2, [flow_line, baseline_line, bars], ["Daily Total Treated Flow", "Dry-Weather Baseline", "Plant Rainfall"]


def plot_full_period_hyetograph(df: pd.DataFrame):
    earliest, latest = df["Date"].min(), df["Date"].max()
    full_range = pd.date_range(earliest, latest, freq="D")

    fig, ax1 = plt.subplots(figsize=(22, 9))
    ax2, handles, labels = draw_hyetograph_panel(ax1, df, earliest, latest, full_range)
    ax1.set_title("Historical Daily Wastewater Flow, Dry-Weather Baseline, and Plant Rainfall – Richmond RRWWTF")
    ax1.legend(handles, labels, loc="upper right")
    style_ts_axes(ax1, fig)
    fig.tight_layout()
    fig.savefig(FIG4_PATH, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: zoomed panels around the largest, well-separated rainfall events
# ---------------------------------------------------------------------------

def select_zoom_events(df: pd.DataFrame) -> pd.DataFrame:
    rain_sorted = df.dropna(subset=["Rainfall_in"]).sort_values("Rainfall_in", ascending=False)
    selected = []
    for _, row in rain_sorted.iterrows():
        if all(abs((row["Date"] - s["Date"]).days) >= ZOOM_MIN_SEPARATION_DAYS for s in selected):
            selected.append(row)
        if len(selected) == N_ZOOM_EVENTS:
            break
    return pd.DataFrame(selected).sort_values("Date")


def plot_zoomed_hyetographs(df: pd.DataFrame, zoom_events: pd.DataFrame):
    n = len(zoom_events)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(18, 6 * n))
    if n == 1:
        axes = [axes]

    for ax1, (_, peak) in zip(axes, zoom_events.iterrows()):
        start = peak["Date"] - pd.Timedelta(days=ZOOM_WINDOW_DAYS)
        end = peak["Date"] + pd.Timedelta(days=ZOOM_WINDOW_DAYS)
        ax2, handles, labels = draw_hyetograph_panel(ax1, df, start, end)
        ax1.set_title(f"Flow Response Around {peak['Date']:%m/%d/%Y} "
                       f"(Peak Daily Rainfall {peak['Rainfall_in']:.2f} in)", fontsize=13)
        ax1.legend(handles, labels, loc="upper right", fontsize=9)
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        ax1.grid(True, linewidth=0.4, alpha=0.4)

    fig.autofmt_xdate(rotation=45)
    fig.suptitle("Zoomed Flow Response to the Largest, Well-Separated Rainfall Events – Richmond RRWWTF",
                 fontsize=16, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG5_PATH, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: flow vs. rainfall over time, dry days removed (wet days only)
# ---------------------------------------------------------------------------

def plot_flow_vs_rainfall_wet_days_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dry days (Rainfall_in == 0) and days with unknown/missing rainfall are
    excluded entirely -- only confirmed wet days (Rainfall_in > 0) remain.
    The flow line connects only the actual kept (wet-day) observations,
    directly in chronological order; no placeholder/break points are drawn
    for the removed dry days (consistent with the wet-weather-only figure
    from the earlier exploratory step).
    """
    wet_df = df[df["Rainfall_in"] > 0].dropna(subset=["Total_Treated_MGD"]).sort_values("Date")

    fig, ax1 = plt.subplots(figsize=(22, 9))
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    bars = ax2.bar(wet_df["Date"], wet_df["Rainfall_in"], width=1.5, color=RAIN_COLOR,
                    alpha=0.65, label="Plant Rainfall")
    ax2.set_ylabel("Plant Rainfall (in)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    line, = ax1.plot(wet_df["Date"], wet_df["Total_Treated_MGD"], color=FLOW_COLOR, linewidth=0.8,
                      marker="o", markersize=3, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
                      label="Daily Total Treated Flow (Wet Days Only)")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)

    ax1.set_title("Flow vs. Rainfall Over Time – Wet Days Only – Richmond RRWWTF\n"
                   "(Dry days and days with unknown rainfall removed)", fontsize=15)
    ax1.legend([line, bars], ["Daily Total Treated Flow (Wet Days Only)", "Plant Rainfall"], loc="upper right")
    style_ts_axes(ax1, fig)

    fig.tight_layout()
    fig.savefig(FIG6_PATH, dpi=DPI)
    plt.close(fig)
    return wet_df


# ---------------------------------------------------------------------------
# Yearly rainfall summary CSV
# ---------------------------------------------------------------------------

def build_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["Rainfall_in"])
    rows = []
    for year, g in valid.groupby("Year"):
        wet = g.loc[g["Rainfall_in"] > 0, "Rainfall_in"]
        rows.append({
            "Year": int(year),
            "Total_Rainfall_in": g["Rainfall_in"].sum(),
            "Wet_Day_Count": int((g["Rainfall_in"] > 0).sum()),
            "Dry_Day_Count": int((g["Rainfall_in"] == 0).sum()),
            "Max_Daily_Rainfall_in": g["Rainfall_in"].max(),
            "Mean_Rainfall_on_Wet_Days_in": wet.mean() if not wet.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data-quality checks (console only)
# ---------------------------------------------------------------------------

def run_data_quality_checks(df: pd.DataFrame):
    print("DATA QUALITY FLAGS")
    print()

    negative_rain = df[df["Rainfall_in"] < 0]
    print(f"Negative rainfall values: {len(negative_rain)}")
    if not negative_rain.empty:
        for _, row in negative_rain.iterrows():
            print(f"    {row['Date'].date()}  Rainfall_in={row['Rainfall_in']}  Source_File={row['Source_File']}")

    high_rain = df[df["Rainfall_in"] > RAINFALL_QA_THRESHOLD_IN]
    print(f"Daily rainfall values above {RAINFALL_QA_THRESHOLD_IN} in (flagged for manual QA, not removed): {len(high_rain)}")
    if not high_rain.empty:
        for _, row in high_rain.sort_values("Rainfall_in", ascending=False).iterrows():
            print(f"    {row['Date'].date()}  Rainfall_in={row['Rainfall_in']:.2f} in  Source_File={row['Source_File']}")

    earliest, latest = df["Date"].min(), df["Date"].max()
    full_range = pd.date_range(earliest, latest, freq="D")
    present_dates = pd.DatetimeIndex(df["Date"].unique())
    missing_calendar_dates = full_range.difference(present_dates)
    missing_ranges = collapse_to_ranges(list(missing_calendar_dates))
    print(f"Missing calendar dates (no plant record at all) within {earliest.date()}-{latest.date()}: "
          f"{len(missing_calendar_dates)} day(s) across {len(missing_ranges)} gap(s)")
    for start, end in missing_ranges:
        n_days = (end - start).days + 1
        print(f"    {start.date()} to {end.date()}  ({n_days} days)")

    n_missing_rain_values = int(df["Rainfall_in"].isna().sum())
    print(f"Missing/NaN Rainfall_in values on otherwise-present plant records: {n_missing_rain_values}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    run_data_quality_checks(df)

    top_n = plot_rainfall_timeseries(df)
    monthly = plot_monthly_and_seasonal_rainfall(df)
    wet_dry_counts = plot_wet_day_histogram(df)
    plot_full_period_hyetograph(df)

    zoom_events = select_zoom_events(df)
    plot_zoomed_hyetographs(df, zoom_events)

    wet_only_df = plot_flow_vs_rainfall_wet_days_only(df)

    yearly_summary = build_yearly_summary(df)
    yearly_summary.to_csv(YEARLY_SUMMARY_CSV, index=False)

    print("RAINFALL ANALYSIS COMPLETE")
    print()
    print("Master dataset used:")
    print(MASTER_BASELINE_CSV.relative_to(BASE_DIR).as_posix())
    print()
    print("Date range:")
    print(f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    print()
    print(f"Wet days: {wet_dry_counts['wet_days']:,}  |  Dry days: {wet_dry_counts['dry_days']:,}  |  "
          f"Missing rainfall days: {wet_dry_counts['missing_days']:,}")
    print()
    print(f"Top {TOP_N_RAIN_DAYS} rainfall days:")
    for _, row in top_n.sort_values("Rainfall_in", ascending=False).iterrows():
        print(f"    {row['Date'].date()}  {row['Rainfall_in']:.2f} in")
    print()
    print("Zoomed events selected (largest, mutually well-separated by "
          f">= {ZOOM_MIN_SEPARATION_DAYS} days):")
    for _, row in zoom_events.iterrows():
        print(f"    {row['Date'].date()}  peak {row['Rainfall_in']:.2f} in")
    print()
    print("Yearly rainfall summary:")
    print(yearly_summary.to_string(index=False))
    print()
    print(f"Flow vs. rainfall (wet days only) observations plotted: {len(wet_only_df):,}")
    print()
    print("Generated files:")
    for p in [FIG1_PATH, FIG2_PATH, FIG3_PATH, FIG4_PATH, FIG5_PATH, FIG6_PATH, YEARLY_SUMMARY_CSV]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

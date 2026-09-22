"""
RRWWTF Antecedent Dry-Weather Baseline Comparison

Reads the already-validated master dataset
(output/richmond_master_flow_rainfall_dataset.csv) and compares four
definitions of a "dry-weather" observation:

  1. Simple:   Rainfall_in <= DRY_RAIN_THRESHOLD on the day itself.
  2. 3-Day:    the day itself AND each of the previous 3 calendar days
               all have Rainfall_in <= DRY_RAIN_THRESHOLD.
  3. 5-Day:    the day itself AND each of the previous 5 calendar days.
  4. 7-Day:    the day itself AND each of the previous 7 calendar days.

This does NOT calculate I&I, RDII, storm volumes, correlations, or any
predictive feature. It only builds four alternative dry-weather subsets from
the master dataset and compares how the estimated flow baseline changes as
the antecedent dry requirement gets stricter. No outliers are removed, no
observations are fabricated, and the master dataset is never modified.

Calendar-day correctness: the antecedent window is evaluated on Rainfall_in
reindexed onto a COMPLETE daily calendar (not on adjacent dataframe rows).
A missing calendar date, or a missing/NaN rainfall value, breaks the
antecedent chain and disqualifies that day -- it is never treated as dry.

Usage:
    python scripts/dry_weather_baseline_analysis.py
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_flow_rainfall_dataset.csv"

OUT_DIR = OUTPUT_DIR / "02_dry_weather_baseline_analysis"

DRY_3DAY_CSV = OUT_DIR / "01_dry_weather_3day.csv"
DRY_5DAY_CSV = OUT_DIR / "02_dry_weather_5day.csv"
DRY_7DAY_CSV = OUT_DIR / "03_dry_weather_7day.csv"
COMPARISON_SUMMARY_CSV = OUT_DIR / "dry_weather_comparison_summary.csv"
MONTHLY_COMPARISON_CSV = OUT_DIR / "monthly_dry_weather_median_comparison.csv"

FIG1_PATH = OUT_DIR / "dry_weather_definition_comparison.png"
FIG2_PATH = OUT_DIR / "dry_weather_distribution_comparison.png"
FIG3_PATH = OUT_DIR / "monthly_dry_weather_median_comparison.png"

DRY_RAIN_THRESHOLD = 0.0

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SIMPLE_COLOR = "#6b7280"
DRY3_COLOR = "#2b6cb0"
DRY5_COLOR = "#7c3aed"
DRY7_COLOR = "#b45309"


# ---------------------------------------------------------------------------
# Antecedent dry-flag computation (calendar-day correct)
# ---------------------------------------------------------------------------

def full_calendar_rainfall(df: pd.DataFrame, full_range: pd.DatetimeIndex) -> pd.Series:
    """Reindex Rainfall_in onto a complete daily calendar. Missing calendar
    dates become NaN -- they are never treated as dry."""
    return df.drop_duplicates(subset="Date").set_index("Date")["Rainfall_in"].reindex(full_range)


def antecedent_dry_flags(rain_full: pd.Series, threshold: float, n_prev_days: int) -> pd.Series:
    """
    A day qualifies only if it and each of the previous n_prev_days
    calendar days all have a valid (non-NaN) Rainfall_in <= threshold.
    NaN rainfall never counts as dry (NaN <= threshold evaluates False).
    Because rain_full is on a gap-free daily calendar, a rolling window of
    (n_prev_days + 1) rows is exactly (n_prev_days + 1) consecutive
    calendar days -- a missing calendar date simply appears as NaN/False
    and breaks the window, correctly disqualifying the day.
    """
    is_dry = rain_full <= threshold
    window = n_prev_days + 1
    dry_count = is_dry.rolling(window=window, min_periods=window).sum()
    return dry_count == window


# ---------------------------------------------------------------------------
# Stats helper (visual/comparison reference only)
# ---------------------------------------------------------------------------

def flow_stats(sub_df: pd.DataFrame) -> dict:
    valid = sub_df["Total_Treated_MGD"].dropna()
    n = int(len(valid))
    if n == 0:
        return dict(Count=0, Mean=np.nan, Median=np.nan, StdDev=np.nan,
                    Min=np.nan, P25=np.nan, P75=np.nan, IQR=np.nan, Max=np.nan)
    p25 = valid.quantile(0.25)
    p75 = valid.quantile(0.75)
    return dict(
        Count=n,
        Mean=valid.mean(),
        Median=valid.median(),
        StdDev=valid.std(),
        Min=valid.min(),
        P25=p25,
        P75=p75,
        IQR=p75 - p25,
        Max=valid.max(),
    )


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:,.{nd}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    rain_full = full_calendar_rainfall(df, full_range)

    flags_full = {
        "Dry_3Day": antecedent_dry_flags(rain_full, DRY_RAIN_THRESHOLD, 3),
        "Dry_5Day": antecedent_dry_flags(rain_full, DRY_RAIN_THRESHOLD, 5),
        "Dry_7Day": antecedent_dry_flags(rain_full, DRY_RAIN_THRESHOLD, 7),
    }

    # Map flags back to the actual plant observations only (no fabricated rows)
    df["Dry_Simple"] = (df["Rainfall_in"] <= DRY_RAIN_THRESHOLD).fillna(False)
    for col, series in flags_full.items():
        df[col] = df["Date"].map(series).fillna(False).astype(bool)

    export_cols = ["Date", "Total_Treated_MGD", "Rainfall_in", "Year", "Month", "Source_File",
                   "Dry_Simple", "Dry_3Day", "Dry_5Day", "Dry_7Day"]

    simple_df = df.loc[df["Dry_Simple"], export_cols].sort_values("Date").reset_index(drop=True)
    dry3_df = df.loc[df["Dry_3Day"], export_cols].sort_values("Date").reset_index(drop=True)
    dry5_df = df.loc[df["Dry_5Day"], export_cols].sort_values("Date").reset_index(drop=True)
    dry7_df = df.loc[df["Dry_7Day"], export_cols].sort_values("Date").reset_index(drop=True)

    dry3_df.to_csv(DRY_3DAY_CSV, index=False)
    dry5_df.to_csv(DRY_5DAY_CSV, index=False)
    dry7_df.to_csv(DRY_7DAY_CSV, index=False)

    # --- Comparison summary (visual/tabular reference only) ---
    definitions = [
        ("Rain=0 (Simple)", simple_df),
        ("3-Day Antecedent Dry", dry3_df),
        ("5-Day Antecedent Dry", dry5_df),
        ("7-Day Antecedent Dry", dry7_df),
    ]
    comparison_rows = []
    for label, sub in definitions:
        stats = flow_stats(sub)
        stats["Definition"] = label
        comparison_rows.append(stats)
    comparison_df = pd.DataFrame(comparison_rows)[
        ["Definition", "Count", "Mean", "Median", "StdDev", "Min", "P25", "P75", "IQR", "Max"]
    ]
    comparison_df.to_csv(COMPARISON_SUMMARY_CSV, index=False)

    # --- Monthly median baseline comparison (3/5/7-day only) ---
    monthly_rows = []
    for m in range(1, 13):
        row = {"Month": MONTH_NAMES[m - 1]}
        for label, sub in [("Dry_3Day_Median", dry3_df), ("Dry_5Day_Median", dry5_df),
                            ("Dry_7Day_Median", dry7_df)]:
            vals = sub.loc[sub["Month"] == m, "Total_Treated_MGD"].dropna()
            row[label] = vals.median() if not vals.empty else np.nan
        monthly_rows.append(row)
    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(MONTHLY_COMPARISON_CSV, index=False)

    # =======================================================================
    # Visualization 1: dry-weather time-series comparison (4 stacked plots)
    # =======================================================================
    y_valid_all = simple_df["Total_Treated_MGD"].dropna()
    y_pad = (y_valid_all.max() - y_valid_all.min()) * 0.05
    y_min = y_valid_all.min() - y_pad
    y_max = y_valid_all.max() + y_pad

    fig1, axes = plt.subplots(4, 1, figsize=(20, 20), sharex=True, sharey=True)
    panel_specs = [
        (axes[0], "Simple Dry (Rainfall Today = 0)", simple_df, SIMPLE_COLOR),
        (axes[1], "3-Day Antecedent Dry", dry3_df, DRY3_COLOR),
        (axes[2], "5-Day Antecedent Dry", dry5_df, DRY5_COLOR),
        (axes[3], "7-Day Antecedent Dry", dry7_df, DRY7_COLOR),
    ]
    for ax, label, sub, color in panel_specs:
        s = sub.dropna(subset=["Total_Treated_MGD"]).drop_duplicates(subset="Date").set_index("Date")["Total_Treated_MGD"]
        s_full = s.reindex(full_range)  # breaks the line across any excluded/missing day
        ax.plot(s_full.index, s_full.values, color=color, linewidth=0.6,
                marker="o", markersize=2, markerfacecolor=color, markeredgewidth=0,
                label=label)
        ax.set_ylabel("Total Treated\nFlow (MGD)")
        ax.set_title(label, fontsize=13, loc="left")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.set_ylim(y_min, y_max)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig1.autofmt_xdate(rotation=45)
    fig1.suptitle("Comparison of Dry-Weather Flow Definitions – Richmond RRWWTF", fontsize=18, y=0.995)
    fig1.tight_layout(rect=[0, 0, 1, 0.98])
    fig1.savefig(FIG1_PATH, dpi=300)

    # =======================================================================
    # Visualization 2: flow distribution comparison (boxplots, outliers shown)
    # =======================================================================
    box_data = [sub["Total_Treated_MGD"].dropna().values for _, sub in definitions]
    box_labels = ["Rain = 0", "3-Day", "5-Day", "7-Day"]

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.boxplot(box_data, tick_labels=box_labels, showfliers=True)
    ax2.set_title("Dry-Weather Flow Distribution by Antecedent Dry Period", fontsize=15)
    ax2.set_xlabel("Dry-Weather Definition")
    ax2.set_ylabel("Total Treated Flow (MGD)")
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig2.tight_layout()
    fig2.savefig(FIG2_PATH, dpi=300)

    # =======================================================================
    # Visualization 3: monthly median dry-weather flow comparison
    # =======================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(range(1, 13), monthly_df["Dry_3Day_Median"], color=DRY3_COLOR, linewidth=1.5,
             marker="o", markersize=6, label="3-Day Dry")
    ax3.plot(range(1, 13), monthly_df["Dry_5Day_Median"], color=DRY5_COLOR, linewidth=1.5,
             marker="s", markersize=6, label="5-Day Dry")
    ax3.plot(range(1, 13), monthly_df["Dry_7Day_Median"], color=DRY7_COLOR, linewidth=1.5,
             marker="^", markersize=6, label="7-Day Dry")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(MONTH_ABBR)
    ax3.set_title("Monthly Median Dry-Weather Flow – Antecedent Dry Comparison", fontsize=15)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Median Total Treated Flow (MGD)")
    ax3.grid(True, linewidth=0.4, alpha=0.5)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(FIG3_PATH, dpi=300)

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

    # =======================================================================
    # Quality control
    # =======================================================================
    n_simple = int(df["Dry_Simple"].sum())
    n_3day = int(df["Dry_3Day"].sum())
    n_5day = int(df["Dry_5Day"].sum())
    n_7day = int(df["Dry_7Day"].sum())

    print("QUALITY CONTROL")
    print()
    print("Number of dates that qualify under each definition:")
    print(f"  Simple (Rain=0):        {n_simple}")
    print(f"  3-Day Antecedent Dry:   {n_3day}")
    print(f"  5-Day Antecedent Dry:   {n_5day}")
    print(f"  7-Day Antecedent Dry:   {n_7day}")
    print()

    ordering_ok = n_7day <= n_5day <= n_3day <= n_simple
    print(f"Expected ordering (7-day <= 5-day <= 3-day <= simple): "
          f"{'PASS' if ordering_ok else 'FAIL'}")
    print()

    dates3 = set(df.loc[df["Dry_3Day"], "Date"])
    dates5 = set(df.loc[df["Dry_5Day"], "Date"])
    dates7 = set(df.loc[df["Dry_7Day"], "Date"])

    check_3_contains_5 = dates5.issubset(dates3)
    check_5_contains_7 = dates7.issubset(dates5)
    check_3_contains_7 = dates7.issubset(dates3)

    print(f"Nested-filter check: every Dry_5Day date also in Dry_3Day: "
          f"{'PASS' if check_3_contains_5 else 'FAIL'}")
    print(f"Nested-filter check: every Dry_7Day date also in Dry_5Day: "
          f"{'PASS' if check_5_contains_7 else 'FAIL'}")
    print(f"Nested-filter check: every Dry_7Day date also in Dry_3Day: "
          f"{'PASS' if check_3_contains_7 else 'FAIL'}")
    print()

    # Random manual-inspection sample of qualifying 5-day dates
    random.seed(42)
    sample_n = min(3, len(dry5_df))
    sample_rows = dry5_df.sample(n=sample_n, random_state=42).sort_values("Date")
    print(f"Manual inspection of {sample_n} randomly sampled 5-Day Antecedent Dry dates:")
    print()
    for _, row in sample_rows.iterrows():
        d = row["Date"]
        print(f"  Target Date:  {d.date()}")
        print(f"  Target Flow:  {row['Total_Treated_MGD']} MGD")
        for k in range(0, 6):
            check_date = d - pd.Timedelta(days=k)
            rain_val = rain_full.get(check_date, np.nan)
            label = "Rain today" if k == 0 else f"Rain {k} day{'s' if k > 1 else ''} before"
            print(f"    {label:<22} ({check_date.date()}): {rain_val}")
        print()

    # =======================================================================
    # Final summary
    # =======================================================================
    print("ANTECEDENT DRY-WEATHER ANALYSIS COMPLETE")
    print()
    print("Master dataset:")
    print(MASTER_CSV.relative_to(BASE_DIR).as_posix())
    print()
    print("Simple Rain=0 observations:")
    print(n_simple)
    print()
    print("3-day antecedent dry observations:")
    print(n_3day)
    print()
    print("5-day antecedent dry observations:")
    print(n_5day)
    print()
    print("7-day antecedent dry observations:")
    print(n_7day)
    print()

    print("Comparison table:")
    print()
    header = f"{'Definition':<22}{'Count':>8}{'Mean':>10}{'Median':>10}{'Std Dev':>10}{'Min':>8}{'25th Pct':>10}{'75th Pct':>10}{'IQR':>8}{'Max':>8}"
    print(header)
    for _, row in comparison_df.iterrows():
        print(f"{row['Definition']:<22}{row['Count']:>8}{fmt(row['Mean']):>10}{fmt(row['Median']):>10}"
              f"{fmt(row['StdDev']):>10}{fmt(row['Min']):>8}{fmt(row['P25']):>10}{fmt(row['P75']):>10}"
              f"{fmt(row['IQR']):>8}{fmt(row['Max']):>8}")
    print()

    print("Nested-filter check:")
    print(f"3-day contains 5-day: {'PASS' if check_3_contains_5 else 'FAIL'}")
    print(f"5-day contains 7-day: {'PASS' if check_5_contains_7 else 'FAIL'}")
    print()

    print("Generated files:")
    for p in [DRY_3DAY_CSV, DRY_5DAY_CSV, DRY_7DAY_CSV, COMPARISON_SUMMARY_CSV,
              MONTHLY_COMPARISON_CSV, FIG1_PATH, FIG2_PATH, FIG3_PATH]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

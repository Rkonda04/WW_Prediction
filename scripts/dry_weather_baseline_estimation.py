"""
RRWWTF Dry-Weather Baseline Estimation (5-Day Antecedent Dry Definition)

Builds a TIME-VARYING expected dry-weather flow baseline from the master
dataset (output/richmond_master_flow_rainfall_dataset.csv), using the 5-day
antecedent dry-weather definition selected as the working definition after
the prior antecedent-dry comparison step.

This is a WORKING analytical choice, not an established regulatory or
engineering standard for antecedent dry-day count.

Baseline assignment priority per Year-Month:
    1. Year-Month median (if >= MIN_DRY_DAYS_PER_MONTH qualifying 5-day dry
       observations exist for that specific Year-Month)
    2. Calendar-Month fallback (median across all years for that calendar
       month, if that pooled sample itself meets MIN_DRY_DAYS_PER_MONTH)
    3. Overall fallback (median of every qualifying 5-day dry observation in
       the whole dataset)

This step only ESTABLISHES the baseline. It does not subtract the baseline
from observed flow, does not compute excess flow/I&I/RDII, does not define
storm events, and does not remove any observation. The master dataset is
never modified -- a new dataset with baseline columns appended is written
separately.

Usage:
    python scripts/dry_weather_baseline_estimation.py
"""

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
MASTER_DIR = OUTPUT_DIR / "00_master_dataset"
MASTER_CSV = MASTER_DIR / "richmond_master_flow_rainfall_dataset.csv"

ANALYSIS_DIR = OUTPUT_DIR / "02_dry_weather_baseline_analysis"
MASTER_WITH_BASELINE_CSV = MASTER_DIR / "richmond_master_with_dry_weather_baseline.csv"
MONTH_YEAR_BASELINE_CSV = ANALYSIS_DIR / "month_year_dry_weather_baseline.csv"
CALENDAR_MONTH_BASELINE_CSV = ANALYSIS_DIR / "calendar_month_dry_weather_baseline.csv"

FIG1_PATH = ANALYSIS_DIR / "historical_flow_with_baseline.png"
FIG2_PATH = ANALYSIS_DIR / "dry_weather_flow_with_baseline.png"
FIG3_PATH = ANALYSIS_DIR / "dry_weather_baseline_trend.png"
FIG4_PATH = ANALYSIS_DIR / "dry_weather_monthly_sample_size.png"

DRY_RAIN_THRESHOLD = 0.0
ANTECEDENT_DAYS = 5           # working definition selected from prior comparison
MIN_DRY_DAYS_PER_MONTH = 5    # configurable minimum sample size for a trusted Year-Month baseline

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FLOW_COLOR = "#2b6cb0"
BASELINE_COLOR = "#c0392b"
DRY_POINT_COLOR = "#7c3aed"
SOURCE_COLORS = {
    "Year-Month Median": "#2b6cb0",
    "Calendar-Month Fallback": "#d97706",
    "Overall Fallback": "#b91c1c",
}


# ---------------------------------------------------------------------------
# Antecedent dry-flag computation -- identical, unchanged logic from the
# prior antecedent dry-weather comparison step.
# ---------------------------------------------------------------------------

def full_calendar_rainfall(df: pd.DataFrame, full_range: pd.DatetimeIndex) -> pd.Series:
    return df.drop_duplicates(subset="Date").set_index("Date")["Rainfall_in"].reindex(full_range)


def antecedent_dry_flags(rain_full: pd.Series, threshold: float, n_prev_days: int) -> pd.Series:
    is_dry = rain_full <= threshold
    window = n_prev_days + 1
    dry_count = is_dry.rolling(window=window, min_periods=window).sum()
    return dry_count == window


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:,.{nd}f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    rain_full = full_calendar_rainfall(df, full_range)
    dry5_full = antecedent_dry_flags(rain_full, DRY_RAIN_THRESHOLD, ANTECEDENT_DAYS)

    df["Dry_5Day"] = df["Date"].map(dry5_full).fillna(False).astype(bool)

    dry5_valid = df[df["Dry_5Day"] & df["Total_Treated_MGD"].notna()].copy()

    # =======================================================================
    # STEP 1: Year-Month dry-weather baseline stats
    # =======================================================================
    all_year_months = df[["Year", "Month"]].drop_duplicates().sort_values(["Year", "Month"]).reset_index(drop=True)

    def group_stats(g):
        valid = g["Total_Treated_MGD"]
        n = int(len(valid))
        if n == 0:
            return pd.Series({
                "Dry_Day_Count": 0, "Dry_Flow_Mean_MGD": np.nan, "Dry_Flow_Median_MGD": np.nan,
                "Dry_Flow_Std_MGD": np.nan, "Dry_Flow_Min_MGD": np.nan, "Dry_Flow_25th_MGD": np.nan,
                "Dry_Flow_75th_MGD": np.nan, "Dry_Flow_IQR_MGD": np.nan, "Dry_Flow_Max_MGD": np.nan,
            })
        p25 = valid.quantile(0.25)
        p75 = valid.quantile(0.75)
        return pd.Series({
            "Dry_Day_Count": n,
            "Dry_Flow_Mean_MGD": valid.mean(),
            "Dry_Flow_Median_MGD": valid.median(),
            "Dry_Flow_Std_MGD": valid.std(),
            "Dry_Flow_Min_MGD": valid.min(),
            "Dry_Flow_25th_MGD": p25,
            "Dry_Flow_75th_MGD": p75,
            "Dry_Flow_IQR_MGD": p75 - p25,
            "Dry_Flow_Max_MGD": valid.max(),
        })

    ym_stats = dry5_valid.groupby(["Year", "Month"]).apply(group_stats, include_groups=False).reset_index()

    month_year_df = all_year_months.merge(ym_stats, on=["Year", "Month"], how="left")
    month_year_df["Dry_Day_Count"] = month_year_df["Dry_Day_Count"].fillna(0).astype(int)
    month_year_df["Sufficient_Dry_Data"] = month_year_df["Dry_Day_Count"] >= MIN_DRY_DAYS_PER_MONTH

    # =======================================================================
    # STEP 2: Calendar-month fallback (pooled across all years)
    # =======================================================================
    calendar_rows = []
    for m in range(1, 13):
        vals = dry5_valid.loc[dry5_valid["Month"] == m, "Total_Treated_MGD"]
        n = int(len(vals))
        calendar_rows.append({
            "Month": m,
            "Dry_Day_Count": n,
            "Median_Dry_Flow_MGD": vals.median() if n > 0 else np.nan,
        })
    calendar_df = pd.DataFrame(calendar_rows)
    calendar_df["Sufficient_Dry_Data"] = calendar_df["Dry_Day_Count"] >= MIN_DRY_DAYS_PER_MONTH

    # =======================================================================
    # STEP 3: Overall fallback
    # =======================================================================
    overall_median = dry5_valid["Total_Treated_MGD"].median()
    overall_count = int(len(dry5_valid))

    # =======================================================================
    # STEP 4/5: Select baseline per Year-Month, priority Year-Month > Calendar > Overall
    # =======================================================================
    def select_baseline(row):
        if row["Sufficient_Dry_Data"]:
            return pd.Series({
                "Selected_Baseline_MGD": row["Dry_Flow_Median_MGD"],
                "Baseline_Source": "Year-Month Median",
                "Baseline_Dry_Day_Count": row["Dry_Day_Count"],
            })
        cal_row = calendar_df.loc[calendar_df["Month"] == row["Month"]].iloc[0]
        if cal_row["Sufficient_Dry_Data"]:
            return pd.Series({
                "Selected_Baseline_MGD": cal_row["Median_Dry_Flow_MGD"],
                "Baseline_Source": "Calendar-Month Fallback",
                "Baseline_Dry_Day_Count": int(cal_row["Dry_Day_Count"]),
            })
        return pd.Series({
            "Selected_Baseline_MGD": overall_median,
            "Baseline_Source": "Overall Fallback",
            "Baseline_Dry_Day_Count": overall_count,
        })

    baseline_selection = month_year_df.apply(select_baseline, axis=1)
    month_year_df = pd.concat([month_year_df, baseline_selection], axis=1)

    # month_year_df keeps ALL columns (including Baseline_Dry_Day_Count) for
    # internal use below; the CSV export uses the exact column set from the spec.
    export_col_order = ["Year", "Month", "Dry_Day_Count", "Dry_Flow_Mean_MGD", "Dry_Flow_Median_MGD",
                         "Dry_Flow_Std_MGD", "Dry_Flow_Min_MGD", "Dry_Flow_25th_MGD", "Dry_Flow_75th_MGD",
                         "Dry_Flow_IQR_MGD", "Dry_Flow_Max_MGD", "Sufficient_Dry_Data",
                         "Selected_Baseline_MGD", "Baseline_Source"]
    month_year_df[export_col_order].to_csv(MONTH_YEAR_BASELINE_CSV, index=False)

    calendar_out = calendar_df[["Month", "Dry_Day_Count", "Median_Dry_Flow_MGD"]].copy()
    calendar_out.insert(1, "Month_Name", [MONTH_NAMES[m - 1] for m in calendar_out["Month"]])
    calendar_out.to_csv(CALENDAR_MONTH_BASELINE_CSV, index=False)

    # =======================================================================
    # STEP 7: Assign baseline to every row of the complete master dataset
    # =======================================================================
    baseline_lookup = month_year_df[["Year", "Month", "Selected_Baseline_MGD",
                                      "Baseline_Source", "Baseline_Dry_Day_Count"]].rename(
        columns={"Selected_Baseline_MGD": "Expected_Baseline_MGD"})

    master_with_baseline = df.merge(baseline_lookup, on=["Year", "Month"], how="left")
    master_with_baseline = master_with_baseline.sort_values("Date").reset_index(drop=True)
    master_with_baseline.to_csv(MASTER_WITH_BASELINE_CSV, index=False)

    # =======================================================================
    # Visualization 1: historical flow + expected baseline
    # =======================================================================
    flow_series = df.drop_duplicates(subset="Date").set_index("Date")["Total_Treated_MGD"].reindex(full_range)
    baseline_series = master_with_baseline.drop_duplicates(subset="Date").set_index("Date")["Expected_Baseline_MGD"].reindex(full_range)

    fig1, ax1 = plt.subplots(figsize=(22, 9))
    ax1.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=0.6,
              marker="o", markersize=1.8, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
              label="Observed Daily Flow", zorder=2)
    ax1.plot(baseline_series.index, baseline_series.values, color=BASELINE_COLOR, linewidth=2.0,
              label="Expected Dry-Weather Baseline", zorder=3)
    ax1.set_title("Historical Wastewater Flow and Estimated Dry-Weather Baseline – Richmond RRWWTF", fontsize=16)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)")
    ax1.grid(True, linewidth=0.4, alpha=0.5)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig1.autofmt_xdate(rotation=45)
    fig1.tight_layout()
    fig1.savefig(FIG1_PATH, dpi=300)
    plt.close(fig1)

    # =======================================================================
    # Visualization 2: dry-weather observations + baseline
    # =======================================================================
    fig2, ax2 = plt.subplots(figsize=(22, 9))
    ax2.scatter(dry5_valid["Date"], dry5_valid["Total_Treated_MGD"], s=10, color=DRY_POINT_COLOR,
                alpha=0.75, label="5-Day Antecedent Dry-Weather Observations", zorder=2)
    ax2.plot(baseline_series.index, baseline_series.values, color=BASELINE_COLOR, linewidth=2.0,
              label="Expected Dry-Weather Baseline", zorder=3)
    ax2.set_title("5-Day Antecedent Dry-Weather Flow and Estimated Baseline – Richmond RRWWTF", fontsize=16)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Treated Flow (MGD)")
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig2.autofmt_xdate(rotation=45)
    fig2.tight_layout()
    fig2.savefig(FIG2_PATH, dpi=300)
    plt.close(fig2)

    # =======================================================================
    # Visualization 3: Year-Month baseline trend, colored by source
    # =======================================================================
    trend_df = month_year_df.copy()
    trend_df["YM_Date"] = pd.to_datetime(dict(year=trend_df["Year"], month=trend_df["Month"], day=1))
    trend_df = trend_df.sort_values("YM_Date")

    fig3, ax3 = plt.subplots(figsize=(22, 9))
    ax3.plot(trend_df["YM_Date"], trend_df["Selected_Baseline_MGD"], color="#9ca3af",
              linewidth=1.0, zorder=1)
    for source, color in SOURCE_COLORS.items():
        sub = trend_df[trend_df["Baseline_Source"] == source]
        ax3.scatter(sub["YM_Date"], sub["Selected_Baseline_MGD"], color=color, s=45,
                    label=source, zorder=2, edgecolor="white", linewidth=0.5)
    ax3.set_title("Estimated Dry-Weather Baseline Over Time – Richmond RRWWTF", fontsize=16)
    ax3.set_xlabel("Year-Month")
    ax3.set_ylabel("Expected Dry-Weather Flow (MGD)")
    ax3.grid(True, linewidth=0.4, alpha=0.5)
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig3.autofmt_xdate(rotation=45)
    fig3.tight_layout()
    fig3.savefig(FIG3_PATH, dpi=300)
    plt.close(fig3)

    # =======================================================================
    # Visualization 4: baseline sample size per Year-Month
    # =======================================================================
    fig4, ax4 = plt.subplots(figsize=(22, 9))
    ax4.bar(trend_df["YM_Date"], trend_df["Dry_Day_Count"], width=20, color=FLOW_COLOR, alpha=0.8)
    ax4.axhline(MIN_DRY_DAYS_PER_MONTH, color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                label=f"MIN_DRY_DAYS_PER_MONTH = {MIN_DRY_DAYS_PER_MONTH}")
    ax4.set_title("5-Day Dry-Weather Observations Available by Month", fontsize=16)
    ax4.set_xlabel("Year-Month")
    ax4.set_ylabel("Number of Qualifying 5-Day Dry-Weather Observations")
    ax4.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax4.legend(loc="upper right")
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig4.autofmt_xdate(rotation=45)
    fig4.tight_layout()
    fig4.savefig(FIG4_PATH, dpi=300)
    plt.close(fig4)

    # =======================================================================
    # Quality control
    # =======================================================================
    n_total_obs = len(df)
    n_qualifying = len(dry5_valid)
    n_periods = len(month_year_df)
    n_sufficient = int(month_year_df["Sufficient_Dry_Data"].sum())
    n_insufficient = n_periods - n_sufficient

    source_counts = master_with_baseline["Baseline_Source"].value_counts()
    n_ym_source = int(source_counts.get("Year-Month Median", 0))
    n_cal_source = int(source_counts.get("Calendar-Month Fallback", 0))
    n_overall_source = int(source_counts.get("Overall Fallback", 0))

    lowest10 = month_year_df.sort_values("Selected_Baseline_MGD").head(10)
    highest10 = month_year_df.sort_values("Selected_Baseline_MGD", ascending=False).head(10)

    print("QUALITY CONTROL")
    print()
    print(f"Total master observations: {n_total_obs}")
    print(f"Total qualifying 5-day dry-weather observations: {n_qualifying}")
    print(f"Number of Year-Month periods: {n_periods}")
    print(f"Number of Year-Month periods with >= {MIN_DRY_DAYS_PER_MONTH} qualifying dry days: {n_sufficient}")
    print(f"Number of Year-Month periods with < {MIN_DRY_DAYS_PER_MONTH} qualifying dry days: {n_insufficient}")
    print()
    print("Number of master observations using each baseline source:")
    print(f"  Year-Month Median:       {n_ym_source}")
    print(f"  Calendar-Month Fallback: {n_cal_source}")
    print(f"  Overall Fallback:        {n_overall_source}")
    print()
    print(f"Overall 5-day dry-weather median: {fmt(overall_median)} MGD")
    print(f"Lowest selected monthly baseline: {fmt(month_year_df['Selected_Baseline_MGD'].min())} MGD")
    print(f"Highest selected monthly baseline: {fmt(month_year_df['Selected_Baseline_MGD'].max())} MGD")
    print()

    print("10 Year-Month periods with the LOWEST selected baseline (validation only):")
    print()
    for _, row in lowest10.iterrows():
        print(f"  {int(row['Year'])}-{int(row['Month']):02d}  Dry_Day_Count={int(row['Dry_Day_Count'])}  "
              f"Selected_Baseline_MGD={fmt(row['Selected_Baseline_MGD'])}  Source={row['Baseline_Source']}")
    print()
    print("10 Year-Month periods with the HIGHEST selected baseline (validation only):")
    print()
    for _, row in highest10.iterrows():
        print(f"  {int(row['Year'])}-{int(row['Month']):02d}  Dry_Day_Count={int(row['Dry_Day_Count'])}  "
              f"Selected_Baseline_MGD={fmt(row['Selected_Baseline_MGD'])}  Source={row['Baseline_Source']}")
    print()

    print("DRY-WEATHER BASELINE ANALYSIS COMPLETE")
    print()
    print("Generated files:")
    for p in [MONTH_YEAR_BASELINE_CSV, CALENDAR_MONTH_BASELINE_CSV, MASTER_WITH_BASELINE_CSV,
              FIG1_PATH, FIG2_PATH, FIG3_PATH, FIG4_PATH]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

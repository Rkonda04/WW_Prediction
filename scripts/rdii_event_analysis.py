"""
RRWWTF Excess Flow (RDII) and Rain-Event Aggregation

Descriptive I&I quantification stage -- NO modeling/prediction is performed
here. This script:

    1. Computes daily excess flow (observed flow minus the already-computed
       time-varying dry-weather baseline) and a QA check that the baseline
       is unbiased on dry days.
    2. Delineates rain events from the plant-recorded rainfall record, each
       extended with a 2-day drain-down tail, merging events whose tails
       overlap the next event's rain days. Any event window touching a
       missing calendar date, or a row with NaN flow/rainfall, is flagged
       "incomplete" and excluded from summary statistics (but still listed).
    3. Produces event-level scatter/histogram/seasonal figures using
       complete events only.
    4. Saves an events CSV (all events, flagged) and a one-row overall
       summary CSV, with a console summary of how many events were
       excluded specifically because of the known 2023/2024 data gaps.
    5. A small sensitivity check against the 3-day/7-day dry-weather
       baselines IF a full Year-Month baseline table exists for them from
       the earlier comparison step -- skipped (not rebuilt) if it doesn't.

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_dry_weather_baseline.csv
        Date, Total_Treated_MGD, Rainfall_in, Year, Month, Source_File,
        Source_Year, Dry_5Day, Expected_Baseline_MGD, Baseline_Source,
        Baseline_Dry_Day_Count
    output/02_dry_weather_baseline_analysis/ (reused, not rebuilt)

KNOWN DATA GAP (client cannot provide more): 2023 raw logs cover only
June-September; the 2024 record is also partial (through Aug 2024, plus one
blank cell right at the end of August). These gaps are handled explicitly
via the completeness flag below -- no event or statistic is allowed to
silently span a missing date.

Usage:
    python scripts/rdii_event_analysis.py
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
MASTER_BASELINE_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_dry_weather_baseline.csv"
BASELINE_DIR = OUTPUT_DIR / "02_dry_weather_baseline_analysis"

EVENTS_DIR = OUTPUT_DIR / "05_rdii_events"

EVENTS_CSV = EVENTS_DIR / "rdii_events.csv"
OVERALL_SUMMARY_CSV = EVENTS_DIR / "rdii_overall_summary.csv"

FIG_RAIN_VS_VOLUME = EVENTS_DIR / "01_rainfall_vs_rdii_volume.png"
FIG_RAIN_VS_PEAK = EVENTS_DIR / "02_rainfall_vs_peak_excess_flow.png"
FIG_DAYS_TO_PEAK = EVENTS_DIR / "03_days_to_peak_distribution.png"
FIG_SEASONAL = EVENTS_DIR / "04_seasonal_rdii_volume.png"

DRAIN_DOWN_TAIL_DAYS = 2
DRY_DAY_QA_WARNING_FRACTION = 0.05  # warn if |dry-day mean excess| exceeds 5% of mean dry-day baseline

# Known gap windows (for the "excluded due to 2023/2024 gap" breakdown only --
# the completeness flag itself is computed from actual missing data, not from
# these hardcoded windows).
GAP_2023_A = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-05-31"))
GAP_2023_B = (pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-31"))
GAP_2024_TAIL_START = pd.Timestamp("2024-08-29")

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
BASELINE_COLOR = "#c0392b"
INCOMPLETE_COLOR = "#9ca3af"
FIT_COLOR = "#27ae60"

DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
# Load data + full calendar reindex (for honest missing-date detection)
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(MASTER_BASELINE_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    indexed = df.drop_duplicates(subset="Date").set_index("Date")
    full = indexed.reindex(full_range)
    full["Row_Exists"] = full_range.isin(indexed.index)
    full.index.name = "Date"
    return df, full, full_range


# ---------------------------------------------------------------------------
# Step 1: daily excess flow + dry-day QA check
# ---------------------------------------------------------------------------

def compute_excess_flow(full: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    full["Excess_Flow_MGD"] = full["Total_Treated_MGD"] - full["Expected_Baseline_MGD"]
    full["RDII_MGD"] = full["Excess_Flow_MGD"].clip(lower=0)
    return full


def run_dry_day_qa(full: pd.DataFrame):
    dry = full[full["Dry_5Day"] == True].dropna(subset=["Excess_Flow_MGD", "Expected_Baseline_MGD"])
    mean_excess = dry["Excess_Flow_MGD"].mean()
    std_excess = dry["Excess_Flow_MGD"].std()
    mean_baseline = dry["Expected_Baseline_MGD"].mean()
    frac = abs(mean_excess) / mean_baseline if mean_baseline else np.nan

    print("DRY-DAY QA CHECK (excess flow should be near zero on dry days)")
    print(f"  Dry-day (5-day antecedent) observations checked: {len(dry)}")
    print(f"  Mean excess on dry days:   {mean_excess:.4f} MGD")
    print(f"  Std dev of dry-day excess: {std_excess:.4f} MGD")
    print(f"  Mean dry-day baseline:     {mean_baseline:.4f} MGD")
    print(f"  |mean excess| / mean baseline = {frac:.2%}")
    if frac > DRY_DAY_QA_WARNING_FRACTION:
        print(f"  WARNING: dry-day mean excess exceeds {DRY_DAY_QA_WARNING_FRACTION:.0%} of baseline "
              f"-- the baseline may be biased.")
    else:
        print("  OK: dry-day mean excess is within tolerance.")
    print()
    return {"mean_excess": mean_excess, "std_excess": std_excess,
            "mean_baseline": mean_baseline, "fraction": frac}


# ---------------------------------------------------------------------------
# Step 2: rain event delineation with drain-down tail + gap flagging
# ---------------------------------------------------------------------------

def delineate_events(full: pd.DataFrame) -> list:
    rain_days = full.index[full["Rainfall_in"] > 0].tolist()
    raw_spans = collapse_to_ranges(rain_days)

    provisional = []
    for rain_start, rain_end in raw_spans:
        tail_end = rain_end + pd.Timedelta(days=DRAIN_DOWN_TAIL_DAYS)
        provisional.append({"rain_start": rain_start, "rain_end": rain_end, "tail_end": tail_end})

    # Merge events whose tail overlaps the next event's rain start
    merged = []
    for ev in provisional:
        if merged and ev["rain_start"] <= merged[-1]["tail_end"]:
            merged[-1]["rain_end"] = max(merged[-1]["rain_end"], ev["rain_end"])
            merged[-1]["tail_end"] = merged[-1]["rain_end"] + pd.Timedelta(days=DRAIN_DOWN_TAIL_DAYS)
        else:
            merged.append(dict(ev))

    return merged


def classify_incomplete_reason(window_dates, full: pd.DataFrame) -> str:
    """Only called for incomplete events -- explains which known gap (if any)
    the event's missing dates fall into, for the console breakdown."""
    missing = [d for d in window_dates if d not in full.index
               or not full.loc[d, "Row_Exists"]
               or pd.isna(full.loc[d, "Total_Treated_MGD"])
               or pd.isna(full.loc[d, "Rainfall_in"])]
    if any(GAP_2023_A[0] <= d <= GAP_2023_A[1] or GAP_2023_B[0] <= d <= GAP_2023_B[1] for d in missing):
        return "2023 gap"
    if any(d >= GAP_2024_TAIL_START for d in missing):
        return "2024 tail gap"
    return "other isolated gap"


def build_events_table(full: pd.DataFrame, merged_events: list) -> pd.DataFrame:
    rows = []
    for i, ev in enumerate(merged_events, start=1):
        window_dates = pd.date_range(ev["rain_start"], ev["tail_end"], freq="D")
        in_window = full.reindex(window_dates)

        is_complete = bool(
            all(d in full.index for d in window_dates) and
            in_window["Row_Exists"].fillna(False).all() and
            in_window["Total_Treated_MGD"].notna().all() and
            in_window["Rainfall_in"].notna().all()
        )

        rain_vals = in_window["Rainfall_in"].dropna()
        flow_vals = in_window["Total_Treated_MGD"].dropna()
        excess_vals = in_window["Excess_Flow_MGD"].dropna()
        rdii_vals = in_window["RDII_MGD"].dropna()

        peak_flow = flow_vals.max() if not flow_vals.empty else np.nan
        peak_flow_date = flow_vals.idxmax() if not flow_vals.empty else pd.NaT
        peak_excess = excess_vals.max() if not excess_vals.empty else np.nan
        days_to_peak = (peak_flow_date - ev["rain_start"]).days if pd.notna(peak_flow_date) else np.nan

        incomplete_reason = "" if is_complete else classify_incomplete_reason(window_dates, full)

        rows.append({
            "Event_ID": f"EV{i:04d}",
            "Start_Date": ev["rain_start"],
            "Rain_End_Date": ev["rain_end"],
            "End_Date": ev["tail_end"],
            "Duration_Days": (ev["tail_end"] - ev["rain_start"]).days + 1,
            "Total_Rainfall_in": rain_vals.sum() if not rain_vals.empty else np.nan,
            "Max_Daily_Rainfall_in": rain_vals.max() if not rain_vals.empty else np.nan,
            "Peak_Flow_MGD": peak_flow,
            "Peak_Flow_Date": peak_flow_date,
            "Peak_Excess_Flow_MGD": peak_excess,
            "Total_Excess_Volume_MG": rdii_vals.sum() if not rdii_vals.empty else np.nan,
            "Days_To_Peak": days_to_peak,
            "Complete": is_complete,
            "Incomplete_Reason": incomplete_reason,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: event-level figures (complete events only for stats/fit)
# ---------------------------------------------------------------------------

def plot_rain_vs_volume(events: pd.DataFrame) -> float:
    complete = events[events["Complete"]].dropna(subset=["Total_Rainfall_in", "Total_Excess_Volume_MG"])
    incomplete = events[~events["Complete"]].dropna(subset=["Total_Rainfall_in", "Total_Excess_Volume_MG"])

    fig, ax = plt.subplots(figsize=(11, 8))
    if not incomplete.empty:
        ax.scatter(incomplete["Total_Rainfall_in"], incomplete["Total_Excess_Volume_MG"],
                   color=INCOMPLETE_COLOR, alpha=0.5, s=35, label="Incomplete event (excluded from fit)")

    slope = np.nan
    if len(complete) >= 2:
        slope, intercept = np.polyfit(complete["Total_Rainfall_in"], complete["Total_Excess_Volume_MG"], 1)
        x_fit = np.linspace(complete["Total_Rainfall_in"].min(), complete["Total_Rainfall_in"].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=FIT_COLOR, linewidth=2,
                 label=f"Fit: {slope:.3f} MG RDII per inch of rain")
    ax.scatter(complete["Total_Rainfall_in"], complete["Total_Excess_Volume_MG"],
               color=FLOW_COLOR, s=45, label="Complete event", zorder=3)

    ax.set_title("Event Rainfall vs. Total RDII Volume – Richmond RRWWTF (Complete Events Only)")
    ax.set_xlabel("Total Event Rainfall (in)")
    ax.set_ylabel("Total Excess Flow / RDII Volume (MG)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_RAIN_VS_VOLUME, dpi=DPI)
    plt.close(fig)
    return slope


def plot_rain_vs_peak_excess(events: pd.DataFrame):
    complete = events[events["Complete"]].dropna(subset=["Total_Rainfall_in", "Peak_Excess_Flow_MGD"])
    incomplete = events[~events["Complete"]].dropna(subset=["Total_Rainfall_in", "Peak_Excess_Flow_MGD"])

    fig, ax = plt.subplots(figsize=(11, 8))
    if not incomplete.empty:
        ax.scatter(incomplete["Total_Rainfall_in"], incomplete["Peak_Excess_Flow_MGD"],
                   color=INCOMPLETE_COLOR, alpha=0.5, s=35, label="Incomplete event (excluded)")
    ax.scatter(complete["Total_Rainfall_in"], complete["Peak_Excess_Flow_MGD"],
               color=FLOW_COLOR, s=45, label="Complete event", zorder=3)

    ax.set_title("Event Rainfall vs. Peak Excess Flow – Richmond RRWWTF (Complete Events Only)")
    ax.set_xlabel("Total Event Rainfall (in)")
    ax.set_ylabel("Peak Excess Flow (MGD above baseline)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_RAIN_VS_PEAK, dpi=DPI)
    plt.close(fig)


def plot_days_to_peak(events: pd.DataFrame):
    complete = events[events["Complete"]].dropna(subset=["Days_To_Peak"])

    fig, ax = plt.subplots(figsize=(10, 7))
    max_lag = int(complete["Days_To_Peak"].max()) if not complete.empty else 1
    bins = np.arange(-0.5, max_lag + 1.5, 1)
    ax.hist(complete["Days_To_Peak"], bins=bins, color=FLOW_COLOR, edgecolor="white", alpha=0.85)
    ax.set_title("Distribution of Days-to-Peak-Flow – Richmond RRWWTF (Complete Events Only)")
    ax.set_xlabel("Days from First Rain Day to Peak Flow")
    ax.set_ylabel("Number of Events")
    ax.set_xticks(range(0, max_lag + 1))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DAYS_TO_PEAK, dpi=DPI)
    plt.close(fig)


def plot_seasonal_rdii(events: pd.DataFrame):
    complete = events[events["Complete"]].dropna(subset=["Total_Excess_Volume_MG"]).copy()
    complete["Month"] = complete["Start_Date"].dt.month
    monthly = complete.groupby("Month")["Total_Excess_Volume_MG"].sum().reindex(range(1, 13), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(range(1, 13), monthly.values, color=FLOW_COLOR)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_title("Total RDII Volume by Calendar Month – All Years Pooled (Complete Events Only)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total RDII Volume (MG)")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.text(0.5, -0.22, "Note: Jan-May and Oct-Dec pools exclude 2023 (raw logs cover only "
                          "Jun-Sep 2023); Sep-Dec pools also exclude 2024 (partial record).",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(FIG_SEASONAL, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 5: sensitivity check against 3-day/7-day baselines, if they exist
# ---------------------------------------------------------------------------

def sensitivity_check(events: pd.DataFrame, full: pd.DataFrame):
    print("SENSITIVITY CHECK (3-day / 7-day dry-weather baselines)")
    candidates = {
        "3-day": BASELINE_DIR / "month_year_dry_weather_baseline_3day.csv",
        "7-day": BASELINE_DIR / "month_year_dry_weather_baseline_7day.csv",
    }
    for label, path in candidates.items():
        if not path.exists():
            print(f"  {label}: SKIPPED -- no full Year-Month baseline table found at "
                  f"{path.relative_to(BASE_DIR).as_posix()} (only a calendar-month median "
                  f"comparison exists from the earlier antecedent-dry comparison; not rebuilding it here).")
            continue
        alt_baseline = pd.read_csv(path, parse_dates=False)
        alt_lookup = alt_baseline.set_index(["Year", "Month"])["Selected_Baseline_MGD"]
        alt_full = full.copy()
        alt_full["Alt_Baseline_MGD"] = [
            alt_lookup.get((d.year, d.month), np.nan) for d in alt_full.index
        ]
        alt_full["Alt_RDII_MGD"] = (alt_full["Total_Treated_MGD"] - alt_full["Alt_Baseline_MGD"]).clip(lower=0)
        complete_ids = events.loc[events["Complete"], "Event_ID"]
        total = 0.0
        for eid, ev in zip(complete_ids, events[events["Complete"]].itertuples()):
            window = pd.date_range(ev.Start_Date, ev.End_Date, freq="D")
            total += alt_full.reindex(window)["Alt_RDII_MGD"].dropna().sum()
        print(f"  {label}: total RDII volume across complete events = {total:.2f} MG")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    df, full, full_range = load_data()
    full = compute_excess_flow(full)

    run_dry_day_qa(full)

    merged_events = delineate_events(full)
    events = build_events_table(full, merged_events)
    events.to_csv(EVENTS_CSV, index=False)

    slope = plot_rain_vs_volume(events)
    plot_rain_vs_peak_excess(events)
    plot_days_to_peak(events)
    plot_seasonal_rdii(events)

    complete = events[events["Complete"]]
    incomplete = events[~events["Complete"]]

    per_event_ratio = (complete["Total_Excess_Volume_MG"] / complete["Total_Rainfall_in"]).replace(
        [np.inf, -np.inf], np.nan).dropna()

    overall = pd.DataFrame([{
        "Num_Complete_Events": len(complete),
        "Num_Incomplete_Events": len(incomplete),
        "Total_RDII_Volume_MG": complete["Total_Excess_Volume_MG"].sum(),
        "Median_RDII_Per_Inch_MG": per_event_ratio.median() if not per_event_ratio.empty else np.nan,
        "Median_Days_To_Peak": complete["Days_To_Peak"].median(),
        "Fitted_Slope_MG_Per_Inch": slope,
    }])
    overall.to_csv(OVERALL_SUMMARY_CSV, index=False)

    sensitivity_check(events, full)

    # =======================================================================
    # Console summary
    # =======================================================================
    reason_counts = incomplete["Incomplete_Reason"].value_counts()

    print("RDII EVENT ANALYSIS COMPLETE")
    print()
    print(f"Total events delineated: {len(events)}")
    print(f"  Complete:   {len(complete)}")
    print(f"  Incomplete: {len(incomplete)}")
    print("  Incomplete breakdown by cause:")
    for reason, count in reason_counts.items():
        print(f"    {reason}: {count}")
    print()
    print(f"Total RDII volume (complete events): {overall['Total_RDII_Volume_MG'].iloc[0]:.2f} MG")
    print(f"Median RDII per inch of rain (per-event ratio): {overall['Median_RDII_Per_Inch_MG'].iloc[0]:.3f} MG/in")
    print(f"Fitted slope (all complete events pooled): {slope:.3f} MG/in")
    print(f"Median days-to-peak: {overall['Median_Days_To_Peak'].iloc[0]:.1f} days")
    print()
    print("Generated files:")
    for p in [EVENTS_CSV, OVERALL_SUMMARY_CSV, FIG_RAIN_VS_VOLUME, FIG_RAIN_VS_PEAK,
              FIG_DAYS_TO_PEAK, FIG_SEASONAL]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

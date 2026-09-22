"""
RRWWTF RDII Event Analysis -- QA Artifacts and Refinements (Follow-up Pass)

Companion to rdii_event_analysis.py. Reuses that script's data loading and
event-delineation logic directly (imported, not re-derived) so the event
definitions stay identical -- this pass only adds QA artifacts that
previously lived in console output only, plus a few additional diagnostic
outputs. All of rdii_event_analysis.py's original output files are left
untouched; every file this script writes is new.

Adds, all under output/05_rdii_events/:
    1. daily_excess_flow.csv        -- full daily excess-flow series
       dry_day_baseline_check.csv   -- dry-day baseline bias check, overall + per year
    2. min_event_threshold_sensitivity.csv -- overall stats at 3 rainfall thresholds
    3. rdii_share_of_treated_volume.csv    -- RDII as % of treated flow, overall + per year
    4. 05_event_outlier_inspection.png     -- multi-panel look at top/long-lag events
    5. 06_rainfall_vs_rdii_volume_filtered.png -- refreshed scatter at the recommended threshold

No modeling. No changes to the dry-weather baseline or event-delineation
methodology already established.

Usage:
    python scripts/rdii_qa_refinements.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rdii_event_analysis as rdii

# ---------------------------------------------------------------------------
# Configuration -- new output files only; nothing from rdii_event_analysis.py
# is overwritten.
# ---------------------------------------------------------------------------

BASE_DIR = rdii.BASE_DIR
EVENTS_DIR = rdii.EVENTS_DIR

DAILY_EXCESS_CSV = EVENTS_DIR / "daily_excess_flow.csv"
DRY_DAY_CHECK_CSV = EVENTS_DIR / "dry_day_baseline_check.csv"
THRESHOLD_SENSITIVITY_CSV = EVENTS_DIR / "min_event_threshold_sensitivity.csv"
RDII_SHARE_CSV = EVENTS_DIR / "rdii_share_of_treated_volume.csv"

FIG_OUTLIER_INSPECTION = EVENTS_DIR / "05_event_outlier_inspection.png"
FIG_FILTERED_SCATTER = EVENTS_DIR / "06_rainfall_vs_rdii_volume_filtered.png"

RAIN_THRESHOLDS_IN = [0.0, 0.25, 0.5]  # 0.0 = "all complete events" (no threshold)
STABILITY_TOLERANCE = 0.25  # relative spread (max-min)/mean below which slopes are called "stable"

TOP_N_BY_VOLUME = 5
LONG_LAG_DAYS_THRESHOLD = 6
INSPECTION_WINDOW_PAD_DAYS = 5

INCOMPLETE_YEARS_NOTE = (
    "2023 covers only Jun-Sep (raw logs unavailable for the rest of the year); "
    "2024 covers only Jan-Aug. Their RDII-share figures are not representative of a full year."
)


# ---------------------------------------------------------------------------
# 1a. Daily excess-flow CSV
# ---------------------------------------------------------------------------

def save_daily_excess_flow(full: pd.DataFrame):
    out = full.reset_index().rename(columns={"index": "Date"})
    out = out[["Date", "Total_Treated_MGD", "Expected_Baseline_MGD", "Excess_Flow_MGD", "RDII_MGD",
               "Row_Exists", "Dry_5Day"]]
    out.to_csv(DAILY_EXCESS_CSV, index=False)


# ---------------------------------------------------------------------------
# 1b. Dry-day baseline check, overall + per year, with PASS/WARN verdict
# ---------------------------------------------------------------------------

def dry_day_baseline_check(full: pd.DataFrame) -> pd.DataFrame:
    dry = full[full["Dry_5Day"] == True].dropna(subset=["Excess_Flow_MGD", "Expected_Baseline_MGD"]).copy()
    dry["Year"] = dry.index.year

    overall_mean_baseline = dry["Expected_Baseline_MGD"].mean()

    def row_for(label, g):
        mean_excess = g["Excess_Flow_MGD"].mean()
        frac = abs(mean_excess) / overall_mean_baseline if overall_mean_baseline else np.nan
        verdict = "WARN" if frac > rdii.DRY_DAY_QA_WARNING_FRACTION else "PASS"
        return {
            "Group": label,
            "N_Dry_Days": int(len(g)),
            "Mean_Excess_MGD": mean_excess,
            "Median_Excess_MGD": g["Excess_Flow_MGD"].median(),
            "Std_Excess_MGD": g["Excess_Flow_MGD"].std(),
            "Mean_Baseline_MGD": g["Expected_Baseline_MGD"].mean(),
            "Fraction_of_Overall_Baseline": frac,
            "Verdict": verdict,
        }

    rows = [row_for("Overall", dry)]
    for year, g in dry.groupby("Year"):
        rows.append(row_for(str(year), g))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Minimum-event-threshold sensitivity
# ---------------------------------------------------------------------------

def fit_slope(sub: pd.DataFrame):
    sub = sub.dropna(subset=["Total_Rainfall_in", "Total_Excess_Volume_MG"])
    if len(sub) < 2:
        return np.nan
    slope, _ = np.polyfit(sub["Total_Rainfall_in"], sub["Total_Excess_Volume_MG"], 1)
    return slope


def threshold_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    complete = events[events["Complete"]]
    rows = []
    for thr in RAIN_THRESHOLDS_IN:
        sub = complete[complete["Total_Rainfall_in"] >= thr]
        ratio = (sub["Total_Excess_Volume_MG"] / sub["Total_Rainfall_in"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "Rainfall_Threshold_in": thr,
            "Num_Events": int(len(sub)),
            "Total_RDII_Volume_MG": sub["Total_Excess_Volume_MG"].sum(),
            "Fitted_Slope_MG_Per_Inch": fit_slope(sub),
            "Median_RDII_Per_Inch_MG": ratio.median() if not ratio.empty else np.nan,
            "Median_Days_To_Peak": sub["Days_To_Peak"].median(),
        })
    return pd.DataFrame(rows)


def recommend_threshold(sensitivity: pd.DataFrame):
    slopes = sensitivity["Fitted_Slope_MG_Per_Inch"].dropna()
    if len(slopes) < 2 or slopes.mean() == 0:
        return None, "insufficient data to assess stability"
    spread = (slopes.max() - slopes.min()) / abs(slopes.mean())
    if spread <= STABILITY_TOLERANCE:
        recommended = sensitivity["Rainfall_Threshold_in"].max()
        note = (f"Fitted slope is stable across thresholds (relative spread {spread:.1%} "
                f"<= {STABILITY_TOLERANCE:.0%}) -- recommend the strictest threshold "
                f"({recommended:.2f} in) since it removes small-event noise without changing the result.")
    else:
        recommended = 0.25
        note = (f"Fitted slope is NOT stable across thresholds (relative spread {spread:.1%} "
                f"> {STABILITY_TOLERANCE:.0%}) -- small events materially change the fit. "
                f"Recommending the middle threshold (0.25 in) as a balance; treat the fitted "
                f"slope as sensitive to event-inclusion criteria.")
    return recommended, note


# ---------------------------------------------------------------------------
# 3. RDII as a share of treated volume, overall + per year
# ---------------------------------------------------------------------------

def rdii_share_of_treated_volume(full: pd.DataFrame) -> pd.DataFrame:
    valid = full.dropna(subset=["RDII_MGD", "Total_Treated_MGD"]).copy()
    valid["Year"] = valid.index.year

    def row_for(label, g):
        rdii_total = g["RDII_MGD"].sum()
        flow_total = g["Total_Treated_MGD"].sum()
        pct = (rdii_total / flow_total * 100) if flow_total else np.nan
        return {
            "Group": label,
            "N_Valid_Days": int(len(g)),
            "Total_RDII_Volume_MG": rdii_total,
            "Total_Observed_Flow_MG": flow_total,
            "RDII_Percent_of_Flow": pct,
        }

    rows = [row_for("Overall", valid)]
    for year, g in valid.groupby("Year"):
        rows.append(row_for(str(year), g))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Outlier / long-lag event inspection
# ---------------------------------------------------------------------------

def classify_event_pattern(event_row, full: pd.DataFrame) -> str:
    if not event_row["Complete"]:
        return "possible baseline issue (event window touches a data gap)"

    rain_window = pd.date_range(event_row["Start_Date"], event_row["Rain_End_Date"], freq="D")
    rain_days = [d for d in rain_window if full.loc[d, "Rainfall_in"] > 0]
    raw_spans = rdii.collapse_to_ranges(rain_days)

    if len(raw_spans) > 1:
        return f"looks merged ({len(raw_spans)} separate rain pulses combined into one event)"
    return "looks genuine (single coherent rain pulse, complete data)"


def select_inspection_events(events: pd.DataFrame) -> pd.DataFrame:
    top_volume = events.dropna(subset=["Total_Excess_Volume_MG"]).nlargest(TOP_N_BY_VOLUME, "Total_Excess_Volume_MG")
    long_lag = events[events["Days_To_Peak"] >= LONG_LAG_DAYS_THRESHOLD]
    selected = pd.concat([top_volume, long_lag]).drop_duplicates(subset="Event_ID").sort_values("Start_Date")
    return selected


def plot_outlier_inspection(selected: pd.DataFrame, full: pd.DataFrame, notes: list):
    n = len(selected)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(18, 6 * n))
    if n == 1:
        axes = [axes]

    for ax1, (_, ev) in zip(axes, selected.iterrows()):
        start = ev["Start_Date"] - pd.Timedelta(days=INSPECTION_WINDOW_PAD_DAYS)
        end = ev["End_Date"] + pd.Timedelta(days=INSPECTION_WINDOW_PAD_DAYS)
        window = pd.date_range(start, end, freq="D")
        win_df = full.reindex(window)

        ax2 = ax1.twinx()
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        rain_valid = win_df.dropna(subset=["Rainfall_in"])
        bars = ax2.bar(rain_valid.index, rain_valid["Rainfall_in"], width=0.8, color=rdii.RAIN_COLOR,
                        alpha=0.6, label="Plant Rainfall")
        ax2.set_ylabel("Rainfall (in)", color="#1a5276")
        ax2.tick_params(axis="y", labelcolor="#1a5276")

        flow_valid = win_df.dropna(subset=["Total_Treated_MGD"])
        line, = ax1.plot(flow_valid.index, flow_valid["Total_Treated_MGD"], color=rdii.FLOW_COLOR,
                          linewidth=1.3, marker="o", markersize=4, label="Observed Flow")
        baseline_line, = ax1.plot(win_df.index, win_df["Expected_Baseline_MGD"], color=rdii.BASELINE_COLOR,
                                   linewidth=1.8, linestyle="--", label="Dry-Weather Baseline")
        ax1.axvspan(ev["Start_Date"], ev["End_Date"], color="#9ca3af", alpha=0.15)

        ax1.set_ylabel("Flow (MGD)", color=rdii.FLOW_COLOR)
        ax1.tick_params(axis="y", labelcolor=rdii.FLOW_COLOR)
        verdict = classify_event_pattern(ev, full)
        notes.append((ev["Event_ID"], verdict))
        ax1.set_title(f"{ev['Event_ID']}  {ev['Start_Date'].date()} to {ev['End_Date'].date()}  "
                       f"|  Rain {ev['Total_Rainfall_in']:.2f} in  |  RDII {ev['Total_Excess_Volume_MG']:.2f} MG  "
                       f"|  Days-to-peak {ev['Days_To_Peak']}  |  {verdict}", fontsize=11)
        ax1.legend([line, baseline_line, bars], ["Observed Flow", "Dry-Weather Baseline", "Plant Rainfall"],
                   loc="upper right", fontsize=8)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
        ax1.grid(True, linewidth=0.4, alpha=0.4)

    fig.autofmt_xdate(rotation=45)
    fig.suptitle("Outlier / Long-Lag Event Inspection – Richmond RRWWTF", fontsize=16, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG_OUTLIER_INSPECTION, dpi=rdii.DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Refreshed scatter at the recommended threshold
# ---------------------------------------------------------------------------

def plot_filtered_scatter(events: pd.DataFrame, threshold: float):
    complete = events[events["Complete"]]
    sub = complete[complete["Total_Rainfall_in"] >= threshold].dropna(
        subset=["Total_Rainfall_in", "Total_Excess_Volume_MG"])

    fig, ax = plt.subplots(figsize=(11, 8))
    slope = np.nan
    if len(sub) >= 2:
        slope, intercept = np.polyfit(sub["Total_Rainfall_in"], sub["Total_Excess_Volume_MG"], 1)
        x_fit = np.linspace(sub["Total_Rainfall_in"].min(), sub["Total_Rainfall_in"].max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=rdii.FIT_COLOR, linewidth=2,
                 label=f"Fit: {slope:.3f} MG RDII per inch of rain")
    ax.scatter(sub["Total_Rainfall_in"], sub["Total_Excess_Volume_MG"], color=rdii.FLOW_COLOR,
               s=50, label=f"Complete event, rainfall >= {threshold:.2f} in", zorder=3)

    ax.set_title(f"Event Rainfall vs. Total RDII Volume – Richmond RRWWTF\n"
                 f"(Rainfall >= {threshold:.2f} in, Complete Events Only)", fontsize=14)
    ax.set_xlabel("Total Event Rainfall (in)")
    ax.set_ylabel("Total Excess Flow / RDII Volume (MG)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FILTERED_SCATTER, dpi=rdii.DPI)
    plt.close(fig)
    return slope, len(sub)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    df, full, full_range = rdii.load_data()
    full = rdii.compute_excess_flow(full)

    merged_events = rdii.delineate_events(full)
    events = rdii.build_events_table(full, merged_events)

    # --- 1. QA artifacts ---
    save_daily_excess_flow(full)
    dry_check = dry_day_baseline_check(full)
    dry_check.to_csv(DRY_DAY_CHECK_CSV, index=False)

    print("DRY-DAY BASELINE CHECK (overall + per year)")
    print(dry_check.to_string(index=False))
    print()

    # --- 2. Minimum-event-threshold sensitivity ---
    sensitivity = threshold_sensitivity(events)
    sensitivity.to_csv(THRESHOLD_SENSITIVITY_CSV, index=False)
    recommended, stability_note = recommend_threshold(sensitivity)

    print("MINIMUM-EVENT-THRESHOLD SENSITIVITY")
    print(sensitivity.to_string(index=False))
    print(f"  {stability_note}")
    print()

    # --- 3. RDII as share of treated volume ---
    share = rdii_share_of_treated_volume(full)
    share.to_csv(RDII_SHARE_CSV, index=False)

    print("RDII AS SHARE OF TREATED VOLUME (overall + per year)")
    print(share.to_string(index=False))
    print(f"  NOTE: {INCOMPLETE_YEARS_NOTE}")
    print()

    # --- 4. Outlier / long-lag inspection ---
    selected = select_inspection_events(events)
    notes = []
    plot_outlier_inspection(selected, full, notes)

    print(f"OUTLIER / LONG-LAG EVENT INSPECTION ({len(selected)} events: "
          f"top {TOP_N_BY_VOLUME} by RDII volume + any with days-to-peak >= {LONG_LAG_DAYS_THRESHOLD})")
    for event_id, verdict in notes:
        print(f"  {event_id}: {verdict}")
    print()

    # --- 5. Refreshed scatter at recommended threshold ---
    if recommended is not None:
        filtered_slope, n_used = plot_filtered_scatter(events, recommended)
        print(f"Refreshed scatter saved at recommended threshold ({recommended:.2f} in, {n_used} events, "
              f"slope {filtered_slope:.3f} MG/in).")
    else:
        print("Refreshed scatter skipped -- no stable recommended threshold could be determined.")
    print()

    # =======================================================================
    # Final summary
    # =======================================================================
    print("RDII QA / REFINEMENTS COMPLETE")
    print()
    print("Generated files:")
    for p in [DAILY_EXCESS_CSV, DRY_DAY_CHECK_CSV, THRESHOLD_SENSITIVITY_CSV, RDII_SHARE_CSV,
              FIG_OUTLIER_INSPECTION, FIG_FILTERED_SCATTER]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

"""
RRWWTF Flow Anomaly Screen (Phase 1 of the Data-Screening / Re-Modeling Pass)

Diagnostic data-quality screen, not modeling. Flags physically implausible
daily flow readings using three data-derived rules, writes a flagged-days
CSV, and writes a CLEANED dataset (flagged values set to NaN -- never
interpolated or filled) that every later phase in this pass must load
instead of the raw flow column.

RULES (all thresholds derived from this dataset's own distribution, not
picked arbitrarily -- see the percentile analysis behind each below):

  1. EXACT ZERO: Total_Treated_MGD == 0. A working WWTF does not
     physically stop treating flow for a full day.

  2. BELOW PLAUSIBILITY FLOOR: Total_Treated_MGD < FLOOR_FRACTION x that
     day's Expected_Baseline_MGD (the existing Year-Month dry-weather
     baseline from the RDII stage -- reused, not recomputed). FLOOR_FRACTION
     = 0.45 was chosen by checking where the ratio-to-local-baseline
     distribution actually separates: 0.45 catches exactly 5 days
     project-wide (0.50 catches 10, 0.40 catches only 2), and both of the
     known visually-flagged suspects fall well inside it
     (2024-05-23 ratio=0.418, 2024-08-19 ratio=0.0). A day trading at
     under half its own month's typical dry-weather flow, with no
     mechanism (a plant doesn't halve its base indoor-water-use flow in a
     day), is implausible on its face.

  3. IMPLAUSIBLE SPIKE, NO RAIN: Total_Treated_MGD > SPIKE_FACTOR x the
     median of its two immediate calendar-day neighbors, AND corrected
     rainfall <= NO_RAIN_THRESHOLD_IN on both t and t-1 (no rain event to
     explain an I&I-driven spike). SPIKE_FACTOR = 2.0 was chosen the same
     way: computed AFTER rule 1/2 cleaning (a raw neighbor-ratio pass
     falsely flagged 2024-08-20 at 3.95x, purely because its neighbor
     2024-08-19 was the erroneous zero from rule 1 -- once that zero is
     removed first, 2024-08-20's ratio drops to 1.97x and is correctly
     NOT flagged). With rule-1/2 cleaning applied first, 2.0x isolates
     exactly one day (2018-10-31, 4.37 MGD vs. a 1.84 MGD neighbor
     median) out of 1,709 no-rain-day observations -- consistent with the
     natural upper tail of that distribution (99.9th percentile 2.37x).
     This is why the spike rule runs in a SECOND pass, after rules 1/2.

KNOWN SUSPECTS FROM VISUAL INSPECTION (per the task brief): a ~0.0 MGD day
in late Aug 2024, and a ~0.75 MGD dip described as "June 2024." Both are
confirmed below -- the first is 2024-08-19 (0.000 MGD, rule 1). The second
is actually **2024-05-23** (0.757 MGD), not June -- there is no comparably
low day anywhere in June 2024 (June 2024's minimum is 1.522 MGD); the May
23 dip sits immediately before the "Jun 2024" tick label in the
validation-framework time-series plot, which is almost certainly what was
visually read as "June."

GAP HANDLING: everything above runs on the continuous calendar-day index
(gaps as NaN rows, never filled), consistent with every prior stage.

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_corrected_rainfall.csv
    output/05_rdii_events/daily_excess_flow.csv

Outputs:
    output/09_data_screening/flagged_days.csv
    output/09_data_screening/richmond_daily_cleaned.csv -- the dataset
        every later phase in this pass (Phase 3, Phase 4) must load.

NOTE ON OUTPUT DIRECTORY NUMBER: output/07_data_screening/ was requested,
but "07" is already output/07_correlation_matrix_analysis/ (added earlier
in this body of work) -- this script writes to output/09_data_screening/
instead, the next free number after output/08_validation_baselines/.

Usage:
    python scripts/flow_anomaly_screen.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

CORRECTED_MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_corrected_rainfall.csv"
DAILY_EXCESS_CSV = OUTPUT_DIR / "05_rdii_events" / "daily_excess_flow.csv"

ANALYSIS_DIR = OUTPUT_DIR / "09_data_screening"
FLAGGED_DAYS_CSV = ANALYSIS_DIR / "flagged_days.csv"
CLEANED_DATASET_CSV = ANALYSIS_DIR / "richmond_daily_cleaned.csv"

FLOOR_FRACTION = 0.45          # flow < this x local Expected_Baseline_MGD -> flagged
SPIKE_FACTOR = 2.0             # flow > this x neighbor-day median -> flagged (if no rain)
NO_RAIN_THRESHOLD_IN = 0.1     # corrected rainfall at or below this counts as "no rain"


def load_ts() -> pd.DataFrame:
    master = pd.read_csv(CORRECTED_MASTER_CSV, parse_dates=["Date"])
    excess = pd.read_csv(DAILY_EXCESS_CSV, parse_dates=["Date"])

    merged = excess.merge(master[["Date", "Rainfall_in_Corrected", "Rainfall_in"]], on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    ts = merged.set_index("Date")
    gaps = ts.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=1)], "Expected one row per calendar day."
    return ts


def apply_rules(ts: pd.DataFrame) -> pd.DataFrame:
    flow = ts["Total_Treated_MGD"]

    # --- Pass 1: exact zero + below-floor (independent of neighbors) ---
    ratio_to_baseline = flow / ts["Expected_Baseline_MGD"]
    is_zero = flow == 0
    is_below_floor = (ratio_to_baseline < FLOOR_FRACTION) & flow.notna() & ~is_zero

    pass1_flagged = is_zero | is_below_floor
    flow_pass1_cleaned = flow.where(~pass1_flagged)

    # --- Pass 2: implausible spike vs. neighbors, no rain, computed on the
    # pass-1-cleaned series so a rule-1/2 artifact can't fabricate a false
    # spike on the neighboring day ---
    neighbor_median = pd.concat([flow_pass1_cleaned.shift(1), flow_pass1_cleaned.shift(-1)], axis=1).median(axis=1)
    neighbor_ratio = flow_pass1_cleaned / neighbor_median

    rain_t = ts["Rainfall_in_Corrected"]
    rain_t1 = ts["Rainfall_in_Corrected"].shift(1)
    no_rain = (rain_t <= NO_RAIN_THRESHOLD_IN).fillna(False) & (rain_t1 <= NO_RAIN_THRESHOLD_IN).fillna(False)

    is_spike = (neighbor_ratio > SPIKE_FACTOR) & no_rain & flow_pass1_cleaned.notna()

    # --- Assemble flagged-days table (priority: zero > floor > spike, but
    # a day can only trigger one rule since spike is scored on already-
    # pass1-cleaned data) ---
    rule = pd.Series("", index=ts.index, dtype=object)
    rule[is_below_floor] = "Below_Plausibility_Floor"
    rule[is_zero] = "Exact_Zero"
    rule[is_spike] = "Implausible_Spike_No_Rain"

    flagged_mask = rule != ""

    flagged = pd.DataFrame({
        "Date": ts.index[flagged_mask],
        "Value_MGD": flow[flagged_mask].values,
        "Rule_Triggered": rule[flagged_mask].values,
        "Ratio_To_Local_Baseline": ratio_to_baseline[flagged_mask].values,
        "Neighbor_Median_MGD": neighbor_median[flagged_mask].values,
        "Ratio_To_Neighbors": neighbor_ratio[flagged_mask].values,
        "Rainfall_Corrected_t": rain_t[flagged_mask].values,
        "Rainfall_Corrected_t_minus_1": rain_t1[flagged_mask].values,
        "Expected_Baseline_MGD": ts["Expected_Baseline_MGD"][flagged_mask].values,
    }).sort_values("Date").reset_index(drop=True)

    return flagged, flagged_mask


def build_cleaned_dataset(ts: pd.DataFrame, flagged_mask: pd.Series) -> pd.DataFrame:
    cleaned = ts.reset_index().rename(columns={"index": "Date"})
    flagged_mask_reset = flagged_mask.reset_index(drop=True)

    cleaned["Original_Total_Treated_MGD"] = cleaned["Total_Treated_MGD"]
    cleaned["Flow_Anomaly_Flagged"] = flagged_mask_reset.values
    cleaned["Flow_Anomaly_Rule"] = ""

    cleaned.loc[flagged_mask_reset.values, "Total_Treated_MGD"] = np.nan
    # Excess/RDII are derived from flow -- must be blanked consistently,
    # never left computed from a flagged (now-NaN) flow value.
    cleaned.loc[flagged_mask_reset.values, "Excess_Flow_MGD"] = np.nan
    cleaned.loc[flagged_mask_reset.values, "RDII_MGD"] = np.nan

    cols = ["Date", "Total_Treated_MGD", "Original_Total_Treated_MGD", "Flow_Anomaly_Flagged",
            "Flow_Anomaly_Rule", "Expected_Baseline_MGD", "Excess_Flow_MGD", "RDII_MGD",
            "Rainfall_in_Corrected", "Rainfall_in", "Row_Exists", "Dry_5Day"]
    return cleaned[cols]


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ts = load_ts()
    print("FLOW ANOMALY SCREEN")
    print()
    print(f"Continuous calendar-day series: {ts.index.min().date()} to {ts.index.max().date()} "
          f"({len(ts):,} calendar days)")
    print(f"Rules: exact zero  |  flow < {FLOOR_FRACTION:g} x local Expected_Baseline_MGD  |  "
          f"flow > {SPIKE_FACTOR:g} x neighbor-day median with rainfall <= {NO_RAIN_THRESHOLD_IN:g} in "
          f"on both t and t-1")
    print()

    flagged, flagged_mask = apply_rules(ts)
    flagged.to_csv(FLAGGED_DAYS_CSV, index=False)

    cleaned = build_cleaned_dataset(ts, flagged_mask)
    for _, row in flagged.iterrows():
        cleaned.loc[cleaned["Date"] == row["Date"], "Flow_Anomaly_Rule"] = row["Rule_Triggered"]
    cleaned.to_csv(CLEANED_DATASET_CSV, index=False)

    print(f"Flagged days: {len(flagged)}")
    print(flagged[["Date", "Value_MGD", "Rule_Triggered", "Ratio_To_Local_Baseline",
                    "Ratio_To_Neighbors"]].to_string(index=False))
    print()

    print("Counts per year:")
    flagged_by_year = flagged.copy()
    flagged_by_year["Year"] = pd.to_datetime(flagged_by_year["Date"]).dt.year
    print(flagged_by_year.groupby("Year").size().to_string())
    print()

    # --- Confirm the two known suspects explicitly ---
    known_zero = flagged[flagged["Date"] == pd.Timestamp("2024-08-19")]
    known_dip = flagged[flagged["Date"] == pd.Timestamp("2024-05-23")]
    print("KNOWN SUSPECT CONFIRMATION")
    print(f"  ~0.0 MGD late Aug 2024: {'CONFIRMED -- ' + known_zero.iloc[0]['Rule_Triggered'] if not known_zero.empty else 'NOT FOUND'} "
          f"(2024-08-19, value={known_zero.iloc[0]['Value_MGD']:.3f} MGD)" if not known_zero.empty else "")
    print(f"  ~0.75 MGD dip: {'CONFIRMED -- ' + known_dip.iloc[0]['Rule_Triggered'] if not known_dip.empty else 'NOT FOUND'} "
          f"on 2024-05-23 (value={known_dip.iloc[0]['Value_MGD']:.3f} MGD) -- NOTE: this is May, not June; "
          f"June 2024's minimum flow is 1.522 MGD, nowhere near 0.75.")
    print()

    n_dropped_total = int(flagged_mask.sum())
    print(f"CLEANED DATASET: {n_dropped_total} of {len(ts):,} calendar days had their "
          f"Total_Treated_MGD (and derived Excess_Flow_MGD / RDII_MGD) set to NaN. "
          f"No values interpolated or filled. Original values preserved in "
          f"'Original_Total_Treated_MGD' for audit.")
    print()

    print("FLOW ANOMALY SCREEN COMPLETE")
    print()
    print("Generated files:")
    for p in [FLAGGED_DAYS_CSV, CLEANED_DATASET_CSV]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

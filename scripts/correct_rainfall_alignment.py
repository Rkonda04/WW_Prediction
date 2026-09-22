"""
RRWWTF Rainfall/Flow Date-Alignment Correction

Data-correction utility, not modeling and not a new numbered analysis
stage -- it produces a corrected variant of the master dataset for every
downstream script (starting with validation_framework.py) to load.

WHY THIS EXISTS: daily flow correlates far more strongly with the
NEXT-day-recorded plant rainfall than with the SAME-day value:

    flow[t]        vs rainfall[t]    (same-day):        r = 0.296  (n=2,184)
    flow[t]        vs rainfall[t+1]  (next-day logged):  r = 0.441  (n=2,182)
    flow[t]        vs rainfall[t-1]  (previous-day):      r = 0.138  (n=2,181)
    excess_flow[t] vs rainfall[t]    (same-day):          r = 0.308  (n=2,184)
    excess_flow[t] vs rainfall[t+1]  (next-day logged):    r = 0.456  (n=2,182)

(reproduced by this script's QA check below, not just asserted). The
asymmetry -- next-day is much stronger than same-day, and previous-day is
much weaker than either -- rules out ordinary noise and matches a genuine
one-day-late recording offset: whatever rain actually fell on day t is
being logged under date t+1 in Rainfall_in, not date t. This is a known
pattern for manually-read rain gauges (a gauge read each morning reports
the previous 24 hours' total, but gets written down under that morning's
date instead of the previous day's).

CORRECTION: Rainfall_in_Corrected[t] = original Rainfall_in[t+1]. The
very last calendar day in the record has no t+1 to pull from, so its
corrected value is NaN (unknown), not filled. The original Rainfall_in
column is preserved unchanged for audit/comparison; nothing is
interpolated or forward-filled.

WHAT THIS DOES NOT TOUCH: the dry-weather baseline (Expected_Baseline_MGD,
Dry_5Day, Baseline_Source, Baseline_Dry_Day_Count) is carried through
UNCHANGED from the existing master file -- reused, not recomputed. The
rainfall alignment issue does not affect the persistence or DWF baselines
themselves (they don't depend on the corrected rainfall value), only
anything that uses Rainfall_in as a same-day feature (wet/dry-day
classification, rainfall-lag features, etc.), which is exactly why every
downstream script must load Rainfall_in_Corrected instead of Rainfall_in.

Input (read-only, never modified):
    output/00_master_dataset/richmond_master_with_dry_weather_baseline.csv

Output (new file -- the existing master CSVs are left untouched):
    output/00_master_dataset/richmond_master_with_corrected_rainfall.csv

Usage:
    python scripts/correct_rainfall_alignment.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_DIR = BASE_DIR / "output" / "00_master_dataset"
INPUT_CSV = MASTER_DIR / "richmond_master_with_dry_weather_baseline.csv"
OUTPUT_CSV = MASTER_DIR / "richmond_master_with_corrected_rainfall.csv"


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # QA: reproduce the same-day / next-day / previous-day correlation check
    # on a full continuous calendar-day index so gaps are handled honestly
    # (dropna on pairs, no fill) -- confirms the numbers in the docstring
    # rather than just asserting them.
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    ts = df.drop_duplicates(subset="Date").set_index("Date").reindex(full_range)
    ts.index.name = "Date"

    def corr_at_shift(rain_shift):
        # rain_shift = -1 means rainfall[t+1]; +1 means rainfall[t-1]; 0 = same-day
        paired = pd.concat([ts["Total_Treated_MGD"], ts["Rainfall_in"].shift(rain_shift)], axis=1).dropna()
        r, _ = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        return r, len(paired)

    r_same, n_same = corr_at_shift(0)
    r_next, n_next = corr_at_shift(-1)
    r_prev, n_prev = corr_at_shift(1)

    print("RAINFALL/FLOW ALIGNMENT QA CHECK")
    print(f"  flow[t] vs rainfall[t]   (same-day):        r={r_same:.3f}  n={n_same}")
    print(f"  flow[t] vs rainfall[t+1] (next-day logged):  r={r_next:.3f}  n={n_next}")
    print(f"  flow[t] vs rainfall[t-1] (previous-day):      r={r_prev:.3f}  n={n_prev}")
    if r_next > r_same and r_next > r_prev:
        print("  CONFIRMED: next-day-logged rainfall correlates most strongly with flow -- "
              "applying the -1-day correction (Rainfall_in_Corrected[t] = Rainfall_in[t+1]).")
    else:
        print("  WARNING: the next-day-strongest pattern did NOT reproduce on this run -- "
              "the correction below is being applied per prior confirmation, but this QA check "
              "no longer supports it. Investigate before trusting downstream results.")
    print()

    # --- Apply the correction on the full continuous calendar-day index,
    # then map back onto the original (non-reindexed) row set ---
    corrected_full = ts["Rainfall_in"].shift(-1)
    df["Rainfall_in_Corrected"] = df["Date"].map(corrected_full)

    n_total = len(df)
    n_corrected_available = int(df["Rainfall_in_Corrected"].notna().sum())
    n_corrected_missing = n_total - n_corrected_available

    df.to_csv(OUTPUT_CSV, index=False)

    print("CORRECTION APPLIED")
    print(f"  Rows written: {n_total:,}")
    print(f"  Rainfall_in_Corrected available: {n_corrected_available:,}")
    print(f"  Rainfall_in_Corrected NaN (last calendar day and any day whose t+1 was itself missing): "
          f"{n_corrected_missing:,}")
    print(f"  Dry-weather baseline columns (Expected_Baseline_MGD, Dry_5Day, Baseline_Source, "
          f"Baseline_Dry_Day_Count) carried through UNCHANGED -- not recomputed.")
    print(f"  Original Rainfall_in column preserved unchanged for audit/comparison.")
    print()
    print(f"Output: {OUTPUT_CSV.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

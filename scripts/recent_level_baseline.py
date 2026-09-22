"""
RRWWTF Recent-Level-Aware Baseline (Phase 3 of the Data-Screening / Re-Modeling Pass)

Diagnostic baseline-upgrade script, not modeling. Adds two new naive
baselines to the validation framework's baseline set:

    5. Recent_Level_Baseline: predicted_flow[t] = rolling median of
       DRY-day flow (corrected rainfall <= 0.1 in) over the trailing 60
       calendar days ending at t-1 (never including day t itself -- no
       leakage). If fewer than MIN_VALID_DRY_DAYS (10) dry observations
       fall in that window, falls back to the last successfully computed
       recent-level value (forward-carried), never to a filled/
       interpolated flow value.
    6. Recent_Level_Hybrid: predicted_flow[t] = recent_level[t] +
       (flow[t-1] - recent_level[t-1]) -- the same persistence+baseline
       decomposition as before, but anchored to the recent-level series
       instead of the frozen Year-Month dry-weather table. This directly
       addresses the level-shift problem found in the original baseline
       run (2024 mean flow ~26.6% above the 2018-2022 train mean):
       Seasonal_DWF_Baseline and Climatology are calibrated to a stale
       train-period average and systematically underpredict 2024, while
       a 60-day rolling recent-level tracks the plant's actual current
       flow regime.

DATA: this and every later phase in this pass loads the CLEANED dataset
from Phase 1 (output/09_data_screening/richmond_daily_cleaned.csv), not
the raw flow column -- 6 flagged days (see flagged_days.csv) have
Total_Treated_MGD set to NaN there.

WHY THIS RECOMPUTES THE ORIGINAL 4 BASELINES TOO: the original
output/08_validation_baselines/baseline_scores.csv was scored on the
UNCLEANED flow series (Phase 1 postdates it), and 2 of the 6 flagged
anomalies (2024-05-23, 2024-08-19) fall inside the 2024 TEST window --
meaning persistence and the original hybrid's TEST scores there were
partly contaminated by exactly the bad readings this pass exists to fix.
Appending 2 new rows to that stale table would compare clean vs. dirty
baselines side by side. Instead, all 6 baselines (the original 4,
rerun on cleaned data, plus these 2 new ones) are scored together here
into a new, internally consistent file --
output/08_validation_baselines/baseline_scores_with_recent_level.csv --
which supersedes the original for the rest of this pass. The original
baseline_scores.csv is left untouched as the historical pre-screening
record.

Usage:
    python scripts/recent_level_baseline.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import model_eval

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

CLEANED_CSV = OUTPUT_DIR / "09_data_screening" / "richmond_daily_cleaned.csv"
BASELINE_DIR = OUTPUT_DIR / "08_validation_baselines"
UPDATED_SCORES_CSV = BASELINE_DIR / "baseline_scores_with_recent_level.csv"

WET_DAY_RAIN_THRESHOLD_IN = 0.1
HIGH_FLOW_PERCENTILE = 0.90

RECENT_LEVEL_WINDOW_DAYS = 60
MIN_VALID_DRY_DAYS = 10

BASELINE_ORDER = ["Persistence", "Seasonal_DWF_Baseline", "Climatology", "Persistence_Baseline_Hybrid",
                   "Recent_Level_Baseline", "Recent_Level_Hybrid"]


def load_ts() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    ts = df.set_index("Date")
    gaps = ts.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=1)], "Expected one row per calendar day."
    return ts


def build_climatology(ts: pd.DataFrame, train_mask: np.ndarray) -> pd.Series:
    train_sub = ts.loc[train_mask, ["Total_Treated_MGD"]].copy()
    train_sub["Month"] = train_sub.index.month
    monthly_mean = train_sub.groupby("Month")["Total_Treated_MGD"].mean()
    return pd.Series(ts.index.month, index=ts.index).map(monthly_mean)


def build_recent_level(ts: pd.DataFrame):
    dry_mask = (ts["Rainfall_in_Corrected"] <= WET_DAY_RAIN_THRESHOLD_IN).fillna(False)
    dry_flow = ts["Total_Treated_MGD"].where(dry_mask)

    # Trailing 60-CALENDAR-day window ending at t-1: the raw rolling window
    # is right-aligned ending at t (positional == calendar-day here, since
    # ts is one row per calendar day with no gaps in the index itself), so
    # shift(1) moves the window to end at t-1, excluding day t.
    rolling_raw = dry_flow.rolling(RECENT_LEVEL_WINDOW_DAYS, min_periods=MIN_VALID_DRY_DAYS).median()
    recent_level_raw = rolling_raw.shift(1)

    # Fallback: when fewer than MIN_VALID_DRY_DAYS dry observations exist
    # in the trailing window, carry the last successfully computed
    # recent-level value forward (never a raw flow value).
    recent_level = recent_level_raw.ffill()
    used_fallback = recent_level_raw.isna() & recent_level.notna()
    return recent_level, used_fallback


def build_baseline_predictions(ts: pd.DataFrame, train_mask: np.ndarray) -> dict:
    recent_level, used_fallback = build_recent_level(ts)
    predictions = {
        "Persistence": ts["Total_Treated_MGD"].shift(1),
        "Seasonal_DWF_Baseline": ts["Expected_Baseline_MGD"],
        "Climatology": build_climatology(ts, train_mask),
        "Persistence_Baseline_Hybrid": ts["Expected_Baseline_MGD"] + ts["Excess_Flow_MGD"].shift(1),
        "Recent_Level_Baseline": recent_level,
        "Recent_Level_Hybrid": recent_level + ts["Total_Treated_MGD"].shift(1) - recent_level.shift(1),
    }
    return predictions, recent_level, used_fallback


def build_wet_dry_masks(ts: pd.DataFrame):
    rain_t = ts["Rainfall_in_Corrected"]
    rain_t1 = ts["Rainfall_in_Corrected"].shift(1)
    is_wet_t = rain_t > WET_DAY_RAIN_THRESHOLD_IN
    is_wet_t1 = rain_t1 > WET_DAY_RAIN_THRESHOLD_IN
    wet_mask = is_wet_t.fillna(False) | is_wet_t1.fillna(False)
    known_mask = rain_t.notna() & rain_t1.notna()
    dry_mask = known_mask & ~wet_mask
    return wet_mask.values, dry_mask.values


def score_all_baselines(ts: pd.DataFrame, predictions: dict, test_mask: np.ndarray,
                         wet_mask: np.ndarray, dry_mask: np.ndarray, high_flow_threshold: float) -> pd.DataFrame:
    dates = ts.index.to_series().reset_index(drop=True).values
    y_true_all = ts["Total_Treated_MGD"].reset_index(drop=True)

    subset_masks = {"All": test_mask, "Dry": test_mask & dry_mask, "Wet": test_mask & wet_mask}

    rows = []
    for name in BASELINE_ORDER:
        pred_series = predictions[name].reset_index(drop=True)
        for subset_name, mask in subset_masks.items():
            hf = high_flow_threshold if subset_name == "All" else None
            metrics = model_eval.score(dates[mask], y_true_all[mask], pred_series[mask], high_flow_threshold=hf)
            row = {"Baseline": name, "Subset": subset_name}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ts = load_ts()

    print("RECENT-LEVEL-AWARE BASELINE (PHASE 3)")
    print()
    print(f"Loaded CLEANED dataset: {CLEANED_CSV.relative_to(BASE_DIR).as_posix()}")
    print(f"Continuous calendar-day series: {ts.index.min().date()} to {ts.index.max().date()} "
          f"({len(ts):,} calendar days)")
    print()

    dates = ts.index.to_series().reset_index(drop=True)
    train_mask, test_mask = model_eval.get_train_test_masks(dates)
    train_mask, test_mask = train_mask.values, test_mask.values

    predictions, recent_level, used_fallback = build_baseline_predictions(ts, train_mask)
    wet_mask, dry_mask = build_wet_dry_masks(ts)

    n_recent_level_valid_test = int((test_mask & recent_level.notna().values).sum())
    n_fallback_test = int((test_mask & used_fallback.values).sum())
    print(f"Recent-level series: {int(recent_level.notna().sum()):,} of {len(ts):,} days have a value "
          f"({n_recent_level_valid_test:,} of {int(test_mask.sum()):,} TEST days); "
          f"{int(used_fallback.sum()):,} days project-wide used the last-valid-value fallback "
          f"({n_fallback_test:,} of those in TEST) because fewer than {MIN_VALID_DRY_DAYS} dry "
          f"observations fell in their trailing {RECENT_LEVEL_WINDOW_DAYS}-day window.")
    print()

    high_flow_threshold = float(ts.loc[train_mask, "Total_Treated_MGD"].quantile(HIGH_FLOW_PERCENTILE))
    print(f"High-flow threshold (train-period {HIGH_FLOW_PERCENTILE:.0%} of flow, cleaned data): "
          f"{high_flow_threshold:.3f} MGD")
    print()

    scores = score_all_baselines(ts, predictions, test_mask, wet_mask, dry_mask, high_flow_threshold)
    scores.to_csv(UPDATED_SCORES_CSV, index=False)

    print("BASELINE SCORES -- ALL TEST DAYS (cleaned data, all 6 baselines)")
    all_scores = scores[scores["Subset"] == "All"].set_index("Baseline")
    print(all_scores[["N", "N_Excluded_NaN", "R2", "RMSE", "MAE", "MAPE", "Mean_Error",
                       "Pct_Days_Underpredicted"]].round(3).to_string())
    print()

    print("BASELINE SCORES -- WET DAYS ONLY (cleaned data, all 6 baselines)")
    wet_scores = scores[scores["Subset"] == "Wet"].set_index("Baseline")
    print(wet_scores[["N", "R2", "RMSE", "MAE", "Mean_Error"]].round(3).to_string())
    print()

    strongest_overall = all_scores["RMSE"].idxmin()
    strongest_wet = wet_scores["RMSE"].idxmin() if wet_scores["RMSE"].notna().any() else "N/A"
    print(f"Strongest baseline overall (lowest RMSE, all test days, cleaned data): {strongest_overall} "
          f"(RMSE={all_scores.loc[strongest_overall, 'RMSE']:.3f} MGD)")
    print(f"Strongest baseline on wet days: {strongest_wet}")
    print()

    print("RECENT-LEVEL BASELINE COMPLETE")
    print()
    print("Generated files:")
    print(f"  {UPDATED_SCORES_CSV.relative_to(BASE_DIR).as_posix()}")
    print()
    print(f"NOTE: original output/08_validation_baselines/baseline_scores.csv is left untouched "
          f"(pre-screening record). This new file supersedes it for the rest of this pass.")


if __name__ == "__main__":
    main()

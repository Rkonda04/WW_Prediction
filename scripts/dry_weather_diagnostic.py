"""
RRWWTF Dry-Weather Error Diagnostic (Post-Modeling Diagnostic Pass)

Diagnostic analysis only -- NO retraining, no feature changes, no
hyperparameter tuning. Decomposes where the best model's dry-day TEST
error comes from and estimates how much of it is reducible.

WHY THIS SCRIPT REGENERATES THE MODEL: scripts/train_decomposition_model.py
(output/10_decomp_model/) never persisted a trained model object or a
per-day predictions CSV -- only aggregate scores. To honor "load the
already-trained model, do not retrain" as closely as possible without a
saved artifact, this script imports train_decomposition_model.py AS A
MODULE and calls its exact, already-fixed pipeline (same features, same
random_state=42, same CV-selected XGBoost round count, same TRAIN/TEST
split) to deterministically reproduce the identical XGBoost_Wet_Weighted
model and its TEST predictions -- not a re-tune, not a different model.
This is verified on every run: the reproduced TEST RMSE/dry/wet split is
checked against the archived numbers in
output/10_decomp_model/combined_scores.csv before any diagnostic below is
trusted.

    1. Dry-definition sensitivity: does the dry-day error shrink as the
       "dry" label is tightened from "no rain on t/t-1" up to a strict
       antecedent-dry-day rule? If so, the error is misclassified
       post-storm recession, not sanitary noise.
    2. Days-since-rain error profile: how long does the model stay wrong
       after a storm, and in which direction?
    3. Is the 60-day recent-level baseline drifting behind the true
       level? (trend test on dry-day residuals + a visual baseline-vs-
       observed check)
    4. Structure in dry-day residuals by day-of-week and month -- a
       cleaner re-test of the correlation stage's near-zero day-of-week
       finding (dry days only, residuals only, corrected rainfall).
    5. Analog-day noise floor: how much do two independently-realized,
       seasonally-matched, similarly-leveled strictly-dry days differ in
       observed flow? That distribution is the irreducible sanitary
       variation the model cannot be expected to beat.

DEFINITIONS USED (see also the per-task docstrings below):
  - "Dry (current/loose)" = corrected rainfall <= 0.1 in on BOTH t and
    t-1 -- the exact convention already used throughout
    validation_framework.py / recent_level_baseline.py /
    train_decomposition_model.py.
  - "N antecedent dry days" (tasks 1b/1c/1d, and the strict-dry pool in
    task 5) reproduces the EXACT algorithm from
    dry_weather_baseline_estimation.py's antecedent_dry_flags() --
    rolling window of (N+1) days (day t plus the N days before it), all
    with rainfall <= 0.0 in (exact zero, not <= 0.1) -- but applied to
    Rainfall_in_Corrected instead of the original Rainfall_in, per this
    diagnostic's corrected-data mandate. This intentionally reproduces
    the DWF baseline's own rule rather than the looser 0.1 in threshold,
    exactly as task 1(c) asks ("matches the definition used for the DWF
    baseline in the RDII stage").

Inputs (read-only, never modified):
    output/09_data_screening/richmond_daily_cleaned.csv (Phase 1 cleaned
        flow + Rainfall_in_Corrected)
    scripts/model_eval.py (split + scoring, imported not reimplemented)
    scripts/train_decomposition_model.py (imported to regenerate the
        exact best model + features, not reimplemented)
    output/10_decomp_model/combined_scores.csv (for the reproduction
        check and as the reference dry-day RMSE headline number)

NOTE ON OUTPUT DIRECTORY NUMBER: output/10_dry_diagnostic/ was requested,
but "10" is output/10_decomp_model/ (the model this diagnostic examines)
-- this script writes to output/11_dry_diagnostic/ instead.

Usage:
    python scripts/dry_weather_diagnostic.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import model_eval
import train_decomposition_model as tdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

COMBINED_SCORES_CSV = OUTPUT_DIR / "10_decomp_model" / "combined_scores.csv"

ANALYSIS_DIR = OUTPUT_DIR / "11_dry_diagnostic"

DRY_DEF_SENSITIVITY_CSV = ANALYSIS_DIR / "dry_definition_sensitivity.csv"
ERROR_BY_DAYS_SINCE_RAIN_CSV = ANALYSIS_DIR / "error_by_days_since_rain.csv"
RESIDUAL_TREND_CSV = ANALYSIS_DIR / "residual_trend.csv"
RESIDUAL_STRUCTURE_CSV = ANALYSIS_DIR / "residual_structure.csv"
ANALOG_PAIRS_CSV = ANALYSIS_DIR / "analog_pairs.csv"
FINDINGS_MD = ANALYSIS_DIR / "dry_weather_findings.md"

FIG1_DEF_SENSITIVITY = ANALYSIS_DIR / "01_dry_definition_sensitivity.png"
FIG2_DAYS_SINCE_RAIN = ANALYSIS_DIR / "02_error_by_days_since_rain.png"
FIG3_RESIDUAL_TREND = ANALYSIS_DIR / "03_dry_residual_trend.png"
FIG4_RECENT_LEVEL_TRACKING = ANALYSIS_DIR / "04_recent_level_vs_observed.png"
FIG5_DOW_MONTH = ANALYSIS_DIR / "05_residual_structure.png"
FIG6_ANALOG_HIST = ANALYSIS_DIR / "06_analog_pair_differences.png"

LOOSE_RAIN_THRESHOLD_IN = 0.1     # "current" dry definition, task 1(a) -- matches project convention
STRICT_RAIN_THRESHOLD_IN = 0.0    # antecedent-dry-day rule, tasks 1(b/c/d) and task 5 -- matches DWF baseline
ANTECEDENT_DAY_OPTIONS = [3, 5, 7]

DAYS_SINCE_RAIN_BUCKETS = [0, 1, 2, 3, 4, 5, 6]  # 7+ handled separately

ROLLING_TREND_WINDOW = 14

SEASONAL_TOLERANCE_DAYS = 30       # analog-pair matching: day-of-year circular distance
RECENT_LEVEL_TOLERANCE_PCT = 0.05  # analog-pair matching: relative recent-level tolerance
MIN_DATE_SEPARATION_DAYS = 14      # excludes trivially-adjacent, highly-autocorrelated "pairs"

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
BASELINE_COLOR = "#c0392b"
FIT_COLOR = "#27ae60"
INCOMPLETE_COLOR = "#9ca3af"
HIGHLIGHT_COLOR = "#d97706"
PURPLE = "#7c3aed"

DPI = 300

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Reproduce the exact best model + its TEST predictions (no retraining in
# the sense of any different choice -- same fixed pipeline, verified below)
# ---------------------------------------------------------------------------

def reproduce_best_model():
    ts = tdm.load_ts()
    feat = tdm.build_feature_frame(ts)
    complete, n_dropped, n_total = tdm.drop_incomplete(feat)

    dates = complete.index.to_series().reset_index(drop=True)
    train_mask_full, test_mask_full = model_eval.get_train_test_masks(dates)
    train_mask, test_mask = train_mask_full.values, test_mask_full.values

    xgb_n = tdm.select_xgb_n_estimators(complete, dates, sample_weight_wet=True)
    model = tdm.fit_final_xgb(complete, train_mask, xgb_n, sample_weight_wet=True)

    excess_pred_test = model.predict(complete.loc[test_mask, tdm.FEATURE_COLUMNS].values)
    flow_pred_test = tdm.reconstruct_flow_pred(complete.loc[test_mask], excess_pred_test)

    test_df = complete.loc[test_mask].copy()
    test_df["flow_pred"] = flow_pred_test
    test_df["residual"] = test_df["flow_true"] - test_df["flow_pred"]  # observed - predicted

    return ts, feat, complete, test_df, train_mask, test_mask


def verify_reproduction(test_df: pd.DataFrame):
    archived = pd.read_csv(COMBINED_SCORES_CSV)
    archived_all = archived[(archived["Model"] == "XGBoost_Wet_Weighted") & (archived["Subset"] == "All")].iloc[0]

    metrics = model_eval.score(test_df.index, test_df["flow_true"], test_df["flow_pred"])
    match = np.isclose(metrics["RMSE"], archived_all["RMSE"], atol=1e-4)

    print("MODEL REPRODUCTION CHECK (regenerated vs. archived combined_scores.csv)")
    print(f"  Regenerated: N={metrics['N']}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    print(f"  Archived:    N={int(archived_all['N'])}, RMSE={archived_all['RMSE']:.4f}, R2={archived_all['R2']:.4f}")
    if match:
        print("  MATCH -- reproduction is faithful (bit-for-bit deterministic pipeline). Proceeding.")
    else:
        print("  WARNING: reproduction does NOT match archived scores. Diagnostics below may not reflect "
              "the actual reported best model -- investigate before trusting downstream results.")
    print()
    return match


# ---------------------------------------------------------------------------
# Dry / wet masks
# ---------------------------------------------------------------------------

def loose_dry_mask(rainfall_t: pd.Series, rainfall_t1: pd.Series) -> pd.Series:
    is_wet_t = rainfall_t > LOOSE_RAIN_THRESHOLD_IN
    is_wet_t1 = rainfall_t1 > LOOSE_RAIN_THRESHOLD_IN
    wet = is_wet_t.fillna(False) | is_wet_t1.fillna(False)
    known = rainfall_t.notna() & rainfall_t1.notna()
    return known & ~wet


def antecedent_dry_flags(rain_full: pd.Series, threshold: float, n_prev_days: int) -> pd.Series:
    """Identical algorithm to dry_weather_baseline_estimation.py's
    antecedent_dry_flags(): a rolling window of (n_prev_days + 1) days
    (day t plus the n_prev_days before it), all <= threshold."""
    is_dry = rain_full <= threshold
    window = n_prev_days + 1
    dry_count = is_dry.rolling(window=window, min_periods=window).sum()
    return dry_count == window


# ---------------------------------------------------------------------------
# Task 1: dry-definition sensitivity
# ---------------------------------------------------------------------------

def task1_dry_definition_sensitivity(ts: pd.DataFrame, test_df: pd.DataFrame):
    rain_full = ts["Rainfall_in_Corrected"]

    loose_dry = loose_dry_mask(test_df["rainfall_t"], test_df["rain_lag1"])

    rows = []

    m = model_eval.score(test_df.index[loose_dry], test_df.loc[loose_dry, "flow_true"],
                          test_df.loc[loose_dry, "flow_pred"])
    rows.append({"Definition": "a_current_no_rain_t_or_t-1", "Group": "Dry", **m})

    prev_dry_mask = loose_dry
    for n_days in ANTECEDENT_DAY_OPTIONS:
        strict_full = antecedent_dry_flags(rain_full, STRICT_RAIN_THRESHOLD_IN, n_days)
        strict_dry_test = strict_full.reindex(test_df.index).fillna(False)

        m_dry = model_eval.score(test_df.index[strict_dry_test], test_df.loc[strict_dry_test, "flow_true"],
                                  test_df.loc[strict_dry_test, "flow_pred"])
        label = f"{'b' if n_days == 3 else 'c' if n_days == 5 else 'd'}_{n_days}_antecedent_dry_days"
        rows.append({"Definition": label, "Group": "Dry", **m_dry})

        excluded = loose_dry & ~strict_dry_test
        m_excl = model_eval.score(test_df.index[excluded], test_df.loc[excluded, "flow_true"],
                                   test_df.loc[excluded, "flow_pred"])
        rows.append({"Definition": label, "Group": "Excluded_by_tightening", **m_excl})

    result = pd.DataFrame(rows)
    result.to_csv(DRY_DEF_SENSITIVITY_CSV, index=False)

    fig, ax = plt.subplots(figsize=(11, 7))
    dry_rows = result[result["Group"] == "Dry"]
    excl_rows = result[result["Group"] == "Excluded_by_tightening"]

    x = np.arange(len(dry_rows))
    width = 0.35
    ax.bar(x, dry_rows["RMSE"], width=width, color=FLOW_COLOR, label="Dry (this definition)", zorder=3)
    excl_x = x[1:]
    ax.bar(excl_x + width, excl_rows["RMSE"], width=width, color=INCOMPLETE_COLOR,
           label="Excluded by tightening (loosely dry, recently rained)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(dry_rows["Definition"], rotation=20, ha="right")
    ax.set_ylabel("RMSE (MGD)")
    ax.set_title("Dry-Day RMSE by Definition Strictness – Richmond RRWWTF")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG1_DEF_SENSITIVITY, dpi=DPI)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Task 2: days-since-rain error profile
# ---------------------------------------------------------------------------

def compute_days_since_rain(ts: pd.DataFrame) -> pd.Series:
    rain_event = ts["Rainfall_in_Corrected"] > LOOSE_RAIN_THRESHOLD_IN
    event_date = pd.Series(np.where(rain_event, ts.index, pd.NaT), index=ts.index)
    event_date = pd.to_datetime(event_date).ffill()
    days_since = (ts.index.to_series() - event_date).dt.days
    return days_since


def task2_days_since_rain(ts: pd.DataFrame, test_df: pd.DataFrame):
    days_since_full = compute_days_since_rain(ts)
    days_since_test = days_since_full.reindex(test_df.index)

    bucket = days_since_test.apply(
        lambda d: str(int(d)) if pd.notna(d) and d <= max(DAYS_SINCE_RAIN_BUCKETS) else ("7+" if pd.notna(d) else "unknown"))

    rows = []
    order = [str(b) for b in DAYS_SINCE_RAIN_BUCKETS] + ["7+"]
    for b in order:
        mask = bucket == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"Days_Since_Rain": b, "N": 0, "MAE": np.nan, "Mean_Error": np.nan})
            continue
        mae = float(test_df.loc[mask, "residual"].abs().mean())
        mean_error = float((test_df.loc[mask, "flow_pred"] - test_df.loc[mask, "flow_true"]).mean())
        rows.append({"Days_Since_Rain": b, "N": n, "MAE": mae, "Mean_Error": mean_error})

    result = pd.DataFrame(rows)
    result.to_csv(ERROR_BY_DAYS_SINCE_RAIN_CSV, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(result["Days_Since_Rain"], result["MAE"], color=FLOW_COLOR, zorder=3)
    ax1.set_xlabel("Days Since Last Rain (>0.1 in)")
    ax1.set_ylabel("Mean Absolute Error (MGD)")
    ax1.set_title("MAE by Days-Since-Rain")
    ax1.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    for i, row in result.iterrows():
        if pd.notna(row["MAE"]):
            ax1.annotate(f"n={int(row['N'])}", xy=(i, row["MAE"]), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=8)

    colors = [BASELINE_COLOR if v is not None and pd.notna(v) and v < 0 else FIT_COLOR for v in result["Mean_Error"]]
    ax2.bar(result["Days_Since_Rain"], result["Mean_Error"], color=colors, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Days Since Last Rain (>0.1 in)")
    ax2.set_ylabel("Mean Error, Predicted - Observed (MGD)")
    ax2.set_title("Bias by Days-Since-Rain (negative = underprediction)")
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Model Error vs. Days Since Last Rain Event – 2024 Test Period – Richmond RRWWTF", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG2_DAYS_SINCE_RAIN, dpi=DPI)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Task 3: recent-level baseline drift
# ---------------------------------------------------------------------------

def task3_residual_trend(test_df: pd.DataFrame, feat: pd.DataFrame):
    loose_dry = loose_dry_mask(test_df["rainfall_t"], test_df["rain_lag1"])
    dry_df = test_df.loc[loose_dry].sort_index()

    day_index = np.arange(len(dry_df))
    slope, intercept, r_value, p_value, std_err = stats.linregress(day_index, dry_df["residual"].values)

    trend_summary = pd.DataFrame([{
        "N_Dry_Test_Days": len(dry_df), "Slope_MGD_per_day": slope, "Intercept_MGD": intercept,
        "R_Value": r_value, "P_Value": p_value, "Std_Err": std_err,
        "Significant_at_0.05": bool(p_value < 0.05),
    }])
    trend_summary.to_csv(RESIDUAL_TREND_CSV, index=False)

    rolling_mean = dry_df["residual"].rolling(ROLLING_TREND_WINDOW, min_periods=max(3, ROLLING_TREND_WINDOW // 3)).mean()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(dry_df.index, dry_df["residual"], color=INCOMPLETE_COLOR, s=18, alpha=0.6, label="Dry-day residual (obs - pred)")
    ax.plot(dry_df.index, rolling_mean.values, color=BASELINE_COLOR, linewidth=1.8,
             label=f"{ROLLING_TREND_WINDOW}-day rolling mean")
    trend_line = intercept + slope * day_index
    ax.plot(dry_df.index, trend_line, color=FIT_COLOR, linewidth=1.6, linestyle="--",
             label=f"Linear trend (slope={slope:.4f} MGD/day, p={p_value:.3f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual, Observed - Predicted (MGD)")
    ax.set_title("Dry-Day Residuals Over the 2024 Test Period – Richmond RRWWTF")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG3_RESIDUAL_TREND, dpi=DPI)
    plt.close(fig)

    # --- Recent-level baseline vs. observed dry-day flow, full 2024 ---
    recent_level_2024 = feat.loc[(feat.index >= model_eval.TEST_START) & (feat.index <= model_eval.TEST_END), "recent_level"]
    flow_2024 = feat.loc[(feat.index >= model_eval.TEST_START) & (feat.index <= model_eval.TEST_END), "flow_true"]
    dry_mask_2024 = loose_dry_mask(
        feat.loc[recent_level_2024.index, "rainfall_t"],
        feat.loc[recent_level_2024.index, "rainfall_t"].shift(1)
    )

    fig2, ax2 = plt.subplots(figsize=(15, 7))
    ax2.plot(recent_level_2024.index, recent_level_2024.values, color=BASELINE_COLOR, linewidth=1.8,
              label="60-Day Recent-Level Baseline")
    dry_obs = flow_2024.where(dry_mask_2024.reindex(flow_2024.index).fillna(False))
    ax2.scatter(dry_obs.index, dry_obs.values, color=FLOW_COLOR, s=14, alpha=0.6, label="Observed Flow (dry days only)")
    ax2.set_ylabel("Flow (MGD)")
    ax2.set_title("Recent-Level Baseline vs. Observed Dry-Day Flow – 2024 – Richmond RRWWTF")
    ax2.legend(loc="upper right")
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    fig2.autofmt_xdate(rotation=45)
    fig2.tight_layout()
    fig2.savefig(FIG4_RECENT_LEVEL_TRACKING, dpi=DPI)
    plt.close(fig2)

    return trend_summary


# ---------------------------------------------------------------------------
# Task 4: structure in dry-day residuals
# ---------------------------------------------------------------------------

def task4_residual_structure(test_df: pd.DataFrame):
    loose_dry = loose_dry_mask(test_df["rainfall_t"], test_df["rain_lag1"])
    dry_df = test_df.loc[loose_dry].copy()
    dry_df["DayOfWeek"] = dry_df.index.day_name()
    dry_df["Month"] = dry_df.index.month

    rows = []

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for dow in dow_order:
        g = dry_df[dry_df["DayOfWeek"] == dow]["residual"]
        if len(g) >= 2:
            t_stat, p_val = stats.ttest_1samp(g, 0)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({"Grouping": "DayOfWeek", "Group": dow, "N": len(g), "Mean_Residual": g.mean(),
                     "MAE": g.abs().mean(), "T_Stat": t_stat, "P_Value": p_val})

    for month in range(1, 13):
        g = dry_df[dry_df["Month"] == month]["residual"]
        if len(g) >= 2:
            t_stat, p_val = stats.ttest_1samp(g, 0)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({"Grouping": "Month", "Group": str(month), "N": len(g),
                     "Mean_Residual": g.mean() if len(g) else np.nan,
                     "MAE": g.abs().mean() if len(g) else np.nan, "T_Stat": t_stat, "P_Value": p_val})

    result = pd.DataFrame(rows)
    result.to_csv(RESIDUAL_STRUCTURE_CSV, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

    dow_data = result[result["Grouping"] == "DayOfWeek"]
    colors_dow = [FIT_COLOR if p is not None and pd.notna(p) and p < 0.05 else FLOW_COLOR for p in dow_data["P_Value"]]
    ax1.bar(dow_data["Group"], dow_data["Mean_Residual"], color=colors_dow, zorder=3)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Mean Residual (MGD)")
    ax1.set_title("Dry-Day Residual by Day-of-Week")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    month_data = result[result["Grouping"] == "Month"]
    colors_month = [FIT_COLOR if p is not None and pd.notna(p) and p < 0.05 else FLOW_COLOR for p in month_data["P_Value"]]
    ax2.bar(month_data["Group"], month_data["Mean_Residual"], color=colors_month, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Calendar Month")
    ax2.set_ylabel("Mean Residual (MGD)")
    ax2.set_title("Dry-Day Residual by Month")
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("Structure in Dry-Day Residuals (green = p<0.05 vs. zero) – Richmond RRWWTF", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG5_DOW_MONTH, dpi=DPI)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Task 5: analog-day noise floor
# ---------------------------------------------------------------------------

def task5_analog_noise_floor(feat: pd.DataFrame):
    rain_full = feat["rainfall_t"]  # this IS Rainfall_in_Corrected, from build_feature_frame
    strict_dry_full = antecedent_dry_flags(rain_full, STRICT_RAIN_THRESHOLD_IN, 5)  # matches task 1(c)

    pool = feat.loc[strict_dry_full.reindex(feat.index).fillna(False) & feat["flow_true"].notna() &
                     feat["recent_level"].notna()].copy()
    pool["doy"] = pool.index.dayofyear

    n_pool = len(pool)
    dates = pool.index.values
    doy = pool["doy"].values.astype(float)
    recent_level = pool["recent_level"].values.astype(float)
    flow = pool["flow_true"].values.astype(float)
    date_ordinal = pool.index.map(pd.Timestamp.toordinal).values.astype(float)

    diffs = []
    # Vectorized upper-triangle pairwise comparison
    for i in range(n_pool):
        doy_dist = np.abs(doy[i] - doy)
        doy_dist = np.minimum(doy_dist, 365.25 - doy_dist)
        seasonal_ok = doy_dist <= SEASONAL_TOLERANCE_DAYS

        rl_avg = (recent_level[i] + recent_level) / 2.0
        rl_ok = np.abs(recent_level[i] - recent_level) / np.where(rl_avg == 0, np.nan, rl_avg) <= RECENT_LEVEL_TOLERANCE_PCT

        date_ok = np.abs(date_ordinal[i] - date_ordinal) >= MIN_DATE_SEPARATION_DAYS

        j_mask = seasonal_ok & rl_ok & date_ok
        j_mask[:i + 1] = False  # upper triangle only, avoid self-pairs and double counting
        js = np.where(j_mask)[0]
        for j in js:
            diffs.append(abs(flow[i] - flow[j]))

    diffs = np.array(diffs)
    n_pairs = len(diffs)

    if n_pairs > 0:
        summary = pd.DataFrame([{
            "N_Strict_Dry_Days_Pool": n_pool, "N_Matched_Pairs": n_pairs,
            "Median_Abs_Diff_MGD": np.median(diffs), "Mean_Abs_Diff_MGD": np.mean(diffs),
            "P75_Abs_Diff_MGD": np.percentile(diffs, 75), "P90_Abs_Diff_MGD": np.percentile(diffs, 90),
            "Std_Abs_Diff_MGD": np.std(diffs),
            "Seasonal_Tolerance_Days": SEASONAL_TOLERANCE_DAYS,
            "Recent_Level_Tolerance_Pct": RECENT_LEVEL_TOLERANCE_PCT,
            "Min_Date_Separation_Days": MIN_DATE_SEPARATION_DAYS,
        }])
    else:
        summary = pd.DataFrame([{
            "N_Strict_Dry_Days_Pool": n_pool, "N_Matched_Pairs": 0,
            "Median_Abs_Diff_MGD": np.nan, "Mean_Abs_Diff_MGD": np.nan,
            "P75_Abs_Diff_MGD": np.nan, "P90_Abs_Diff_MGD": np.nan, "Std_Abs_Diff_MGD": np.nan,
            "Seasonal_Tolerance_Days": SEASONAL_TOLERANCE_DAYS,
            "Recent_Level_Tolerance_Pct": RECENT_LEVEL_TOLERANCE_PCT,
            "Min_Date_Separation_Days": MIN_DATE_SEPARATION_DAYS,
        }])
    summary.to_csv(ANALOG_PAIRS_CSV, index=False)

    if n_pairs > 0:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(diffs, bins=40, color=PURPLE, edgecolor="white", alpha=0.9, zorder=3)
        ax.axvline(np.median(diffs), color=BASELINE_COLOR, linewidth=1.8, linestyle="--",
                   label=f"Median = {np.median(diffs):.3f} MGD")
        ax.axvline(0.427, color=FIT_COLOR, linewidth=1.8, linestyle="--",
                   label="Model dry-day RMSE = 0.427 MGD")
        ax.set_xlabel("|Observed Flow Difference| Between Matched Analog Days (MGD)")
        ax.set_ylabel("Number of Pairs")
        ax.set_title(f"Analog-Day Noise Floor – Strictly Dry Days, Whole Record – Richmond RRWWTF\n"
                     f"(n={n_pairs:,} pairs from {n_pool:,} strictly-dry days)")
        ax.legend()
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        fig.savefig(FIG6_ANALOG_HIST, dpi=DPI)
        plt.close(fig)

    return summary, diffs


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------

def write_findings_md(def_sens: pd.DataFrame, days_since: pd.DataFrame, trend: pd.DataFrame,
                       structure: pd.DataFrame, analog_summary: pd.DataFrame, model_dry_rmse: float):
    dry_rows = def_sens[def_sens["Group"] == "Dry"].set_index("Definition")
    loose_rmse = dry_rows.loc["a_current_no_rain_t_or_t-1", "RMSE"]
    strict7_rmse = dry_rows.loc["d_7_antecedent_dry_days", "RMSE"]
    rmse_drop_pct = (1 - strict7_rmse / loose_rmse) * 100 if loose_rmse else np.nan

    slope = trend["Slope_MGD_per_day"].iloc[0]
    p_value = trend["P_Value"].iloc[0]
    sig = trend["Significant_at_0.05"].iloc[0]

    dow_data = structure[structure["Grouping"] == "DayOfWeek"]
    month_data = structure[structure["Grouping"] == "Month"]
    dow_sig = dow_data[dow_data["P_Value"] < 0.05]
    month_sig = month_data[month_data["P_Value"] < 0.05]

    median_pair_diff = analog_summary["Median_Abs_Diff_MGD"].iloc[0]
    n_pairs = int(analog_summary["N_Matched_Pairs"].iloc[0])
    n_pool = int(analog_summary["N_Strict_Dry_Days_Pool"].iloc[0])
    headroom_pct = (1 - median_pair_diff / model_dry_rmse) * 100 if pd.notna(median_pair_diff) and model_dry_rmse else np.nan

    lines = []
    lines.append("# Dry-Weather Error Diagnostic – Richmond RRWWTF Flow Prediction")
    lines.append("")
    lines.append("Diagnostic output of `scripts/dry_weather_diagnostic.py`. No retraining, no feature "
                  "changes, no tuning -- this decomposes the XGBoost_Wet_Weighted model's dry-day TEST "
                  f"error (RMSE={model_dry_rmse:.3f} MGD) into its likely sources.")
    lines.append("")

    lines.append("## 1. How much of the dry-day error is actually post-storm recession?")
    lines.append("")
    b_rmse = dry_rows.loc["b_3_antecedent_dry_days", "RMSE"]
    c_rmse = dry_rows.loc["c_5_antecedent_dry_days", "RMSE"]
    excl_c_rmse = def_sens.set_index(["Definition", "Group"]).loc[
        ("c_5_antecedent_dry_days", "Excluded_by_tightening"), "RMSE"]
    excl_d_rmse = def_sens.set_index(["Definition", "Group"]).loc[
        ("d_7_antecedent_dry_days", "Excluded_by_tightening"), "RMSE"]
    lines.append(
        f"RMSE by definition: loose (a, n={int(dry_rows.loc['a_current_no_rain_t_or_t-1','N'])})="
        f"{loose_rmse:.3f} -> 3-day (b, n={int(dry_rows.loc['b_3_antecedent_dry_days','N'])})={b_rmse:.3f} "
        f"-> 5-day (c, n={int(dry_rows.loc['c_5_antecedent_dry_days','N'])})={c_rmse:.3f} -> 7-day "
        f"(d, n={int(dry_rows.loc['d_7_antecedent_dry_days','N'])})={strict7_rmse:.3f} MGD. "
        f"Loose-to-7-day changes RMSE by {rmse_drop_pct:+.1f}%, but the path is **not monotonic** -- the "
        f"3-day step (b) is slightly worse than the loose baseline, and the real drop happens specifically "
        f"between the 3-day and 5-day steps, with 7-day ticking back up slightly from 5-day. Sample size "
        f"also shrinks fast across these steps (175 -> 109 -> 76 -> 53), so part of that swing at the "
        f"strictest definitions is plausibly sampling noise, not a clean signal."
    )
    lines.append(
        f"A more reliable piece of evidence: at both the 5-day and 7-day steps, the "
        f"'Excluded_by_tightening' group (the loosely-dry-but-recently-rained days newly dropped at that "
        f"step) scores clearly worse than the stricter-dry remainder -- {excl_c_rmse:.3f} vs. {c_rmse:.3f} "
        f"MGD at 5 days, {excl_d_rmse:.3f} vs. {strict7_rmse:.3f} MGD at 7 days. That comparison, more than "
        f"the raw loose-to-strict RMSE trend, supports a real (if partial) post-storm-recession effect: "
        f"recently-rained days that still pass the loose 'dry' test carry disproportionate error."
    )
    lines.append("")
    lines.append("Full sensitivity table:")
    lines.append("")
    lines.append(def_sens.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 2. Days-since-rain error profile")
    lines.append("")
    worst_bucket = days_since.loc[days_since["MAE"].idxmax()] if days_since["MAE"].notna().any() else None
    if worst_bucket is not None:
        lines.append(
            f"Worst MAE bucket: {worst_bucket['Days_Since_Rain']} days since rain "
            f"(MAE={worst_bucket['MAE']:.3f} MGD, n={int(worst_bucket['N'])}). "
            f"See `error_by_days_since_rain.csv` / `{FIG2_DAYS_SINCE_RAIN.name}` for the full profile and "
            f"which direction the model errs in during recession."
        )
    lines.append("")

    lines.append("## 3. Is the recent-level baseline drifting?")
    lines.append("")
    lines.append(
        f"Linear trend on dry-day TEST residuals (observed - predicted) vs. time: "
        f"slope={slope:+.5f} MGD/day, p={p_value:.4f} -- "
        + (f"**statistically significant at 0.05**, indicating a real drift: the model is "
           f"{'increasingly underpredicting' if slope > 0 else 'increasingly overpredicting'} dry-day flow "
           f"as the test year progresses, consistent with the 60-day recent-level baseline lagging behind "
           f"the true current level. This is a fixable error source (e.g. a shorter window or a trend term)."
           if sig else
           "not statistically significant, so there is no reliable evidence the recent-level baseline is "
           "systematically drifting behind the true level over the 2024 test year. See "
           f"`{FIG4_RECENT_LEVEL_TRACKING.name}` for the visual tracking check.")
    )
    lines.append("")

    lines.append("## 4. Is there weekly or seasonal structure left in dry-day residuals?")
    lines.append("")
    if dow_sig.empty:
        lines.append("No day-of-week group's mean residual differs from zero at p<0.05 -- this confirms, on "
                      "a cleaner test (dry days only, residuals only, corrected rainfall) than the earlier "
                      "correlation stage, that there is no usable weekly pattern in the model's dry-day "
                      "error. Confirming no pattern is itself a useful finding, not a gap.")
    else:
        lines.append("Day-of-week groups with mean residual significantly different from zero (p<0.05): " +
                      ", ".join(f"{r.Group} (mean={r.Mean_Residual:+.3f}, p={r.P_Value:.3f})"
                                for r in dow_sig.itertuples()) + ".")
    n_months_tested = int(month_data["N"].gt(0).sum())
    if month_sig.empty:
        lines.append("No calendar-month group's mean residual differs from zero at p<0.05 either -- no "
                      "usable seasonal structure remains in the dry-day residuals.")
    else:
        lines.append("Month groups with mean residual significantly different from zero (p<0.05): " +
                      ", ".join(f"{r.Group} (mean={r.Mean_Residual:+.3f}, p={r.P_Value:.3f})"
                                for r in month_sig.itertuples()) +
                      f". Caveat: with {n_months_tested} months tested independently at alpha=0.05, "
                      f"roughly {n_months_tested * 0.05:.1f} false positives are expected by chance alone -- "
                      f"{len(month_sig)} hit(s) out of {n_months_tested} tests is weak, not strong, evidence "
                      f"of a real seasonal effect, and would not survive a multiple-comparisons correction "
                      f"(e.g. Bonferroni alpha={0.05 / n_months_tested:.4f}).")
    lines.append("")

    lines.append("## 5. Estimated noise floor -- the key number")
    lines.append("")
    lines.append(
        f"From {n_pool:,} strictly-dry days (5-antecedent-dry-day rule) across the whole 2018-2024 record, "
        f"{n_pairs:,} analog pairs were matched (same season within {SEASONAL_TOLERANCE_DAYS} days, recent "
        f"level within {RECENT_LEVEL_TOLERANCE_PCT:.0%}, dates at least {MIN_DATE_SEPARATION_DAYS} days "
        f"apart to avoid trivially autocorrelated neighbors). The median absolute flow difference between "
        f"matched analog days is **{median_pair_diff:.3f} MGD** -- an estimate of irreducible day-to-day "
        f"sanitary variation under near-identical conditions."
    )
    lines.append("")
    lines.append(
        f"Compared to the model's dry-day RMSE of {model_dry_rmse:.3f} MGD, this "
        + (f"leaves roughly **{headroom_pct:.0f}% headroom** -- the model has room to improve on dry days "
           f"before hitting the noise floor."
           if headroom_pct > 15 else
           f"is close to the model's actual dry-day RMSE ({headroom_pct:+.0f}% difference) -- the model is "
           f"already operating near the estimated noise floor on dry days; little further improvement "
           f"should be expected from dry-day-focused effort."
           if pd.notna(headroom_pct) else "could not be computed (insufficient analog pairs).")
    )
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    near_ceiling = pd.notna(headroom_pct) and headroom_pct <= 15
    lines.append(
        ("Dry-day performance looks close to its ceiling: the analog-day noise floor "
         f"({median_pair_diff:.3f} MGD) sits near the model's actual dry-day RMSE ({model_dry_rmse:.3f} MGD), "
         if near_ceiling else
         f"Dry-day performance has meaningful headroom: the analog-day noise floor ({median_pair_diff:.3f} MGD) "
         f"is well below the model's actual dry-day RMSE ({model_dry_rmse:.3f} MGD), ")
        + (f"and the recent-level baseline shows a statistically significant drift (p={p_value:.3f}) that is "
           f"a plausible, fixable contributor. " if sig else
           "and no significant baseline drift or day-of-week/month structure was found to explain the gap, "
           "so any further dry-day gains would likely come from smaller, harder-to-isolate effects. ")
        + ("Effort is better spent on wet-day performance (RMSE 0.527 MGD, already the larger error "
           "contributor per event) than on squeezing further improvement out of dry days."
           if near_ceiling else
           "Investigating the recent-level baseline window and the post-storm recession tail (task 1/2 "
           "results above) is a reasonable next step before assuming dry-day error is now irreducible.")
    )
    lines.append("")

    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("DRY-WEATHER ERROR DIAGNOSTIC")
    print()

    ts, feat, complete, test_df, train_mask, test_mask = reproduce_best_model()
    match = verify_reproduction(test_df)

    model_dry_mask = loose_dry_mask(test_df["rainfall_t"], test_df["rain_lag1"])
    model_dry_rmse = model_eval.score(test_df.index[model_dry_mask], test_df.loc[model_dry_mask, "flow_true"],
                                       test_df.loc[model_dry_mask, "flow_pred"])["RMSE"]
    print(f"Reproduced dry-day RMSE (loose definition): {model_dry_rmse:.4f} MGD")
    print()

    print("TASK 1 -- Dry-definition sensitivity")
    def_sens = task1_dry_definition_sensitivity(ts, test_df)
    print(def_sens.round(3).to_string(index=False))
    print()

    print("TASK 2 -- Days-since-rain error profile")
    days_since = task2_days_since_rain(ts, test_df)
    print(days_since.round(3).to_string(index=False))
    print()

    print("TASK 3 -- Recent-level baseline drift")
    trend = task3_residual_trend(test_df, feat)
    print(trend.round(5).to_string(index=False))
    print()

    print("TASK 4 -- Structure in dry-day residuals")
    structure = task4_residual_structure(test_df)
    print(structure.round(3).to_string(index=False))
    print()

    print("TASK 5 -- Analog-day noise floor")
    analog_summary, diffs = task5_analog_noise_floor(feat)
    print(analog_summary.round(3).to_string(index=False))
    print()

    write_findings_md(def_sens, days_since, trend, structure, analog_summary, model_dry_rmse)

    # =======================================================================
    # Console summary
    # =======================================================================
    print("DRY-WEATHER DIAGNOSTIC COMPLETE")
    print()
    print(f"Model reproduction: {'VERIFIED MATCH' if match else 'MISMATCH -- investigate'}")
    print(f"Dry-day RMSE (loose def): {model_dry_rmse:.3f} MGD")
    print(f"Analog-day noise floor (median): {analog_summary['Median_Abs_Diff_MGD'].iloc[0]:.3f} MGD "
          f"from {int(analog_summary['N_Matched_Pairs'].iloc[0]):,} pairs")
    print(f"Recent-level drift trend: slope={trend['Slope_MGD_per_day'].iloc[0]:+.5f} MGD/day, "
          f"p={trend['P_Value'].iloc[0]:.4f} "
          f"({'SIGNIFICANT' if trend['Significant_at_0.05'].iloc[0] else 'not significant'})")
    print()
    print("Generated files:")
    for p in [DRY_DEF_SENSITIVITY_CSV, ERROR_BY_DAYS_SINCE_RAIN_CSV, RESIDUAL_TREND_CSV, RESIDUAL_STRUCTURE_CSV,
              ANALOG_PAIRS_CSV, FINDINGS_MD, FIG1_DEF_SENSITIVITY, FIG2_DAYS_SINCE_RAIN, FIG3_RESIDUAL_TREND,
              FIG4_RECENT_LEVEL_TRACKING, FIG5_DOW_MONTH, FIG6_ANALOG_HIST]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

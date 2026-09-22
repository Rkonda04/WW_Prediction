"""
RRWWTF Flow Prediction -- Validation Framework and Naive Baselines

Establishes the evaluation protocol and the performance floor every future
model must beat. NO ML modeling happens here -- this is the split
definition, four naive baselines, and the shared scoring applied to them.

DATA-ALIGNMENT PREREQUISITE: this script loads the CORRECTED master
dataset (Rainfall_in_Corrected), not the original Rainfall_in, per
scripts/correct_rainfall_alignment.py -- daily flow correlates far more
strongly with next-day-recorded plant rainfall (r=0.441-0.456) than
same-day (r=0.296-0.308), consistent with a one-day-late recording
offset in the plant's rainfall log. See that script's docstring for the
full QA reproduction. Every wet/dry-day classification below uses the
corrected rainfall; the dry-weather baseline itself (Expected_Baseline_MGD)
is reused unchanged from the existing RDII stage, since the rainfall
offset does not affect how that baseline was built.

SPLIT AND SCORING: imported from scripts/model_eval.py (TRAIN_START/
TRAIN_END/TEST_START/TEST_END, CV_FOLDS, score()) -- not redefined here --
so every future model script that imports the same module is evaluated
identically to the baselines below.

    1. Print exact row counts for TRAIN, TEST, and the 3 expanding-window
       CV folds, after gap removal.
    2. Score 4 naive baselines on the TEST set: persistence, the seasonal/
       DWF baseline, calendar-month climatology, and a persistence+
       baseline hybrid (carry yesterday's excess-over-baseline forward).
    3. Every baseline is scored 3 ways (all/dry/wet test days) plus a
       high-flow slice (test days above the train-period 90th percentile
       of flow), using the single shared model_eval.score() function.
    4. output/08_validation_baselines/baseline_scores.csv,
       validation_findings.md, and 2 figures (test-period small multiples,
       predicted-vs-observed scatter grid).

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_corrected_rainfall.csv
        (built by correct_rainfall_alignment.py; adds Rainfall_in_Corrected
        to every column already in richmond_master_with_dry_weather_baseline.csv)
    output/05_rdii_events/daily_excess_flow.csv
        Date, Total_Treated_MGD, Expected_Baseline_MGD, Excess_Flow_MGD,
        RDII_MGD, Row_Exists, Dry_5Day

NOTE ON OUTPUT DIRECTORY NUMBER: output/06_validation_baselines/ was
requested, but "06" is already used by output/06_correlation_analysis/
(the lag-structure companion script added earlier in this body of work),
and "05" is used by the existing output/05_rdii_events/. This script
therefore writes to output/08_validation_baselines/ (07 is the
correlation-matrix stage) to keep the existing numbered sequence intact.

Usage:
    python scripts/validation_framework.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model_eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

CORRECTED_MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_corrected_rainfall.csv"
DAILY_EXCESS_CSV = OUTPUT_DIR / "05_rdii_events" / "daily_excess_flow.csv"

ANALYSIS_DIR = OUTPUT_DIR / "08_validation_baselines"

BASELINE_SCORES_CSV = ANALYSIS_DIR / "baseline_scores.csv"
CV_FOLD_SUMMARY_CSV = ANALYSIS_DIR / "cv_fold_summary.csv"
FINDINGS_MD = ANALYSIS_DIR / "validation_findings.md"

FIG1_TIMESERIES = ANALYSIS_DIR / "01_test_period_timeseries.png"
FIG2_SCATTER = ANALYSIS_DIR / "02_predicted_vs_observed_scatter.png"

WET_DAY_RAIN_THRESHOLD_IN = 0.1
HIGH_FLOW_PERCENTILE = 0.90

BASELINE_ORDER = ["Persistence", "Seasonal_DWF_Baseline", "Climatology", "Persistence_Baseline_Hybrid"]
BASELINE_LABELS = {
    "Persistence": "Persistence (flow[t-1])",
    "Seasonal_DWF_Baseline": "Seasonal / DWF Baseline",
    "Climatology": "Climatology (train monthly mean)",
    "Persistence_Baseline_Hybrid": "Persistence + Baseline Hybrid",
}
BASELINE_WHAT_BEATING_IT_SHOWS = {
    "Persistence": "Beating persistence demonstrates a model captures more than pure day-to-day "
                   "carryover -- i.e. it uses real information, not just the assumption that tomorrow "
                   "looks like today.",
    "Seasonal_DWF_Baseline": "Beating the seasonal/DWF baseline demonstrates a model captures dynamics "
                              "beyond the pre-established monthly dry-weather pattern -- i.e. it is using "
                              "rainfall or event information, not just calendar position.",
    "Climatology": "Beating climatology demonstrates a model captures more than the plain average "
                   "seasonal cycle -- i.e. day-specific dynamics (rainfall, recent flow) actually matter.",
    "Persistence_Baseline_Hybrid": "Beating the persistence+baseline hybrid is the real bar: it "
                                    "demonstrates a model captures how excess-over-baseline flow actually "
                                    "evolves (rising, decaying, responding to new rain) rather than "
                                    "assuming yesterday's excess simply persists unchanged into today.",
}

BASELINE_COLORS = {
    "Persistence": "#9ca3af",
    "Seasonal_DWF_Baseline": "#c0392b",
    "Climatology": "#d97706",
    "Persistence_Baseline_Hybrid": "#27ae60",
}
FLOW_COLOR = "#2b6cb0"

DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Load + merge onto a single continuous calendar-day series
# ---------------------------------------------------------------------------

def load_ts() -> pd.DataFrame:
    corrected_master = pd.read_csv(CORRECTED_MASTER_CSV, parse_dates=["Date"])
    excess = pd.read_csv(DAILY_EXCESS_CSV, parse_dates=["Date"])

    merged = excess.merge(corrected_master[["Date", "Rainfall_in_Corrected", "Rainfall_in"]],
                           on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    ts = merged.set_index("Date")
    gaps = ts.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=1)], (
        "Expected one row per calendar day (gaps present as NaN, not missing rows)."
    )
    return ts


# ---------------------------------------------------------------------------
# Split / fold row-count reporting
# ---------------------------------------------------------------------------

def report_splits_and_folds(ts: pd.DataFrame):
    dates = ts.index.to_series().reset_index(drop=True)
    row_exists = ts["Row_Exists"].reset_index(drop=True)

    train_mask, test_mask = model_eval.get_train_test_masks(dates)

    print("SPLIT DEFINITION (fixed, chronological)")
    print(f"  TRAIN: {model_eval.TRAIN_START.date()} to {model_eval.TRAIN_END.date()}")
    print(f"  TEST:  {model_eval.TEST_START.date()} to {model_eval.TEST_END.date()} "
          f"(bounded by whatever's actually available)")
    print(f"  EXCLUDED: {model_eval.EXCLUDED_YEAR} -- {model_eval.EXCLUDED_YEAR_REASON}")
    print()
    print(f"  TRAIN calendar days: {int(train_mask.sum()):,}  |  with an actual plant record: "
          f"{int((train_mask & row_exists).sum()):,}")
    print(f"  TEST  calendar days: {int(test_mask.sum()):,}  |  with an actual plant record: "
          f"{int((test_mask & row_exists).sum()):,}")
    print()

    print("EXPANDING-WINDOW CV FOLDS (within TRAIN only, never touching TEST)")
    fold_rows = []
    for name, fold_train_mask, fold_val_mask in model_eval.get_cv_fold_masks(dates):
        train_n = int(fold_train_mask.sum())
        train_n_present = int((fold_train_mask & row_exists).sum())
        val_n = int(fold_val_mask.sum())
        val_n_present = int((fold_val_mask & row_exists).sum())
        print(f"  {name}: train calendar days={train_n:,} (present={train_n_present:,})  |  "
              f"val calendar days={val_n:,} (present={val_n_present:,})")
        fold_rows.append({"Fold": name, "Train_Calendar_Days": train_n, "Train_Days_Present": train_n_present,
                           "Val_Calendar_Days": val_n, "Val_Days_Present": val_n_present})
    print()

    cv_summary = pd.DataFrame(fold_rows)
    cv_summary.to_csv(CV_FOLD_SUMMARY_CSV, index=False)

    split_report = {
        "train_calendar_days": int(train_mask.sum()),
        "train_days_present": int((train_mask & row_exists).sum()),
        "test_calendar_days": int(test_mask.sum()),
        "test_days_present": int((test_mask & row_exists).sum()),
    }
    return train_mask.values, test_mask.values, cv_summary, split_report


# ---------------------------------------------------------------------------
# Baseline predictions
# ---------------------------------------------------------------------------

def build_climatology(ts: pd.DataFrame, train_mask: np.ndarray) -> pd.Series:
    train_sub = ts.loc[train_mask, ["Total_Treated_MGD"]].copy()
    train_sub["Month"] = train_sub.index.month
    monthly_mean = train_sub.groupby("Month")["Total_Treated_MGD"].mean()
    return pd.Series(ts.index.month, index=ts.index).map(monthly_mean)


def build_baseline_predictions(ts: pd.DataFrame, train_mask: np.ndarray) -> dict:
    return {
        "Persistence": ts["Total_Treated_MGD"].shift(1),
        "Seasonal_DWF_Baseline": ts["Expected_Baseline_MGD"],
        "Climatology": build_climatology(ts, train_mask),
        "Persistence_Baseline_Hybrid": ts["Expected_Baseline_MGD"] + ts["Excess_Flow_MGD"].shift(1),
    }


# ---------------------------------------------------------------------------
# Wet / dry day classification (corrected rainfall, t or t-1)
# ---------------------------------------------------------------------------

def build_wet_dry_masks(ts: pd.DataFrame):
    rain_t = ts["Rainfall_in_Corrected"]
    rain_t1 = ts["Rainfall_in_Corrected"].shift(1)
    is_wet_t = rain_t > WET_DAY_RAIN_THRESHOLD_IN
    is_wet_t1 = rain_t1 > WET_DAY_RAIN_THRESHOLD_IN
    wet_mask = is_wet_t.fillna(False) | is_wet_t1.fillna(False)
    known_mask = rain_t.notna() & rain_t1.notna()
    dry_mask = known_mask & ~wet_mask
    return wet_mask.values, dry_mask.values


# ---------------------------------------------------------------------------
# Scoring across baselines x subsets
# ---------------------------------------------------------------------------

def score_all_baselines(ts: pd.DataFrame, predictions: dict, test_mask: np.ndarray,
                         wet_mask: np.ndarray, dry_mask: np.ndarray,
                         high_flow_threshold: float) -> pd.DataFrame:
    dates = ts.index.to_series().reset_index(drop=True).values
    y_true_all = ts["Total_Treated_MGD"].reset_index(drop=True)

    subset_masks = {
        "All": test_mask,
        "Dry": test_mask & dry_mask,
        "Wet": test_mask & wet_mask,
    }

    rows = []
    for name in BASELINE_ORDER:
        pred_series = predictions[name].reset_index(drop=True)
        for subset_name, mask in subset_masks.items():
            hf = high_flow_threshold if subset_name == "All" else None
            metrics = model_eval.score(dates[mask], y_true_all[mask], pred_series[mask],
                                        high_flow_threshold=hf)
            row = {"Baseline": name, "Subset": subset_name}
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_test_period_timeseries(ts: pd.DataFrame, predictions: dict, test_mask: np.ndarray, path: Path):
    test_dates = ts.index[test_mask]
    start, end = test_dates.min(), test_dates.max()
    full_range = pd.date_range(start, end, freq="D")

    observed = ts["Total_Treated_MGD"].reindex(full_range)

    fig, axes = plt.subplots(len(BASELINE_ORDER), 1, figsize=(16, 4 * len(BASELINE_ORDER)), sharex=True)

    for ax, name in zip(axes, BASELINE_ORDER):
        pred = predictions[name].reindex(full_range)
        ax.plot(full_range, observed.values, color=FLOW_COLOR, linewidth=1.2, label="Observed Flow")
        ax.plot(full_range, pred.values, color=BASELINE_COLORS[name], linewidth=1.2, linestyle="--",
                 label=BASELINE_LABELS[name])
        ax.set_ylabel("Flow (MGD)")
        ax.set_title(BASELINE_LABELS[name], fontsize=12)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    fig.suptitle("2024 Test Period – Observed Flow vs. Each Baseline – Richmond RRWWTF", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_scatter_grid(ts: pd.DataFrame, predictions: dict, test_mask: np.ndarray, path: Path):
    observed = ts.loc[test_mask, "Total_Treated_MGD"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes = axes.flatten()

    for ax, name in zip(axes, BASELINE_ORDER):
        pred = predictions[name].loc[test_mask]
        paired = pd.concat([observed, pred], axis=1, keys=["obs", "pred"]).dropna()

        ax.scatter(paired["obs"], paired["pred"], color=BASELINE_COLORS[name], alpha=0.5, s=18, zorder=3)
        lims = [min(paired["obs"].min(), paired["pred"].min()), max(paired["obs"].max(), paired["pred"].max())]
        ax.plot(lims, lims, color="black", linewidth=1.0, linestyle="--", label="1:1 line")

        ax.set_xlabel("Observed Flow (MGD)")
        ax.set_ylabel("Predicted Flow (MGD)")
        ax.set_title(f"{BASELINE_LABELS[name]}\n(n={len(paired):,})", fontsize=11)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Predicted vs. Observed – 2024 Test Period, All Days – Richmond RRWWTF", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------

def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal pipe-table renderer (the `tabulate` package used by
    DataFrame.to_markdown() is not installed in this environment)."""
    headers = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

def write_findings_md(split_report, cv_summary, scores: pd.DataFrame, high_flow_threshold: float,
                       n_wet_test, n_dry_test, train_mean: float, test_mean: float):
    all_scores = scores[scores["Subset"] == "All"].set_index("Baseline")
    wet_scores = scores[scores["Subset"] == "Wet"].set_index("Baseline")
    dry_scores = scores[scores["Subset"] == "Dry"].set_index("Baseline")

    strongest_overall = all_scores["RMSE"].idxmin()
    strongest_wet = wet_scores["RMSE"].idxmin() if wet_scores["RMSE"].notna().any() else None

    lines = []
    lines.append("# Validation Findings – Richmond RRWWTF Flow Prediction")
    lines.append("")
    lines.append("Diagnostic/protocol output of `scripts/validation_framework.py`. This establishes the "
                  "evaluation protocol and naive-baseline performance floor -- no ML modeling is done here. "
                  "Every future model script must import `scripts/model_eval.py` and be scored against "
                  "these same baselines on this same TEST set.")
    lines.append("")

    lines.append("## Data-alignment prerequisite")
    lines.append("")
    lines.append("Rainfall/flow date misalignment was confirmed before this framework was built (flow "
                  "correlates r=0.441-0.456 with next-day-recorded rainfall vs. r=0.296-0.308 same-day) -- "
                  "see `scripts/correct_rainfall_alignment.py`. This framework loads "
                  "`richmond_master_with_corrected_rainfall.csv` (`Rainfall_in_Corrected`) for all wet/dry-day "
                  "classification. The dry-weather baseline itself is unaffected and reused unchanged from "
                  "the existing RDII stage.")
    lines.append("")

    lines.append("## Split protocol")
    lines.append("")
    lines.append(f"- **TRAIN**: {model_eval.TRAIN_START.date()} to {model_eval.TRAIN_END.date()} -- "
                  f"{split_report['train_calendar_days']:,} calendar days, "
                  f"{split_report['train_days_present']:,} with an actual plant record.")
    lines.append(f"- **TEST**: {model_eval.TEST_START.date()} to {model_eval.TEST_END.date()} (bounded by "
                  f"availability) -- {split_report['test_calendar_days']:,} calendar days, "
                  f"{split_report['test_days_present']:,} with an actual plant record "
                  f"({n_dry_test:,} classified dry, {n_wet_test:,} classified wet).")
    lines.append(f"- **EXCLUDED**: {model_eval.EXCLUDED_YEAR} entirely. {model_eval.EXCLUDED_YEAR_REASON}")
    lines.append("")
    lines.append("Expanding-window CV folds (training period only, never touching TEST):")
    lines.append("")
    lines.append(df_to_markdown_table(cv_summary))
    lines.append("")

    lines.append("## Why every baseline has negative R2")
    lines.append("")
    lines.append(
        f"Train-period mean flow (2018-2022) is {train_mean:.3f} MGD; 2024 test-period mean flow is "
        f"{test_mean:.3f} MGD -- a {(test_mean / train_mean - 1):+.1%} level shift, consistent with a "
        f"visible upward trend already present within the training years themselves "
        f"(2018 annual mean 1.51 MGD -> 2022 annual mean 1.70 MGD). R2 is computed against the TEST set's "
        f"own mean, so any baseline calibrated to the (lower) train-period level will systematically "
        f"underpredict 2024 and score worse than simply guessing the test mean every day -- hence negative "
        f"R2 across the board. This is why `Persistence` and the hybrid (both of which use an actual t-1 "
        f"*observation* from the current regime, not a train-period average) come out ahead of "
        f"`Seasonal_DWF_Baseline` and `Climatology` (both calibrated to the stale train-period level, and "
        f"both show a substantially larger negative Mean_Error as a result). A model that re-estimates its "
        f"baseline level from recent data, rather than a fixed train-period average, should close most of "
        f"this gap on its own."
    )
    lines.append("")

    lines.append("## Strongest baseline")
    lines.append("")
    lines.append(f"- **Overall (all TEST days)**: `{strongest_overall}`, RMSE={all_scores.loc[strongest_overall, 'RMSE']:.3f} "
                  f"MGD, R2={all_scores.loc[strongest_overall, 'R2']:.3f}, n={int(all_scores.loc[strongest_overall, 'N']):,}.")
    if strongest_wet:
        lines.append(f"- **Wet days**: `{strongest_wet}`, RMSE={wet_scores.loc[strongest_wet, 'RMSE']:.3f} MGD, "
                      f"R2={wet_scores.loc[strongest_wet, 'R2']:.3f}, n={int(wet_scores.loc[strongest_wet, 'N']):,}.")
    else:
        lines.append("- **Wet days**: insufficient wet-day test observations to rank.")
    lines.append(f"- High-flow threshold (train-period {HIGH_FLOW_PERCENTILE:.0%} of flow): "
                  f"{high_flow_threshold:.3f} MGD.")
    for name in BASELINE_ORDER:
        hf_n = all_scores.loc[name, "High_Flow_N"]
        hf_rmse = all_scores.loc[name, "High_Flow_RMSE"]
        hf_me = all_scores.loc[name, "High_Flow_Mean_Error"]
        lines.append(f"  - `{name}` on high-flow days (n={int(hf_n) if pd.notna(hf_n) else 0}): "
                      f"RMSE={hf_rmse:.3f} MGD, mean error={hf_me:+.3f} MGD.")
    lines.append("")

    lines.append("## Exact scores (all baselines x subsets)")
    lines.append("")
    display_cols = ["Baseline", "Subset", "N", "N_Excluded_NaN", "R2", "RMSE", "MAE", "MAPE",
                     "Mean_Error", "Pct_Days_Underpredicted"]
    lines.append(df_to_markdown_table(scores[display_cols].round(3)))
    lines.append("")

    lines.append("## What beating each baseline would demonstrate")
    lines.append("")
    for name in BASELINE_ORDER:
        lines.append(f"- **{BASELINE_LABELS[name]}**: {BASELINE_WHAT_BEATING_IT_SHOWS[name]}")
    lines.append("")

    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ts = load_ts()

    print("VALIDATION FRAMEWORK AND NAIVE BASELINES")
    print()
    print(f"Continuous calendar-day series: {ts.index.min().date()} to {ts.index.max().date()} "
          f"({len(ts):,} calendar days)")
    print()

    train_mask, test_mask, cv_summary, split_report = report_splits_and_folds(ts)

    predictions = build_baseline_predictions(ts, train_mask)
    wet_mask, dry_mask = build_wet_dry_masks(ts)

    n_wet_test = int((test_mask & wet_mask & ts["Row_Exists"].values).sum())
    n_dry_test = int((test_mask & dry_mask & ts["Row_Exists"].values).sum())
    print(f"TEST-period wet/dry classification (corrected rainfall, t or t-1 > "
          f"{WET_DAY_RAIN_THRESHOLD_IN} in): {n_wet_test:,} wet, {n_dry_test:,} dry, "
          f"{split_report['test_days_present'] - n_wet_test - n_dry_test:,} unclassified (missing rainfall)")
    print()

    high_flow_threshold = float(ts.loc[train_mask, "Total_Treated_MGD"].quantile(HIGH_FLOW_PERCENTILE))
    print(f"High-flow threshold (train-period {HIGH_FLOW_PERCENTILE:.0%} of flow): "
          f"{high_flow_threshold:.3f} MGD")
    print()

    scores = score_all_baselines(ts, predictions, test_mask, wet_mask, dry_mask, high_flow_threshold)
    scores.to_csv(BASELINE_SCORES_CSV, index=False)

    plot_test_period_timeseries(ts, predictions, test_mask, FIG1_TIMESERIES)
    plot_scatter_grid(ts, predictions, test_mask, FIG2_SCATTER)

    train_mean = float(ts.loc[train_mask, "Total_Treated_MGD"].mean())
    test_mean = float(ts.loc[test_mask, "Total_Treated_MGD"].mean())
    print(f"Train-period mean flow: {train_mean:.3f} MGD  |  Test-period mean flow: {test_mean:.3f} MGD  "
          f"({(test_mean / train_mean - 1):+.1%} level shift -- explains the negative R2 below)")
    print()

    write_findings_md(split_report, cv_summary, scores, high_flow_threshold, n_wet_test, n_dry_test,
                       train_mean, test_mean)

    # =======================================================================
    # Console summary
    # =======================================================================
    print("BASELINE SCORES -- ALL TEST DAYS")
    all_scores = scores[scores["Subset"] == "All"].set_index("Baseline")
    print(all_scores[["N", "N_Excluded_NaN", "R2", "RMSE", "MAE", "MAPE", "Mean_Error",
                       "Pct_Days_Underpredicted"]].round(3).to_string())
    print()

    print("BASELINE SCORES -- WET DAYS ONLY")
    wet_scores = scores[scores["Subset"] == "Wet"].set_index("Baseline")
    print(wet_scores[["N", "R2", "RMSE", "MAE", "Mean_Error"]].round(3).to_string())
    print()

    print("BASELINE SCORES -- DRY DAYS ONLY")
    dry_scores = scores[scores["Subset"] == "Dry"].set_index("Baseline")
    print(dry_scores[["N", "R2", "RMSE", "MAE", "Mean_Error"]].round(3).to_string())
    print()

    strongest_overall = all_scores["RMSE"].idxmin()
    strongest_wet = wet_scores["RMSE"].idxmin() if wet_scores["RMSE"].notna().any() else "N/A"
    print(f"Strongest baseline overall (lowest RMSE, all test days): {strongest_overall} "
          f"(RMSE={all_scores.loc[strongest_overall, 'RMSE']:.3f} MGD)")
    print(f"Strongest baseline on wet days (lowest RMSE): {strongest_wet}")
    print()

    print("VALIDATION FRAMEWORK COMPLETE")
    print()
    print("Generated files:")
    for p in [BASELINE_SCORES_CSV, CV_FOLD_SUMMARY_CSV, FINDINGS_MD, FIG1_TIMESERIES, FIG2_SCATTER]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

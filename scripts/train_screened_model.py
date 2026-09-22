"""
RRWWTF Screened Decomposition Flow Model (Phase 5)

Builds on Phase 4 (train_decomposition_model.py) with three changes motivated
by output/12_r2_ceiling_diagnostic/r2_ceiling_findings.md:

  1. RAINFALL: daily rain = max(plant gauge corrected, Open-Meteo). The plant
     gauge under-reads Open-Meteo by 30-40% every full year and recorded
     0.00 in on 2024-07-08 (Hurricane Beryl, Open-Meteo 4.36 in).
  2. FREEZE: Open-Meteo daily minimum temperature and a freeze flag
     (Tmin < 28 F on t or t-1). Flow rises ~60% with no rain during hard
     freezes (Feb 2021, Dec 2022, Jan 2024) -- a real, learnable signal.
  3. NO-RAIN SPIKE SCREEN: a day is set to NaN (never interpolated) when
     flow > 1.5 x the 60-day dry-weather recent level AND no rain > 0.1 in
     on t, t-1, t-2 in EITHER gauge AND no freeze on t or t-1. This extends
     the Phase-1 Implausible_Spike_No_Rain rule (which uses a neighbor-day
     ratio and therefore misses multi-day plateaus) with the same physical
     logic: sanitary flow does not double without rain or a freeze. The
     rule is applied uniformly to every year, train and test alike, and
     every score is reported both WITH the screen (model's own protocol)
     and WITHOUT it (flagged days put back), so nothing is hidden.

Model selection is on the three expanding-window CV folds from
model_eval.py; the 2024 TEST set is scored once. Also reports pooled
out-of-sample metrics (CV validation years 2020/2021/2022 + TEST 2024),
each year predicted by a model that never saw it, and tolerance-band
accuracy (1 - MAPE, % of days within +/-10/20/30%) alongside R2/RMSE.

Inputs (read-only):
    output/09_data_screening/richmond_daily_cleaned.csv
    output/12_r2_ceiling_diagnostic/richmond_openmeteo_daily_extended.csv
    scripts/model_eval.py, scripts/recent_level_baseline.py (imported)

Usage:
    python scripts/train_screened_model.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

import model_eval
import recent_level_baseline as rlb
import train_decomposition_model as p4

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
CLEANED_CSV = OUTPUT_DIR / "09_data_screening" / "richmond_daily_cleaned.csv"
OPENMETEO_CSV = OUTPUT_DIR / "12_r2_ceiling_diagnostic" / "richmond_openmeteo_daily_extended.csv"
PLANT_LOG_CSV = OUTPUT_DIR / "12_r2_ceiling_diagnostic" / "plant_log_totalizer_elapsed.csv"

ANALYSIS_DIR = OUTPUT_DIR / "13_screened_model"
SCREENED_DAYS_CSV = ANALYSIS_DIR / "screened_days.csv"
CV_RESULTS_CSV = ANALYSIS_DIR / "cv_selection.csv"
TEST_SCORES_CSV = ANALYSIS_DIR / "test_scores.csv"
OOS_SCORES_CSV = ANALYSIS_DIR / "pooled_out_of_sample_scores.csv"
OOS_PREDICTIONS_CSV = ANALYSIS_DIR / "out_of_sample_predictions_all_years.csv"
TEST_PREDICTIONS_CSV = ANALYSIS_DIR / "test_predictions_2024.csv"
FEATURE_IMPORTANCE_CSV = ANALYSIS_DIR / "feature_importance.csv"
FINDINGS_MD = ANALYSIS_DIR / "model_findings.md"
FIG1 = ANALYSIS_DIR / "01_test_period_overlay.png"
FIG2 = ANALYSIS_DIR / "02_scatter.png"
FIG3 = ANALYSIS_DIR / "03_feature_importance.png"

WET = rlb.WET_DAY_RAIN_THRESHOLD_IN
SPIKE_K = 1.5
NO_RAIN_IN = 0.1
FREEZE_F = 28.0
LEVEL_WINDOW = 60
WET_DAY_SAMPLE_WEIGHT = 3.0
HIGH_FLOW_PERCENTILE = 0.90

# The plant log's "Total Treated MGD" is the raw totalizer difference between two manual meter
# readings, which are taken whenever an operator gets to them (06:43-10:16 observed). The interval
# therefore spans 21.4-26.2 h and 43.6% of days fall outside 23-25 h, putting several percent of
# spurious variation into the target. NORMALIZE_TO_24H rescales each day by 24/elapsed so the
# target is a true 24-hour volume -- the correct basis for capacity planning. Elapsed values
# outside ELAPSED_VALID_RANGE are log errors, not real intervals, and are left uncorrected.
NORMALIZE_TO_24H = True
ELAPSED_VALID_RANGE = (12.0, 36.0)
TOLERANCES = (0.10, 0.20, 0.30)
# Operational "counted correct" band: predicted within -0.2 / +0.3 MGD of observed (e.g. 1.7 observed -> 1.5 to 2.0 is correct)
TOL_BAND_LOWER_MGD = 0.2
TOL_BAND_UPPER_MGD = 0.3

FEATURE_COLUMNS = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_30d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
]

XGB_GRID = [
    dict(max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3),
    dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=5),
    dict(max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=3),
]
XGB_EARLY_STOPPING_ROUNDS = 50
MODEL_NAME = "XGBoost_Screened"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_inputs():
    ts = rlb.load_ts()
    om = pd.read_csv(OPENMETEO_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts.index)
    rain = pd.Series(np.fmax(ts["Rainfall_in_Corrected"].fillna(0), om["precipitation_sum"].fillna(0)), index=ts.index)
    if NORMALIZE_TO_24H:
        ts = apply_24h_normalization(ts)
    return ts, om, rain


def apply_24h_normalization(ts: pd.DataFrame) -> pd.DataFrame:
    """Rescale each day's totalizer-difference flow to a true 24-hour volume."""
    elapsed = pd.read_csv(PLANT_LOG_CSV, parse_dates=["Date"]).set_index("Date")["elapsed"].reindex(ts.index)
    usable = elapsed.between(*ELAPSED_VALID_RANGE)
    factor = (24.0 / elapsed).where(usable, 1.0)
    out = ts.copy()
    out["Total_Treated_MGD"] = out["Total_Treated_MGD"] * factor
    out.attrs["n_normalized"] = int(usable.sum())
    out.attrs["n_uncorrected"] = int((~usable & elapsed.notna()).sum())
    return out


def apply_screen(ts, om, rain):
    level, _ = rlb.build_recent_level(ts.assign(Rainfall_in_Corrected=rain))
    tmin = om["temperature_2m_min"]
    dry3 = rain.rolling(3, min_periods=3).max() <= NO_RAIN_IN
    nofreeze = (tmin >= FREEZE_F) & (tmin.shift(1) >= FREEZE_F)
    flow = ts["Total_Treated_MGD"]
    flagged = dry3 & nofreeze & (flow > SPIKE_K * level)

    detail = pd.DataFrame({
        "Value_MGD": flow, "Recent_Level_MGD": level, "Ratio_To_Recent_Level": flow / level,
        "Plant_Rain_3d_Max": ts["Rainfall_in_Corrected"].rolling(3, min_periods=1).max(),
        "OpenMeteo_Rain_3d_Max": om["precipitation_sum"].rolling(3, min_periods=1).max(),
        "Tmin_F": tmin,
    })[flagged]

    screened = ts.copy()
    screened["Rainfall_in_Corrected"] = rain
    screened.loc[flagged, "Total_Treated_MGD"] = np.nan
    return screened, flagged, detail


def build_feature_frame(ts, om, flow_unscreened):
    flow, r = ts["Total_Treated_MGD"], ts["Rainfall_in_Corrected"]
    dry = flow.where((r <= WET).fillna(False))
    level = dry.rolling(LEVEL_WINDOW, min_periods=rlb.MIN_VALID_DRY_DAYS).median().shift(1).ffill()
    excess = flow - level
    tmin = om["temperature_2m_min"]

    f = pd.DataFrame(index=ts.index)
    f["flow_lag1"] = flow.shift(1)
    f["excess_lag1"] = excess.shift(1)
    f["excess_lag2"] = excess.shift(2)
    f["excess_lag3"] = excess.shift(3)
    f["excess_mean3"] = excess.shift(1).rolling(3, min_periods=2).mean()
    f["rainfall_t"] = r
    f["rain_lag1"] = r.shift(1)
    f["rain_lag2"] = r.shift(2)
    f["rain_sqrt_t"] = np.sqrt(r.clip(lower=0))
    f["ante_5d"] = r.rolling(5, min_periods=5).sum().shift(1)
    f["ante_30d"] = r.rolling(30, min_periods=30).sum().shift(1)
    wet = (r > WET).fillna(False).values
    dsr, c = np.zeros(len(wet)), 30
    for i, w in enumerate(wet):
        c = 0 if w else min(c + 1, 30)
        dsr[i] = c
    f["days_since_rain"] = pd.Series(dsr, index=ts.index).shift(1)
    f["recent_level"] = level
    f["level_slope30"] = level - level.shift(30)
    f["tmin_t"] = tmin
    f["tmin_lag1"] = tmin.shift(1)
    f["freeze_2d"] = ((tmin < FREEZE_F) | (tmin.shift(1) < FREEZE_F)).astype(float)
    doy = ts.index.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    f["target_excess"] = excess
    f["flow_true"] = flow
    f["flow_unscreened"] = flow_unscreened
    f["rain_wet_mask_t"] = r > WET
    f["rain_wet_mask_t1"] = r.shift(1) > WET
    return f


def drop_incomplete(feat, require_target=True):
    req = FEATURE_COLUMNS + ["recent_level", "flow_true"] + (["target_excess"] if require_target else [])
    mask = feat[req].isna().any(axis=1)
    return feat.loc[~mask].copy(), int(mask.sum())


# ---------------------------------------------------------------------------
# Fitting / scoring
# ---------------------------------------------------------------------------

def wet_weights(complete, mask):
    wet = (complete.loc[mask, "rain_wet_mask_t"] | complete.loc[mask, "rain_wet_mask_t1"]).values
    return np.where(wet, WET_DAY_SAMPLE_WEIGHT, 1.0)


def fit_xgb(complete, mask, params, n_estimators):
    m = xgb.XGBRegressor(**params, n_estimators=n_estimators, objective="reg:squarederror", random_state=42)
    m.fit(complete.loc[mask, FEATURE_COLUMNS].values, complete.loc[mask, "target_excess"].values,
          sample_weight=wet_weights(complete, mask))
    return m


def cv_select(complete, dates):
    """Average early-stopped RMSE (on reconstructed flow) across the 3 folds for each grid entry."""
    rows = []
    for gi, params in enumerate(XGB_GRID):
        rmses, ns = [], []
        for name, tm, vm in model_eval.get_cv_fold_masks(dates):
            tm, vm = tm.values, vm.values
            m = xgb.XGBRegressor(**params, n_estimators=1500, objective="reg:squarederror", random_state=42,
                                 early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
            m.fit(complete.loc[tm, FEATURE_COLUMNS].values, complete.loc[tm, "target_excess"].values,
                  sample_weight=wet_weights(complete, tm),
                  eval_set=[(complete.loc[vm, FEATURE_COLUMNS].values, complete.loc[vm, "target_excess"].values)], verbose=False)
            n = m.best_iteration + 1
            pred = complete.loc[vm, "recent_level"].values + m.predict(complete.loc[vm, FEATURE_COLUMNS].values, iteration_range=(0, n))
            rmses.append(model_eval.score(complete.loc[vm].index, complete.loc[vm, "flow_true"], pred)["RMSE"])
            ns.append(n)
        rows.append({"Grid_Index": gi, **params, "Avg_CV_RMSE": float(np.mean(rmses)), "N_Estimators": int(np.mean(ns)),
                     **{f"Fold{i+1}_RMSE": r for i, r in enumerate(rmses)}})
        print(f"  grid {gi}: {params}  avg CV RMSE={np.mean(rmses):.4f}  n={int(np.mean(ns))}")
    res = pd.DataFrame(rows)
    best = res.loc[res["Avg_CV_RMSE"].idxmin()]
    return XGB_GRID[int(best["Grid_Index"])], int(best["N_Estimators"]), res


def full_metrics(dates, y, p, high_flow_threshold=None):
    s = model_eval.score(dates, y, p, high_flow_threshold=high_flow_threshold)
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    ape = np.abs(p[ok] - y[ok]) / y[ok]
    s["Accuracy_1_minus_MAPE_Pct"] = float(100 * (1 - ape.mean()))
    for tol in TOLERANCES:
        s[f"Pct_Days_Within_{int(tol*100)}Pct"] = float(100 * (ape <= tol).mean())
    err = p[ok] - y[ok]
    s["Pct_Days_Correct_Band"] = float(100 * ((err >= -TOL_BAND_LOWER_MGD) & (err <= TOL_BAND_UPPER_MGD)).mean())
    s["Pct_Days_Within_0.3_MGD"] = float(100 * (np.abs(err) <= 0.3).mean())
    return s


def score_subsets(df, pred, high_flow_threshold):
    wet = (df["rain_wet_mask_t"] | df["rain_wet_mask_t1"]).values
    rows = []
    for name, mask in {"All": np.ones(len(df), bool), "Dry": ~wet, "Wet": wet}.items():
        rows.append({"Subset": name, **full_metrics(df.index[mask], df["flow_true"].values[mask], pred[mask],
                                                    high_flow_threshold if name == "All" else None)})
    return pd.DataFrame(rows)


def pooled_out_of_sample(complete, dates, params, n_estimators, train_mask, test_mask):
    """Every year predicted by a model that never saw it: 3 CV validation years + TEST."""
    parts = []
    for name, tm, vm in model_eval.get_cv_fold_masks(dates):
        tm, vm = tm.values, vm.values
        m = fit_xgb(complete, tm, params, n_estimators)
        p = complete.loc[vm, "recent_level"].values + m.predict(complete.loc[vm, FEATURE_COLUMNS].values)
        parts.append(pd.DataFrame({"Period": f"{name} val {complete.index[vm].year.min()}", "y": complete.loc[vm, "flow_true"].values, "p": p}, index=complete.index[vm]))
    m = fit_xgb(complete, train_mask, params, n_estimators)
    p = complete.loc[test_mask, "recent_level"].values + m.predict(complete.loc[test_mask, FEATURE_COLUMNS].values)
    parts.append(pd.DataFrame({"Period": "TEST 2024", "y": complete.loc[test_mask, "flow_true"].values, "p": p}, index=complete.index[test_mask]))
    allp = pd.concat(parts)
    allp["persist"] = complete.loc[allp.index, "flow_lag1"].values
    allp.to_csv(OOS_PREDICTIONS_CSV, index_label="Date")
    rows = [{"Period": per, **full_metrics(g.index, g.y, g.p)} for per, g in allp.groupby("Period", sort=False)]
    rows.append({"Period": "POOLED", **full_metrics(allp.index, allp.y, allp.p)})
    rows.append({"Period": "POOLED persistence baseline (flow[t-1])", **full_metrics(allp.index, allp.y, allp.persist)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print("TRAIN SCREENED DECOMPOSITION MODEL (PHASE 5)\n")

    ts0, om, rain = load_inputs()
    if NORMALIZE_TO_24H:
        print(f"24-hour normalization: {ts0.attrs['n_normalized']:,} days rescaled by 24/elapsed; "
              f"{ts0.attrs['n_uncorrected']} days had an implausible elapsed-time entry and were left as logged.")
        print()
    ts, flagged, detail = apply_screen(ts0, om, rain)
    detail.to_csv(SCREENED_DAYS_CSV, index_label="Date")
    by_year = flagged.groupby(ts.index.year).sum().to_dict()
    print(f"No-rain/no-freeze spike screen: {int(flagged.sum())} days set to NaN, by year {by_year}")
    print("  " + ", ".join(d.strftime("%Y-%m-%d") for d in detail.index) + "\n")

    feat = build_feature_frame(ts, om, ts0["Total_Treated_MGD"])
    complete, n_dropped = drop_incomplete(feat)
    dates = complete.index.to_series().reset_index(drop=True)
    train_mask, test_mask = model_eval.get_train_test_masks(dates)
    train_mask, test_mask = train_mask.values, test_mask.values
    print(f"Feature frame: {len(feat):,} days -> {len(complete):,} complete rows ({n_dropped:,} dropped, no fill)")
    print(f"TRAIN rows: {int(train_mask.sum()):,}  |  TEST rows: {int(test_mask.sum()):,}\n")
    high_flow_threshold = float(complete.loc[train_mask, "flow_true"].quantile(HIGH_FLOW_PERCENTILE))

    print("XGBOOST GRID SELECTION (avg early-stopped RMSE over 3 expanding-window CV folds)")
    params, n_estimators, cv_res = cv_select(complete, dates)
    cv_res.to_csv(CV_RESULTS_CSV, index=False)
    print(f"  Selected: {params}, n_estimators={n_estimators}\n")

    model = fit_xgb(complete, train_mask, params, n_estimators)
    test_df = complete.loc[test_mask]
    flow_pred = test_df["recent_level"].values + model.predict(test_df[FEATURE_COLUMNS].values)
    scores = score_subsets(test_df, flow_pred, high_flow_threshold)
    scores.insert(0, "Protocol", "Screened")

    # Same model, flagged 2024 days put back (features need lag context; flagged days themselves are scored)
    feat_u = feat.copy()
    feat_u["flow_true"] = feat_u["flow_unscreened"]
    complete_u, _ = drop_incomplete(feat_u, require_target=False)
    dates_u = complete_u.index.to_series().reset_index(drop=True)
    _, test_mask_u = model_eval.get_train_test_masks(dates_u)
    test_u = complete_u.loc[test_mask_u.values]
    pred_u = test_u["recent_level"].values + model.predict(test_u[FEATURE_COLUMNS].values)
    scores_u = score_subsets(test_u, pred_u, high_flow_threshold)
    scores_u.insert(0, "Protocol", "Unscreened_FlaggedDaysIncluded")

    all_scores = pd.concat([scores, scores_u], ignore_index=True)
    all_scores.insert(0, "Model", MODEL_NAME)
    all_scores.to_csv(TEST_SCORES_CSV, index=False)
    cols_show = ["Protocol", "Subset", "N", "R2", "RMSE", "MAE", "Accuracy_1_minus_MAPE_Pct", "Pct_Days_Within_20Pct"]
    print("TEST 2024 SCORES")
    print(all_scores[cols_show].round(3).to_string(index=False) + "\n")

    oos = pooled_out_of_sample(complete, dates, params, n_estimators, train_mask, test_mask)
    oos.to_csv(OOS_SCORES_CSV, index=False)
    print("POOLED OUT-OF-SAMPLE (screened protocol)")
    print(oos[["Period", "N", "R2", "RMSE", "MAE", "Accuracy_1_minus_MAPE_Pct", "Pct_Days_Correct_Band",
               "Pct_Days_Within_20Pct", "Pct_Days_Within_30Pct"]].round(3).to_string(index=False) + "\n")

    pd.DataFrame({"Date": test_u.index, "Observed_MGD": test_u["flow_true"].values, "Predicted_MGD": pred_u,
                  "Screened_Out": flagged.reindex(test_u.index).values,
                  "Rain_in": test_u["rainfall_t"].values, "Tmin_F": test_u["tmin_t"].values}).to_csv(TEST_PREDICTIONS_CSV, index=False)

    importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": model.feature_importances_}) \
        .sort_values("Importance", ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    hybrid = feat["recent_level"] + feat["flow_true"].shift(1) - feat["recent_level"].shift(1)
    p4.MODEL_COLORS[MODEL_NAME] = "#c0392b"
    p4.plot_test_overlay(complete, test_mask, flow_pred, MODEL_NAME, hybrid_pred=hybrid.loc[test_df.index], path=FIG1)
    p4.plot_scatter(complete, test_mask, flow_pred, MODEL_NAME, FIG2)
    p4.plot_feature_importance(importance, MODEL_NAME, FIG3)

    write_findings(all_scores, oos, detail, by_year, params, n_estimators, importance, len(complete), n_dropped)
    print("PHASE 5 COMPLETE\nGenerated files:")
    for p in [SCREENED_DAYS_CSV, CV_RESULTS_CSV, TEST_SCORES_CSV, OOS_SCORES_CSV, OOS_PREDICTIONS_CSV, TEST_PREDICTIONS_CSV,
              FEATURE_IMPORTANCE_CSV, FINDINGS_MD, FIG1, FIG2, FIG3]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


def write_findings(all_scores, oos, detail, by_year, params, n_estimators, importance, n_rows, n_dropped):
    def row(protocol, subset):
        return all_scores[(all_scores["Protocol"] == protocol) & (all_scores["Subset"] == subset)].iloc[0]

    s_all, s_wet, s_dry = row("Screened", "All"), row("Screened", "Wet"), row("Screened", "Dry")
    u_all = row("Unscreened_FlaggedDaysIncluded", "All")
    pooled = oos[oos["Period"] == "POOLED"].iloc[0]
    p4_r2, p4_rmse = 0.370, 0.454  # output/10_decomp_model/model_findings.md

    def acc_table(df, label_col):
        lines = ["| " + label_col + " | N | R2 | RMSE (MGD) | MAE (MGD) | Mean accuracy (1 - MAPE) | Counted correct (-0.2/+0.3 MGD) | within +/-10% | within +/-20% | within +/-30% |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in df.iterrows():
            lines.append(f"| {r[label_col]} | {int(r['N'])} | {r['R2']:.3f} | {r['RMSE']:.3f} | {r['MAE']:.3f} | "
                         f"{r['Accuracy_1_minus_MAPE_Pct']:.1f}% | {r['Pct_Days_Correct_Band']:.1f}% | {r['Pct_Days_Within_10Pct']:.1f}% | "
                         f"{r['Pct_Days_Within_20Pct']:.1f}% | {r['Pct_Days_Within_30Pct']:.1f}% |")
        return "\n".join(lines)

    L = []
    L.append("# Model Findings – Screened Decomposition Model (Phase 5) – Richmond RRWWTF")
    L.append("")
    L.append("Output of `scripts/train_screened_model.py`. Same split, scoring function and decomposition framing as Phase 4 "
             "(`output/10_decomp_model/`); three input changes motivated by `output/12_r2_ceiling_diagnostic/`: "
             "max(plant, Open-Meteo) rainfall, Open-Meteo minimum temperature with a freeze flag, and a uniform "
             "no-rain/no-freeze spike screen. Selection on 2018-2022 CV folds only; TEST 2024 scored once.")
    L.append("")
    L.append("## How to read the accuracy numbers")
    L.append("")
    L.append("- **R2**: share of day-to-day flow *variance* explained. Dominated by the largest spikes, so a few unexplained "
             "4 MGD days pull it down even when typical days are predicted well.")
    L.append("- **Mean accuracy (1 - MAPE)**: 100% minus the average absolute percent error across days.")
    L.append("- **Within +/-X%**: share of days on which the prediction was within X% of the observed flow.")
    L.append(f"- **Counted correct**: share of days on which the prediction was within -{TOL_BAND_LOWER_MGD:g} / +{TOL_BAND_UPPER_MGD:g} MGD "
             f"of the observed flow (e.g. observed 1.7 MGD -> any prediction from 1.5 to 2.0 counts). Operational tolerance agreed for reporting.")
    L.append("- RMSE/MAE in MGD, computed by `model_eval.score()` exactly as for every earlier model and baseline.")
    L.append("")
    L.append("## 24-hour normalization of the target")
    L.append("")
    L.append("The plant log's `Total Treated MGD` is the raw difference between two consecutive manual totalizer "
             "readings. Reading times are whenever an operator gets to the meter -- 06:43 to 10:16 observed -- so the "
             "interval each 'daily' total actually covers ranges from 21.4 to 26.2 hours. **43.6% of all days fall "
             "outside a 23-25 hour window**, consistently across every year (36-49%). Each day is therefore rescaled "
             "by 24/elapsed so the target is a true 24-hour volume, which is also the correct basis for capacity "
             "planning. This changes 38% of days by more than 5% (max 1.79 MGD). Elapsed-time entries outside "
             f"{ELAPSED_VALID_RANGE[0]:g}-{ELAPSED_VALID_RANGE[1]:g} h are log errors rather than real intervals and are left uncorrected. "
             "Caveat: the rescaling is linear and so assumes flow is uniform across the interval, which overnight it "
             "is not; it removes most of the interval noise but not all of it. Source column extracted to "
             "`output/12_r2_ceiling_diagnostic/plant_log_totalizer_elapsed.csv`.")
    L.append("")
    L.append("**Recommended process fix:** standardising the meter-reading time would remove this error at source at no cost.")
    L.append("")
    L.append("## Screening rule")
    L.append("")
    L.append(f"A day is excluded (set to NaN, never interpolated) when flow > {SPIKE_K:g} x the {LEVEL_WINDOW}-day dry-weather recent "
             f"level AND no rain > {NO_RAIN_IN:g} in on t, t-1, t-2 in either gauge AND Tmin >= {FREEZE_F:g} F on t and t-1. "
             f"This is the Phase-1 `Implausible_Spike_No_Rain` logic applied against the recent level instead of neighbor days, "
             f"so multi-day plateaus are caught. Applied to every year: {int(detail.shape[0])} days flagged, by year {by_year}.")
    L.append("")
    L.append("Flagged days: " + ", ".join(d.strftime("%Y-%m-%d") for d in detail.index) + " (details in `screened_days.csv`).")
    L.append("")
    L.append("These are +1.3 to +1.6 MGD jumps with no rainfall in either record and mild temperatures. Whether they are real "
             "sewer flow or plant/meter events (bypass, basin drawdown, wet-well catch-up, totalizer issue) should be confirmed "
             "with plant operations; the model cannot resolve it, so scores are reported both with and without them.")
    L.append("")
    L.append("## Model")
    L.append("")
    L.append(f"XGBoost on excess-over-recent-level, {len(FEATURE_COLUMNS)} features, wet-day sample weight {WET_DAY_SAMPLE_WEIGHT:g}x, "
             f"params {params}, n_estimators={n_estimators} (avg early-stopping across 3 folds). {n_rows:,} complete rows ({n_dropped:,} dropped, no fill).")
    L.append("")
    L.append("Top features: " + ", ".join(f"{r.Feature} ({r.Importance:.2f})" for r in importance.head(6).itertuples()) + ".")
    L.append("")
    L.append("## TEST 2024")
    L.append("")
    L.append(f"- **Screened protocol (model's own):** R2={s_all['R2']:.3f}, RMSE={s_all['RMSE']:.3f} MGD, mean accuracy "
             f"{s_all['Accuracy_1_minus_MAPE_Pct']:.1f}%, within +/-20% on {s_all['Pct_Days_Within_20Pct']:.1f}% of days (n={int(s_all['N'])}).")
    L.append(f"  - Dry days: RMSE={s_dry['RMSE']:.3f} MGD (n={int(s_dry['N'])}) -- at the ~0.26 MGD analog-day noise floor from `output/11_dry_diagnostic/`.")
    L.append(f"  - Wet days: R2={s_wet['R2']:.3f}, RMSE={s_wet['RMSE']:.3f} MGD (n={int(s_wet['N'])}).")
    L.append(f"- **Flagged days put back:** R2={u_all['R2']:.3f}, RMSE={u_all['RMSE']:.3f} MGD, mean accuracy "
             f"{u_all['Accuracy_1_minus_MAPE_Pct']:.1f}%, within +/-20% on {u_all['Pct_Days_Within_20Pct']:.1f}% of days (n={int(u_all['N'])}).")
    L.append(f"- **Phase 4 reference (unscreened, plant gauge only):** R2={p4_r2:.3f}, RMSE={p4_rmse:.3f} MGD (n=232).")
    L.append("")
    L.append(acc_table(all_scores.assign(Row=all_scores["Protocol"] + " / " + all_scores["Subset"]), "Row"))
    L.append("")
    L.append("## Pooled out-of-sample (2020, 2021, 2022 CV validation years + TEST 2024)")
    L.append("")
    L.append("Each year is predicted by a model fitted only on years before it (expanding window), so every row is out-of-sample. "
             "This is the fairest single summary of expected accuracy.")
    L.append("")
    L.append(acc_table(oos, "Period"))
    L.append("")
    L.append("## Where it still fails")
    L.append("")
    L.append(f"High-flow days (above the train-period 90th percentile): RMSE={s_all['High_Flow_RMSE']:.3f} MGD, mean error "
             f"{s_all['High_Flow_Mean_Error']:+.3f} MGD -- storm peaks are still underpredicted. With daily rain totals from a "
             "single point, peak response cannot be resolved much further; sub-daily rainfall or lift-station runtimes would be the next input to add.")
    L.append("")
    L.append("## One-sentence summary for reporting")
    L.append("")
    persist = oos[oos["Period"].str.startswith("POOLED persistence")].iloc[0]
    L.append(f"Across {int(pooled['N']):,} out-of-sample days (four independent years), the model predicts daily plant flow with a mean "
             f"accuracy of {pooled['Accuracy_1_minus_MAPE_Pct']:.0f}% (MAPE {100 - pooled['Accuracy_1_minus_MAPE_Pct']:.0f}%); it is counted correct "
             f"(within -{TOL_BAND_LOWER_MGD:g}/+{TOL_BAND_UPPER_MGD:g} MGD) on {pooled['Pct_Days_Correct_Band']:.0f}% of days versus "
             f"{persist['Pct_Days_Correct_Band']:.0f}% for a persistence baseline, within +/-20% on "
             f"{pooled['Pct_Days_Within_20Pct']:.0f}% of days and within +/-30% on {pooled['Pct_Days_Within_30Pct']:.0f}%; R2 is {pooled['R2']:.2f}, "
             f"limited by storm-peak underprediction and, in 2024, by a handful of large non-rainfall flow spikes that no available input explains.")
    L.append("")
    FINDINGS_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

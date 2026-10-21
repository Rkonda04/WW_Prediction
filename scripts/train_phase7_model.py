"""
RRWWTF Flow Model -- Phase 7 (MRMS radar/gauge rainfall). Production model.

Phase 6 (nearby rain-gauge network) plus MRMS multi-sensor rainfall, which in
output/25_mrms_rain improved every out-of-sample year over the gauge network
alone. MRMS daily rain comes from the Iowa Environmental Mesonet IEMRE point
service (`mrms_precip_in`, calendar day) at a 3 x 3 grid of points +/-0.1 deg
around the plant, averaged.

Protocol (shared pipeline in train_screened_model.py, unchanged): 24-hour
normalization, decomposition target (flow minus 60-day dry-weather level),
wet-day weight, expanding-window CV on 2018-2022 for all choices, 2024 scored
once.

Decided before scoring:
  * MRMS day shift: chosen by correlation with the gauge network on 2018-2022,
    never against flow.
  * The no-rain spike screen asks for no rain in ANY record -- max(plant,
    Open-Meteo, gauge network, MRMS).
  * Three candidates on one shared row set; CV on 2018-2022 picks one:
      gauge_rain_x_antecedent        Phase 6 as shipped (control)
      avg_rain_x_antecedent          rain = mean(gauge network, MRMS)
      gauge_rain_plus_mrms           gauge-network rain + mrms_t, mrms_lag1
  * The Phase 6 control is scored on the same rows so the comparison is like
    for like.

Outputs (output/26_phase7_mrms_rain/): same file names and formats as the
earlier phase folders, so the dashboard, 7-day forecast and rating curve read
it directly, plus mrms_daily_9pt.csv and production_model.json.

Usage:
    python scripts/train_phase7_model.py
"""

import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import model_eval
import scorecard
import train_decomposition_model as p4
import train_screened_model as tsm
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "26_phase7_mrms_rain"
MRMS_CSV = OUT_DIR / "mrms_daily_9pt.csv"
MODEL_NAME = "XGBoost_Phase7_MRMS"

CENTER_LAT, CENTER_LON = 29.5838, -95.7552  # Richmond COOP station, ~1 mi from the plant
OFFSETS = [-0.1, 0.0, 0.1]
YEARS = range(2017, 2025)
IEM_URL = "https://mesonet.agron.iastate.edu/iemre/multiday/{a}/{b}/{lat:.4f}/{lon:.4f}/json"
SHIFTS = [-1, 0, 1]

PHASE6_FEATURES = list(tsm.FEATURE_COLUMNS) + wx.INTERACTION_FEATURES
MRMS_FEATURES = ["mrms_t", "mrms_lag1"]
CONTROL = "gauge_rain_x_antecedent"


def fetch_mrms() -> pd.Series:
    """Daily MRMS area-mean rain (in), cached to MRMS_CSV."""
    if not MRMS_CSV.exists():
        cols = {}
        for dy in OFFSETS:
            for dx in OFFSETS:
                lat, lon = CENTER_LAT + dy, CENTER_LON + dx
                parts = []
                for y in YEARS:
                    a, b = ("2017-12-01", "2017-12-31") if y == 2017 else (f"{y}-01-01", f"{y}-12-31")
                    with urllib.request.urlopen(IEM_URL.format(a=a, b=b, lat=lat, lon=lon), timeout=60) as r:
                        data = json.load(r)["data"]
                    parts.append(pd.Series({pd.Timestamp(x["date"]): x["mrms_precip_in"] for x in data}, dtype=float))
                cols[f"pt_{dy:+.1f}_{dx:+.1f}"] = pd.concat(parts)
        df = pd.DataFrame(cols).sort_index()
        df["mrms_mean_in"] = df.mean(axis=1, skipna=True)
        df.round(3).to_csv(MRMS_CSV, index_label="Date")
    return pd.read_csv(MRMS_CSV, parse_dates=["Date"]).set_index("Date")["mrms_mean_in"]


def mrms_shift(mrms: pd.Series, net: pd.Series) -> int:
    corr = {s: pd.concat([mrms.shift(s).reindex(net.index), net], axis=1).loc["2018":"2022"].dropna().corr().iloc[0, 1]
            for s in SHIFTS}
    return max(corr, key=corr.get)


def load_inputs():
    """(ts0, om, gauge-network rain, MRMS rain aligned to the plant day, spike-screen rain)."""
    ts0, om, rain_p5 = tsm.load_inputs()
    net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts0.index)
    net_rain = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_p5)
    mrms = fetch_mrms()
    mrms = mrms.shift(mrms_shift(mrms, net_rain)).reindex(ts0.index).fillna(net_rain)
    screen = np.fmax(np.fmax(rain_p5, net_rain), mrms)
    return ts0, om, net_rain, mrms, screen


def build_frame(ts0, om, rain, screen_rain, mrms=None):
    _, flagged, detail = tsm.apply_screen(ts0, om, screen_rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan
    feat = wx.add_interactions(tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"]))
    if mrms is not None:
        feat["mrms_t"], feat["mrms_lag1"] = mrms, mrms.shift(1)
    return feat, flagged, detail


def candidates(ts0, om, net_rain, mrms, screen):
    """name -> (frame, feature columns, rain input description)."""
    f_net, flagged, detail = build_frame(ts0, om, net_rain, screen, mrms)
    f_avg, _, _ = build_frame(ts0, om, (net_rain + mrms) / 2, screen, mrms)
    return {
        CONTROL: (f_net, PHASE6_FEATURES, "nearby gauge network mean (Phase 6)"),
        "avg_rain_x_antecedent": (f_avg, PHASE6_FEATURES, "mean of gauge network and MRMS (inches, plant-day aligned)"),
        "gauge_rain_plus_mrms": (f_net, PHASE6_FEATURES + MRMS_FEATURES,
                                 "gauge network mean as rainfall_t; MRMS area mean as mrms_t / mrms_lag1"),
    }, flagged, detail


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TRAIN PHASE 7 -- MRMS RAINFALL\n")
    ts0, om, net_rain, mrms, screen = load_inputs()
    cands, flagged, detail = candidates(ts0, om, net_rain, mrms, screen)
    detail.to_csv(OUT_DIR / "screened_days.csv", index_label="Date")
    print(f"Spike screen: {int(flagged.sum())} days\n")

    need = ["recent_level", "flow_true", "target_excess"]
    rows = None
    for f, cols, _ in cands.values():
        ok = f.index[f[cols + need].notna().all(axis=1)]
        rows = ok if rows is None else rows.intersection(ok)
    n_dropped = int(len(cands[CONTROL][0].loc["2018":]) - len(rows))
    dates = rows.to_series().reset_index(drop=True)
    train_mask, test_mask = (m.values for m in model_eval.get_train_test_masks(dates))
    print(f"Shared row set: {len(rows):,} days\n")

    print("CV SELECTION (2018-2022 folds only) over candidates x XGBoost grid")
    cv_rows, fits = [], {}
    for name, (f, cols, _) in cands.items():
        print(f" candidate: {name} ({len(cols)} features)")
        params, n_est, res = tsm.cv_select(f.loc[rows], dates, cols)
        cv_rows.append(res.assign(Feature_Set=name))
        fits[name] = (float(res["Avg_CV_RMSE"].min()), params, n_est)
    set_name = min((k for k in fits if k != CONTROL), key=lambda k: fits[k][0])
    if fits[CONTROL][0] < fits[set_name][0]:
        set_name = CONTROL
    cv_score, params, n_est = fits[set_name]
    f, cols, rain_desc = cands[set_name]
    pd.concat(cv_rows).to_csv(OUT_DIR / "cv_selection.csv", index=False)
    print(f" -> selected '{set_name}', {params}, n_estimators={n_est}, avg CV RMSE {cv_score:.4f}\n")

    complete = f.loc[rows]
    high_flow = float(complete.loc[train_mask, "flow_true"].quantile(tsm.HIGH_FLOW_PERCENTILE))
    model = tsm.fit_xgb(complete, train_mask, params, n_est, cols)
    test_df = complete.loc[test_mask]
    flow_pred = test_df["recent_level"].values + model.predict(test_df[cols].values)
    scores = tsm.score_subsets(test_df, flow_pred, high_flow).assign(Protocol="Screened")

    feat_u = f.copy()
    feat_u["flow_true"] = feat_u["flow_unscreened"]
    complete_u, _ = tsm.drop_incomplete(feat_u, require_target=False, cols=cols)
    _, test_mask_u = model_eval.get_train_test_masks(complete_u.index.to_series().reset_index(drop=True))
    test_u = complete_u.loc[test_mask_u.values]
    pred_u = test_u["recent_level"].values + model.predict(test_u[cols].values)
    scores_u = tsm.score_subsets(test_u, pred_u, high_flow).assign(Protocol="Unscreened_FlaggedDaysIncluded")
    all_scores = pd.concat([scores, scores_u], ignore_index=True)
    all_scores = all_scores[["Protocol"] + [c for c in all_scores.columns if c != "Protocol"]]
    all_scores.insert(0, "Model", MODEL_NAME)
    all_scores.to_csv(OUT_DIR / "test_scores.csv", index=False)

    oos_path = OUT_DIR / "out_of_sample_predictions_all_years.csv"
    oos = tsm.pooled_out_of_sample(complete, dates, params, n_est, train_mask, test_mask, cols, oos_path)
    oos.to_csv(OUT_DIR / "pooled_out_of_sample_scores.csv", index=False)

    # Phase 6 control on the same rows, with its own CV-selected hyperparameters.
    _, c_params, c_n = fits[CONTROL]
    c_frame, c_cols, _ = cands[CONTROL]
    control_path = OUT_DIR / "control_phase6_out_of_sample_predictions.csv"
    tsm.pooled_out_of_sample(c_frame.loc[rows], dates, c_params, c_n, train_mask, test_mask, c_cols, control_path)

    pd.DataFrame({"Date": test_u.index, "Observed_MGD": test_u["flow_true"].values, "Predicted_MGD": pred_u,
                  "Screened_Out": flagged.reindex(test_u.index).values, "Rain_in": test_u["rainfall_t"].values,
                  "Tmin_F": test_u["tmin_t"].values}).to_csv(OUT_DIR / "test_predictions_2024.csv", index=False)

    importance = pd.DataFrame({"Feature": cols, "Importance": model.feature_importances_}) \
        .sort_values("Importance", ascending=False)
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    production = tsm.fit_xgb(complete, np.ones(len(complete), bool), params, n_est, cols)
    production.save_model(OUT_DIR / "production_model.json")
    (OUT_DIR / "production_features.json").write_text(json.dumps(
        {"feature_set": set_name, "features": cols, "params": params, "n_estimators": n_est,
         "trained_on": f"{complete.index.min():%Y-%m-%d} to {complete.index.max():%Y-%m-%d}",
         "n_rows": int(len(complete)), "rain_input": rain_desc,
         "forecast_rain": "the forecast feeds every rain input (rainfall_t and, if present, mrms_t)",
         "prediction": "flow_MGD = recent_level + model output"}, indent=1), encoding="utf-8")

    hybrid = complete["recent_level"] + complete["flow_true"].shift(1) - complete["recent_level"].shift(1)
    p4.MODEL_COLORS[MODEL_NAME] = "#c0392b"
    p4.plot_test_overlay(complete, test_mask, flow_pred, MODEL_NAME, hybrid_pred=hybrid.loc[test_df.index],
                         path=OUT_DIR / "01_test_period_overlay.png")
    p4.plot_scatter(complete, test_mask, flow_pred, MODEL_NAME, OUT_DIR / "02_scatter.png")
    p4.plot_feature_importance(importance, MODEL_NAME, OUT_DIR / "03_feature_importance.png")

    ctrl = scorecard.run(control_path, "Phase6_control_on_Phase7_rows")
    p7 = scorecard.run(oos_path, "Phase7_mrms_rain")
    write_findings(all_scores, detail, set_name, cols, params, n_est, cv_rows, importance, len(complete),
                   n_dropped, ctrl, p7, rain_desc)
    print("\nPHASE 7 COMPLETE ->", OUT_DIR.relative_to(BASE_DIR))


def write_findings(all_scores, detail, set_name, cols, params, n_est, cv_rows, importance, n_rows, n_dropped,
                   ctrl, p7, rain_desc):
    def row(protocol, subset):
        return all_scores[(all_scores["Protocol"] == protocol) & (all_scores["Subset"] == subset)].iloc[0]

    s_all, s_wet = row("Screened", "All"), row("Screened", "Wet")
    cv = pd.concat(cv_rows).groupby("Feature_Set")["Avg_CV_RMSE"].min()
    track = [("MAE, all days (MGD)", "MAE_All"), ("MAE, wet days (MGD)", "MAE_Wet"), ("R2, all days", "R2_All"),
             ("R2, wet days", "R2_Wet"), ("MAE, high-flow days >= 2.5 MGD", "MAE_HighFlow"),
             ("Bias, high-flow days (MGD)", "Bias_HighFlow"), ("Within +/-0.3 MGD (% of days)", "Within_0.3MGD_Pct"),
             ("Wet-day P99 gap, predicted - observed (MGD)", "CDF_Gap_P99_Wet")]
    L = [
        "# Model Findings -- Phase 7 (MRMS rainfall) -- Richmond RRWWTF", "",
        f"Output of `scripts/train_phase7_model.py`. Production model. Rain input: {rain_desc}.", "",
        "## Pooled out-of-sample scorecard (2020, 2021, 2022, 2024 -- each year predicted by a model that never saw it)", "",
        "Phase 6 control = the Phase 6 configuration refitted and scored on the same rows as Phase 7.", "",
        "| Metric | Phase 6 control | Phase 7 |", "| --- | --- | --- |",
        *[f"| {name} | {ctrl[k]:.3f} | {p7[k]:.3f} |" for name, k in track],
        f"| Days scored | {ctrl['N_Days']} | {p7['N_Days']} |", "",
        "Full tables and CDF figures: `output/21_wet_weather_review/scorecard/`.", "",
        "## TEST 2024 (screened protocol)", "",
        f"- Phase 7: R2={s_all['R2']:.3f}, RMSE={s_all['RMSE']:.3f}, MAE={s_all['MAE']:.3f} MGD (n={int(s_all['N'])}); "
        f"wet days R2={s_wet['R2']:.3f}, MAE={s_wet['MAE']:.3f}.", "",
        "## Choices (all made on 2018-2022 CV folds only)", "",
        *[f"- Candidate `{k}`: best avg CV RMSE {v:.4f}" for k, v in cv.items()],
        f"- Selected `{set_name}` ({len(cols)} features), params {params}, n_estimators={n_est}.", "",
        "## Spike screen", "",
        f"{len(detail)} days flagged (no rain > 0.1 in on t, t-1, t-2 in any record incl. MRMS, no freeze, "
        "flow > 1.5 x dry level).", "",
        "## Top features", "",
        ", ".join(f"{r.Feature} ({r.Importance:.2f})" for r in importance.head(8).itertuples()), "",
        "## Caveats", "",
        "- MRMS comes from the IEM IEMRE grid (~0.125 deg); the 9 sample points fall in about 6 distinct cells.",
        "- For forward use there is one rain forecast; it feeds every rain input. The backtests used recorded rain, "
        "so real forecast-rain error is extra.",
        "- Storm peaks remain under-predicted; see the rating curve (`output/21_wet_weather_review/rating_curve/`).",
        f"- {n_rows:,} complete rows ({n_dropped:,} dropped, no fill).",
    ]
    (OUT_DIR / "model_findings.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

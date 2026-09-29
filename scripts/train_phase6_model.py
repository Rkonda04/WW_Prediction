"""
RRWWTF Flow Model -- Phase 6 (nearby rain-gauge network).

Phase 5 (scripts/train_screened_model.py) with the rainfall input replaced by
the average of the nearby NOAA/CoCoRaHS gauges (scripts/fetch_wet_weather_inputs.py).
In the wet-weather review (output/21_wet_weather_review/experiments) that one
change improved every out-of-sample year.

Everything else is the Phase 5 protocol unchanged: 24-hour normalization,
decomposition target (flow minus 60-day dry-weather level), wet-day weight,
expanding-window CV on 2018-2022 for all choices, 2024 scored once.

Two differences from Phase 5, both decided before scoring:
  * The no-rain spike screen asks for no rain in ANY record -- max(plant,
    Open-Meteo, gauge network) -- so a day a nearby gauge saw rain is not
    thrown out as an unexplained spike.
  * Two feature sets are candidates -- the Phase 5 features, and the same plus
    rain x 30-day antecedent rain interactions. CV on the training years picks
    one; 2024 plays no part in the choice.

Outputs (output/22_phase6_gauge_network/): same file names and formats as
output/13_screened_model so the dashboards read either, plus
production_model.json (fitted on 2018-2024, for forward prediction).
"""

import json
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
OUT_DIR = BASE_DIR / "output" / "22_phase6_gauge_network"
PHASE5_DIR = BASE_DIR / "output" / "13_screened_model"
MODEL_NAME = "XGBoost_Phase6_GaugeNetwork"
FEATURE_SETS = {
    "gauge_rain": list(tsm.FEATURE_COLUMNS),
    "gauge_rain_x_antecedent": list(tsm.FEATURE_COLUMNS) + wx.INTERACTION_FEATURES,
}


def load_inputs():
    ts0, om, rain_p5 = tsm.load_inputs()
    net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts0.index)
    rain = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_p5)
    return ts0, om, rain, np.fmax(rain_p5, rain)


def build_frame(ts0, om, rain, screen_rain):
    _, flagged, detail = tsm.apply_screen(ts0, om, screen_rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan
    feat = wx.add_interactions(tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"]))
    return feat, flagged, detail


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("TRAIN PHASE 6 -- GAUGE NETWORK RAINFALL\n")
    ts0, om, rain, screen_rain = load_inputs()
    feat, flagged, detail = build_frame(ts0, om, rain, screen_rain)
    detail.to_csv(OUT_DIR / "screened_days.csv", index_label="Date")
    p5_flagged = pd.read_csv(PHASE5_DIR / "screened_days.csv", parse_dates=["Date"])["Date"]
    unflagged = sorted(set(p5_flagged) - set(detail.index))
    print(f"Spike screen: {int(flagged.sum())} days (Phase 5: {len(p5_flagged)}); "
          f"no longer flagged because a nearby gauge recorded rain: {[d.strftime('%Y-%m-%d') for d in unflagged]}\n")

    all_cols = FEATURE_SETS["gauge_rain_x_antecedent"]
    complete, n_dropped = tsm.drop_incomplete(feat, cols=all_cols)
    dates = complete.index.to_series().reset_index(drop=True)
    train_mask, test_mask = (m.values for m in model_eval.get_train_test_masks(dates))

    print("CV SELECTION (2018-2022 folds only) over feature sets x XGBoost grid")
    cv_rows, best = [], None
    for set_name, cols in FEATURE_SETS.items():
        print(f" feature set: {set_name} ({len(cols)} features)")
        params, n_est, res = tsm.cv_select(complete, dates, cols)
        cv_rows.append(res.assign(Feature_Set=set_name))
        score = float(res["Avg_CV_RMSE"].min())
        if best is None or score < best[0]:
            best = (score, set_name, cols, params, n_est)
    cv_score, set_name, cols, params, n_est = best
    pd.concat(cv_rows).to_csv(OUT_DIR / "cv_selection.csv", index=False)
    print(f" -> selected '{set_name}', {params}, n_estimators={n_est}, avg CV RMSE {cv_score:.4f}\n")

    high_flow = float(complete.loc[train_mask, "flow_true"].quantile(tsm.HIGH_FLOW_PERCENTILE))
    model = tsm.fit_xgb(complete, train_mask, params, n_est, cols)
    test_df = complete.loc[test_mask]
    flow_pred = test_df["recent_level"].values + model.predict(test_df[cols].values)
    scores = tsm.score_subsets(test_df, flow_pred, high_flow).assign(Protocol="Screened")

    feat_u = feat.copy()
    feat_u["flow_true"] = feat_u["flow_unscreened"]
    complete_u, _ = tsm.drop_incomplete(feat_u, require_target=False, cols=all_cols)
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
         "n_rows": int(len(complete)), "rain_input": "nearby gauge network mean (inches, plant-day aligned)",
         "prediction": "flow_MGD = recent_level + model output"}, indent=1), encoding="utf-8")

    hybrid = feat["recent_level"] + feat["flow_true"].shift(1) - feat["recent_level"].shift(1)
    p4.MODEL_COLORS[MODEL_NAME] = "#c0392b"
    p4.plot_test_overlay(complete, test_mask, flow_pred, MODEL_NAME, hybrid_pred=hybrid.loc[test_df.index],
                         path=OUT_DIR / "01_test_period_overlay.png")
    p4.plot_scatter(complete, test_mask, flow_pred, MODEL_NAME, OUT_DIR / "02_scatter.png")
    p4.plot_feature_importance(importance, MODEL_NAME, OUT_DIR / "03_feature_importance.png")

    p5 = scorecard.run(PHASE5_DIR / "out_of_sample_predictions_all_years.csv", "Phase5_shipped")
    p6 = scorecard.run(oos_path, "Phase6_gauge_network")
    write_findings(all_scores, oos, detail, unflagged, set_name, cols, params, n_est, cv_rows, importance,
                   len(complete), n_dropped, p5, p6)
    print("\nPHASE 6 COMPLETE ->", OUT_DIR.relative_to(BASE_DIR))


def write_findings(all_scores, oos, detail, unflagged, set_name, cols, params, n_est, cv_rows, importance,
                   n_rows, n_dropped, p5, p6):
    def row(protocol, subset):
        return all_scores[(all_scores["Protocol"] == protocol) & (all_scores["Subset"] == subset)].iloc[0]

    p5_test = pd.read_csv(PHASE5_DIR / "test_scores.csv")
    p5_all = p5_test[(p5_test["Protocol"] == "Screened") & (p5_test["Subset"] == "All")].iloc[0]
    p5_wet = p5_test[(p5_test["Protocol"] == "Screened") & (p5_test["Subset"] == "Wet")].iloc[0]
    s_all, s_wet = row("Screened", "All"), row("Screened", "Wet")
    cv = pd.concat(cv_rows).groupby("Feature_Set")["Avg_CV_RMSE"].min()

    track = [("MAE, all days (MGD)", "MAE_All"), ("MAE, wet days (MGD)", "MAE_Wet"), ("R2, all days", "R2_All"),
             ("R2, wet days", "R2_Wet"), ("MAE, high-flow days >= 2.5 MGD", "MAE_HighFlow"),
             ("Bias, high-flow days (MGD)", "Bias_HighFlow"), ("Within +/-0.3 MGD (% of days)", "Within_0.3MGD_Pct"),
             ("Wet-day P99 gap, predicted - observed (MGD)", "CDF_Gap_P99_Wet")]
    L = [
        "# Model Findings -- Phase 6 (nearby rain-gauge network) -- Richmond RRWWTF", "",
        "Output of `scripts/train_phase6_model.py`. Phase 5 protocol unchanged except the rainfall input: the daily "
        f"average of the NOAA/CoCoRaHS gauges within 6 mi of Richmond replaces max(plant gauge, Open-Meteo).", "",
        "## Pooled out-of-sample scorecard (2020, 2021, 2022, 2024 -- each year predicted by a model that never saw it)", "",
        "| Metric | Phase 5 | Phase 6 |", "| --- | --- | --- |",
        *[f"| {name} | {p5[k]:.3f} | {p6[k]:.3f} |" for name, k in track],
        f"| Days scored | {p5['N_Days']} | {p6['N_Days']} |", "",
        "Full tables and CDF figures: `output/21_wet_weather_review/scorecard/`.", "",
        "## TEST 2024 (screened protocol)", "",
        f"- Phase 6: R2={s_all['R2']:.3f}, RMSE={s_all['RMSE']:.3f}, MAE={s_all['MAE']:.3f} MGD (n={int(s_all['N'])}); "
        f"wet days R2={s_wet['R2']:.3f}, MAE={s_wet['MAE']:.3f}.",
        f"- Phase 5: R2={p5_all['R2']:.3f}, RMSE={p5_all['RMSE']:.3f}, MAE={p5_all['MAE']:.3f} MGD (n={int(p5_all['N'])}); "
        f"wet days R2={p5_wet['R2']:.3f}, MAE={p5_wet['MAE']:.3f}.", "",
        "## Choices (all made on 2018-2022 CV folds only)", "",
        *[f"- Feature set `{k}`: best avg CV RMSE {v:.4f}" for k, v in cv.items()],
        f"- Selected `{set_name}` ({len(cols)} features), params {params}, n_estimators={n_est}.", "",
        "## Spike screen", "",
        f"{len(detail)} days flagged (no rain > 0.1 in on t, t-1, t-2 in any record, no freeze, flow > 1.5 x dry level). "
        + (f"No longer flagged vs Phase 5 because a nearby gauge recorded rain: {', '.join(d.strftime('%Y-%m-%d') for d in unflagged)}."
           if unflagged else "Same days as Phase 5."), "",
        "## Top features", "",
        ", ".join(f"{r.Feature} ({r.Importance:.2f})" for r in importance.head(8).itertuples()), "",
        "## Caveats", "",
        "- The gauge network is volunteer (CoCoRaHS) plus one NOAA COOP station; a median of 4 gauges report each day. "
        "For forward use, the rain input is a forecast of sewershed-average rainfall, which this network mean represents "
        "better than a single point gauge.",
        "- Storm peaks remain under-predicted; see the rating curve (`output/21_wet_weather_review/rating_curve/`).",
        f"- {n_rows:,} complete rows ({n_dropped:,} dropped, no fill).",
    ]
    (OUT_DIR / "model_findings.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()

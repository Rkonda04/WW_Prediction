"""
Seven-day flow forecast from the Phase 6 model.

The Phase 6 model predicts one day from yesterday's flow. A 7-day forecast
rolls it forward: day 1 is predicted from the measured state, and each later
day is predicted from the previous day's PREDICTION (flow lag and the three
excess lags), with that day's forecast rainfall and minimum temperature.

Held fixed across the 7 days, because they depend on flows not yet measured:
the 60-day dry-weather level and its 30-day slope, both as of the start day.

Backtest (what this script measures):
  * Every start day whose state is complete, in each out-of-sample year --
    2020, 2021, 2022 by the expanding-window CV folds and 2024 held out, each
    year rolled out by a model that never saw it (Phase 6 hyperparameters).
  * Rain and temperature for the 7 days are the RECORDED values, i.e. a perfect
    weather forecast. Real forecasts add rainfall error on top of these scores.
  * Scored per lead day against measured flow, beside two no-skill references:
    persistence (yesterday's flow held) and the dry-weather level.

Outputs (output/23_seven_day_forecast/):
    lead_time_scores.csv        error by lead day, all / wet / dry
    backtest_predictions.csv    every start day x lead day
    forecast_tool.json          the production model's trees + error ranges +
                                latest state, read by the dashboard (app/)
Usage (after scripts/train_phase6_model.py):
    python scripts/forecast_7day.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import model_eval
import train_phase6_model as p6
import train_screened_model as tsm
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "23_seven_day_forecast"
HORIZON = 7
TOL_MGD = 0.3
QUANTILES = (10, 25, 75, 90)
MIN_N_FOR_RANGE = 30


def rollout(feat: pd.DataFrame, starts: np.ndarray, model, cols) -> np.ndarray:
    """Predicted flow, shape (len(starts), HORIZON), for start-day positions `starts` in `feat`."""
    base = feat.iloc[starts]
    level = base["recent_level"].to_numpy()
    slope = base["level_slope30"].to_numpy()
    lag1, lag2 = base["excess_lag1"].to_numpy(), base["excess_lag2"].to_numpy()
    pred = np.empty((len(starts), HORIZON))
    for k in range(HORIZON):
        x = feat.iloc[starts + k][cols].copy()
        if k:
            x["flow_lag1"] = pred[:, k - 1]
            x["excess_lag3"], x["excess_lag2"], x["excess_lag1"] = lag2, lag1, pred[:, k - 1] - level
            x["excess_mean3"] = (x["excess_lag1"] + x["excess_lag2"] + x["excess_lag3"]) / 3
            x["recent_level"], x["level_slope30"] = level, slope
            lag2, lag1 = lag1, x["excess_lag1"].to_numpy()
        pred[:, k] = level + model.predict(x[cols].values)
    return pred


def valid_starts(feat: pd.DataFrame, cols, year_mask: np.ndarray) -> np.ndarray:
    """Positions with a complete measured state on the start day and weather for all 7 days."""
    exo = feat[["rainfall_t", "rain_lag1", "rain_lag2", "ante_5d", "ante_30d", "days_since_rain",
                "tmin_t", "tmin_lag1"]].notna().all(axis=1).to_numpy()
    state = feat[cols].notna().all(axis=1).to_numpy()
    n = len(feat)
    ok = [i for i in np.flatnonzero(year_mask & state) if i + HORIZON <= n and exo[i:i + HORIZON].all()]
    return np.array(ok, dtype=int)


def score(d: pd.DataFrame, col: str) -> dict:
    y, p = d["actual"].to_numpy(), d[col].to_numpy()
    e = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"MAE": float(np.abs(e).mean()), "Bias": float(e.mean()),
            "R2": 1 - float((e ** 2).sum()) / ss_tot if len(d) > 2 and ss_tot > 0 else np.nan,
            "Within_0.3MGD_Pct": float((np.abs(e) <= TOL_MGD).mean() * 100)}


def export_trees(model) -> dict:
    learner = json.loads(model.get_booster().save_raw("json"))["learner"]
    trees = [{"f": t["split_indices"], "t": t["split_conditions"], "l": t["left_children"],
              "r": t["right_children"]} for t in learner["gradient_booster"]["model"]["trees"]]
    base = float(learner["learner_model_param"]["base_score"].strip("[]"))
    return {"base_score": base, "trees": trees}


def predict_trees(tool: dict, x: np.ndarray) -> float:
    """Same walk the dashboard does in JavaScript; used here to prove the export matches XGBoost."""
    x, out = x.astype(np.float32), tool["base_score"]
    for t in tool["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] < np.float32(t["t"][n]) else t["r"][n]
        out += t["t"][n]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("SEVEN-DAY FORECAST -- recursive rollout of the Phase 6 model\n")
    meta = json.loads((p6.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols, params, n_est = meta["features"], meta["params"], meta["n_estimators"]
    inputs = p6.load_inputs()
    feat, _, _ = p6.build_frame(*inputs)
    complete, _ = tsm.drop_incomplete(feat, cols=cols)
    cdates = complete.index.to_series().reset_index(drop=True)
    fdates = feat.index.to_series().reset_index(drop=True)

    train_mask, _ = model_eval.get_train_test_masks(cdates)
    folds = [(f["Val_Start"].year, model_eval.split_mask(cdates, f["Train_Start"], f["Train_End"]).values,
              model_eval.split_mask(fdates, f["Val_Start"], f["Val_End"]).values) for f in model_eval.CV_FOLDS]
    folds.append((model_eval.TEST_START.year, train_mask.values,
                  model_eval.split_mask(fdates, model_eval.TEST_START, model_eval.TEST_END).values))

    rows = []
    for year, fit_mask, year_mask in folds:
        model = tsm.fit_xgb(complete, fit_mask, params, n_est, cols)
        starts = valid_starts(feat, cols, year_mask)
        pred = rollout(feat, starts, model, cols)
        base = feat.iloc[starts]
        for k in range(HORIZON):
            day = feat.iloc[starts + k]
            rows.append(pd.DataFrame({
                "Start_Date": base.index, "Lead_Day": k + 1, "Date": day.index, "Year": year,
                "actual": day["flow_true"].to_numpy(), "predicted": pred[:, k],
                "persistence": base["flow_lag1"].to_numpy(), "dry_level": base["recent_level"].to_numpy(),
                "rain_in": day["rainfall_t"].to_numpy(),
                "wet": ((day["rainfall_t"] > tsm.WET) | (day["rain_lag1"] > tsm.WET)).to_numpy()}))
        print(f" {year}: {len(starts)} start days, model fitted on {int(fit_mask.sum())} earlier days")
    bt = pd.concat(rows, ignore_index=True)
    bt = bt[bt["actual"].notna() & (bt["Date"].dt.year == bt["Year"])]
    bt.to_csv(OUT_DIR / "backtest_predictions.csv", index=False, float_format="%.4f")

    scores, ranges = [], {"all": [], "wet": [], "dry": []}
    for k in range(1, HORIZON + 1):
        lead = bt[bt["Lead_Day"] == k]
        for name, d in (("All", lead), ("Wet", lead[lead["wet"]]), ("Dry", lead[~lead["wet"]])):
            m, ref_p, ref_l = score(d, "predicted"), score(d, "persistence"), score(d, "dry_level")
            scores.append({"Lead_Day": k, "Subset": name, "N": len(d), **m,
                           "MAE_Persistence": ref_p["MAE"], "R2_Persistence": ref_p["R2"],
                           "MAE_Dry_Level": ref_l["MAE"], "R2_Dry_Level": ref_l["R2"]})
            resid = (d["actual"] - d["predicted"]).to_numpy()
            ranges[name.lower()].append({
                "n": int(len(d)), "mae": round(m["MAE"], 3), "within_0_3": round(m["Within_0.3MGD_Pct"] / 100, 3),
                **{f"q{q}": round(float(np.percentile(resid, q)), 3) for q in QUANTILES}}
                if len(d) >= MIN_N_FOR_RANGE else None)
    scores = pd.DataFrame(scores)
    scores.round(4).to_csv(OUT_DIR / "lead_time_scores.csv", index=False)

    production = xgb.XGBRegressor()
    production.load_model(p6.OUT_DIR / "production_model.json")
    tool = export_trees(production)
    check = complete[cols].sample(200, random_state=0).values
    worst = max(abs(predict_trees(tool, x) - float(production.predict(x[None])[0])) for x in check)
    assert worst < 1e-4, f"exported trees disagree with XGBoost by {worst}"

    last, last_date = complete.iloc[-1], complete.index[-1]
    tmin = inputs[1]["temperature_2m_min"]
    a = scores[scores["Subset"] == "All"].set_index("Lead_Day")
    tool = {
        "features": cols, **tool, "horizon": HORIZON, "wet_rain_in": tsm.WET, "freeze_f": tsm.FREEZE_F,
        # Above the 99th-percentile wet day the trees have few examples and go flat.
        "p99_rain_in": round(float(feat.loc[feat["rainfall_t"] > tsm.WET, "rainfall_t"].quantile(0.99)), 2),
        # Residual (measured - predicted) by lead day from the backtest; index 0 = day 1.
        "ranges": ranges,
        "backtest": {"years": [f[0] for f in folds], "n_start_days": int(bt.loc[bt["Lead_Day"] == 1, "Start_Date"].nunique()),
                     "mae_day1": round(float(a.loc[1, "MAE"]), 3), "mae_day7": round(float(a.loc[HORIZON, "MAE"]), 3),
                     "mae_persistence_day7": round(float(a.loc[HORIZON, "MAE_Persistence"]), 3)},
        # State on the last recorded day -- only a placeholder for the form, the record is not live.
        "latest_state": {"date": last_date.strftime("%Y-%m-%d"), "dry_level_mgd": round(float(last["recent_level"]), 3),
                         "flow_mgd": round(float(last["flow_true"]), 3)},
        # Used only when the live weather feed cannot be reached.
        "tmin_normal_f_by_month": [round(float(v), 1) for v in tmin.groupby(tmin.index.month).median()],
        "model": f"Phase 6 production model, fitted {meta['trained_on']}",
    }
    (OUT_DIR / "forecast_tool.json").write_text(json.dumps(tool, separators=(",", ":")), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(f"\nexported trees match XGBoost to {worst:.1e} MGD on 200 rows")
    print("\nPooled out-of-sample, perfect weather forecast:")
    print(scores.round(3).to_string(index=False))
    print("\nSEVEN-DAY FORECAST COMPLETE ->", OUT_DIR.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()

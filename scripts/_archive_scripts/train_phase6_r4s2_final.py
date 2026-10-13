"""
RRWWTF Flow Model -- Phase 6 Final (R4S2: 4-day rain + 2-day soil antecedent).

Best-performing configuration from optimization testing:
- 4-day antecedent rainfall (captures recent saturation)
- 2-day antecedent soil moisture (multi-source: ERA5, ERA5-Land, ECMWF averaged over 3x3 grid)
- Rain x antecedent interactions for both rain and soil
- Gauge network rainfall input

Out-of-sample performance (1,214 common rows, 2020/21/22 CV + 2024):
  R² 0.605, MAE 0.196 MGD, Wet R² 0.642, High-flow bias -0.477

Outputs: output/22_phase6_gauge_network/r4s2_final/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xgboost as xgb

import model_eval
import scorecard
import train_screened_model as tsm
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "22_phase6_gauge_network" / "r4s2_final"
MODEL_NAME = "XGBoost_Phase6_R4S2_RainSoilAntecedent"
SOIL_CSV = wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv"

FEATURE_COLUMNS = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_4d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
    "rain_x_ante4", "rain_lag1_x_ante4",
    "sm_0_7cm_ante2", "sm_7_28cm_ante2",
]


def load_inputs():
    ts0, om, rain_p5 = tsm.load_inputs()
    net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts0.index)
    rain = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_p5)
    soil = pd.read_csv(SOIL_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts0.index)
    return ts0, om, rain, np.fmax(rain_p5, rain), soil


def build_frame(ts0, om, rain, screen_rain, soil):
    _, flagged, detail = tsm.apply_screen(ts0, om, screen_rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan

    feat = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])

    # Replace ante_30d with 4-day antecedent rain
    r = ts["Rainfall_in_Corrected"]
    feat["ante_4d"] = r.rolling(4, min_periods=4).sum().shift(1)
    feat.drop(columns=["ante_30d"], inplace=True)

    # Rain x 4-day antecedent interactions
    feat["rain_x_ante4"] = feat["rainfall_t"] * feat["ante_4d"]
    feat["rain_lag1_x_ante4"] = feat["rain_lag1"] * feat["ante_4d"]

    # 2-day soil antecedent features
    soil_2d = soil[["sm_0_7cm", "sm_7_28cm"]].reindex(ts.index)
    feat["sm_0_7cm_ante2"] = soil_2d["sm_0_7cm"].rolling(2, min_periods=2).mean().shift(1)
    feat["sm_7_28cm_ante2"] = soil_2d["sm_7_28cm"].rolling(2, min_periods=2).mean().shift(1)

    return feat


def fit_predict(f, cols, train_idx, pred_idx, params, n_estimators):
    tr = f.loc[train_idx]
    wet = (tr["rain_wet_mask_t"] | tr["rain_wet_mask_t1"]).values
    m = xgb.XGBRegressor(**params, n_estimators=n_estimators, objective="reg:squarederror", random_state=42)
    m.fit(tr[cols].values, tr["target_excess"].values,
          sample_weight=np.where(wet, tsm.WET_DAY_SAMPLE_WEIGHT, 1.0))
    pr = f.loc[pred_idx]
    return pr["recent_level"].values + m.predict(pr[cols].values), m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts0, om, rain, screen_rain, soil = load_inputs()
    feat = build_frame(ts0, om, rain, screen_rain, soil)

    # Drop incomplete rows
    complete, n_dropped = tsm.drop_incomplete(feat, cols=FEATURE_COLUMNS)
    print(f"Complete rows: {len(complete):,} (dropped {n_dropped} with NaN)\n")

    # Use Phase 6 original hyperparameters (proved best for this feature set)
    params = {'max_depth': 2, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 2.0, 'min_child_weight': 3}
    n_est = 71

    # Score with expanding-window CV on 2018-2022 + hold-out 2024
    dates = complete.index.to_series().reset_index(drop=True)
    feat_idx = complete.index
    ctx = scorecard.load_context()
    oos_parts = []

    print("Expanding-window CV on 2018-2022:")
    for period, train_mask, val_mask in model_eval.get_cv_fold_masks(dates):
        train_idx = feat_idx[train_mask.values]
        val_idx = feat_idx[val_mask.values]
        pred, m = fit_predict(complete, FEATURE_COLUMNS, train_idx, val_idx, params, n_est)
        s = model_eval.score(val_idx, complete.loc[val_idx, "flow_true"].values, pred)
        print(f"  {period}: R²={s['R2']:.3f}, MAE={s['MAE']:.3f}")
        oos_parts.append(pd.DataFrame({"Period": period, "y": complete.loc[val_idx, "flow_true"].values, "p": pred},
                                     index=val_idx))

    # 2024 test
    print("Hold-out test 2024:")
    train_mask, test_mask = model_eval.get_train_test_masks(dates)
    train_idx = feat_idx[train_mask.values]
    test_idx = feat_idx[test_mask.values]
    pred_test, m_test = fit_predict(complete, FEATURE_COLUMNS, train_idx, test_idx, params, n_est)
    s_test = model_eval.score(test_idx, complete.loc[test_idx, "flow_true"].values, pred_test)
    print(f"  TEST 2024: R²={s_test['R2']:.3f}, MAE={s_test['MAE']:.3f}")
    oos_parts.append(pd.DataFrame({"Period": "TEST 2024", "y": complete.loc[test_idx, "flow_true"].values, "p": pred_test},
                                 index=test_idx))

    # Combine and score pooled OOS
    oos = pd.concat(oos_parts)
    path = OUT_DIR / "oos_predictions.csv"
    oos.to_csv(path, index_label="Date")
    pooled = scorecard.run(path, "Phase6_R4S2", ctx)
    print(f"\nPooled OOS Performance (n={len(oos)}):")
    for k in ["R2_All", "MAE_All", "RMSE_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]:
        print(f"  {k}: {pooled[k]:.3f}")

    # Fit final model on full 2018-2024
    print("\nTraining production model on full 2018-2024...")
    m_prod = xgb.XGBRegressor(**params, n_estimators=n_est, objective="reg:squarederror", random_state=42)
    wet_all = (complete["rain_wet_mask_t"] | complete["rain_wet_mask_t1"]).values
    m_prod.fit(complete[FEATURE_COLUMNS].values, complete["target_excess"].values,
              sample_weight=np.where(wet_all, tsm.WET_DAY_SAMPLE_WEIGHT, 1.0))

    # Save outputs
    with open(OUT_DIR / "production_model.json", "w") as f:
        json.dump(m_prod.get_booster().get_dump(dump_format="json"), f, indent=2)

    with open(OUT_DIR / "production_features.json", "w") as f:
        json.dump({
            "feature_set": "gauge_rain_x_antecedent_soil",
            "features": FEATURE_COLUMNS,
            "params": params,
            "n_estimators": n_est,
            "trained_on": f"{complete.index.min():%Y-%m-%d} to {complete.index.max():%Y-%m-%d}",
            "n_rows": int(len(complete)),
            "rain_input": "nearby gauge network mean (inches, plant-day aligned)",
            "soil_input": "multi-source antecedent (ERA5, ERA5-Land, ECMWF averaged 3x3 grid, 2-day rolling mean)",
            "prediction": "flow_MGD = recent_level + model output"
        }, f, indent=1)

    with open(OUT_DIR / "model_metadata.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training: 2018-2024 (n={len(complete):,} days)\n")
        f.write(f"Features: {len(FEATURE_COLUMNS)}\n")
        f.write(f"  - Base: flow lags, rainfall, antecedent (5d + 4d), days since rain, temperature, day-of-year\n")
        f.write(f"  - Interactions: rain × 4-day antecedent (today + lag1)\n")
        f.write(f"  - Soil: 2-day antecedent soil moisture (0-7cm, 7-28cm layers, multi-source)\n")
        f.write(f"Hyperparameters: {params}\n")
        f.write(f"N_Estimators: {n_est}\n")
        f.write(f"\nPooled OOS Performance ({len(oos)} days):\n")
        for k in ["R2_All", "MAE_All", "RMSE_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]:
            f.write(f"  {k}: {pooled[k]:.3f}\n")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

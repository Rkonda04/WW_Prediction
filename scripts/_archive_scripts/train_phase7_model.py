"""
RRWWTF Flow Model -- Phase 7 (gauge network rain + soil antecedent).

Phase 6 (scripts/train_phase6_model.py) with two refinements from testing:
  1. Antecedent rain window: 30 days → 4 days (better captures recent saturation)
  2. Added 2-day antecedent soil moisture features (multi-source: ERA5, ERA5-Land, ECMWF)

Out-of-sample test (2020/21/22/24, n=1,214):
  Phase 6: R² 0.593, MAE 0.199, Wet R² 0.631, High-flow bias -0.511
  Phase 7: R² 0.605, MAE 0.196, Wet R² 0.642, High-flow bias -0.477 (+0.034)

Everything else identical to Phase 6: 24-hour normalization, decomposition target,
wet-day weight, expanding-window CV on 2018-2022, 2024 scored once.

Outputs: output/23_phase7_soil_antecedent/ (same structure as Phase 6)
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
OUT_DIR = BASE_DIR / "output" / "23_phase7_soil_antecedent"
MODEL_NAME = "XGBoost_Phase7_GaugeNetwork_SoilAntecedent"
SOIL_CSV = wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv"

# Feature columns: base features minus ante_30d, plus ante_4d, interactions, and soil
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
    """Build feature frame with 4-day rain antecedent + 2-day soil antecedent."""
    _, flagged, detail = tsm.apply_screen(ts0, om, screen_rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan

    # Base features
    feat = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])

    # Replace ante_30d with 4-day antecedent rain
    r = ts["Rainfall_in_Corrected"]
    feat["ante_4d"] = r.rolling(4, min_periods=4).sum().shift(1)
    feat.drop(columns=["ante_30d"], inplace=True)

    # Add rain x 4-day antecedent interactions
    feat["rain_x_ante4"] = feat["rainfall_t"] * feat["ante_4d"]
    feat["rain_lag1_x_ante4"] = feat["rain_lag1"] * feat["ante_4d"]

    # Add 2-day soil antecedent features
    soil_2d = soil[["sm_0_7cm", "sm_7_28cm"]].reindex(ts.index)
    feat["sm_0_7cm_ante2"] = soil_2d["sm_0_7cm"].rolling(2, min_periods=2).mean().shift(1)
    feat["sm_7_28cm_ante2"] = soil_2d["sm_7_28cm"].rolling(2, min_periods=2).mean().shift(1)

    return feat, flagged, detail


def fit_predict(f, cols, train_idx, pred_idx, params, n_estimators):
    """Fit XGBoost with wet-day weights and predict."""
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
    feat, flagged, spike_detail = build_frame(ts0, om, rain, screen_rain, soil)

    # Drop rows with NaN in required features
    complete, n_dropped = tsm.drop_incomplete(feat, cols=FEATURE_COLUMNS)
    print(f"Complete rows: {len(complete):,} (dropped {n_dropped} with NaN)\n")

    # CV selection (re-use Phase 6's hyperparameters)
    params, n_est = wx.shipped_params()

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
    pooled = scorecard.run(path, "Phase7", ctx)
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

    # Save metadata
    with open(OUT_DIR / "model_metadata.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training: 2018-2024 (n={len(complete):,} days)\n")
        f.write(f"Features: {len(FEATURE_COLUMNS)}\n")
        f.write(f"  - Base: flow lags, rainfall, antecedent (5d + 4d), days since rain, temperature, day-of-year\n")
        f.write(f"  - Interactions: rain × 4-day antecedent (today + lag1)\n")
        f.write(f"  - Soil: 2-day antecedent soil moisture (0-7cm, 7-28cm layers, multi-source)\n")
        f.write(f"Hyperparameters: {params}\n")
        f.write(f"N_Estimators: {n_est}\n")
        f.write(f"\nPooled OOS Performance (2020/21/22/24, n={len(oos)}):\n")
        for k in ["R2_All", "MAE_All", "RMSE_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]:
            f.write(f"  {k}: {pooled[k]:.3f}\n")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

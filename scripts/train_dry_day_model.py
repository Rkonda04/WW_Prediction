"""
Dry-day flow prediction model (separate from wet-day R4S2).

Dry days (low rainfall) have different drivers than wet days:
- No I&I dynamics
- Flow = baseflow + small leakage + seasonal pattern
- Temperature, day-of-week matter more

Strategy: Train lightweight model on dry conditions, then use hybrid:
  if rainfall < 0.1 and ante_5d < 0.5:
    use dry_day_model
  else:
    use wet_day_model (R4S2)

Outputs: output/22_phase6_gauge_network/dry_day_model/
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
OUT_DIR = BASE_DIR / "output" / "22_phase6_gauge_network" / "dry_day_model"
MODEL_NAME = "XGBoost_DryDayBaseline"

# Features for dry-day model (simpler, focused on baseflow)
# Using only features that build_feature_frame creates
DRY_FEATURES = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3",  # Recent flow history
    "rain_lag1", "rain_lag2",  # Yesterday/day-before rain (should be zero on dry days)
    "tmin_t", "tmin_lag1",  # Temperature (affects infiltration)
    "doy_sin", "doy_cos",  # Seasonal pattern
    "recent_level", "level_slope30",  # Baseline and trend
]


def load_wet_weather_data():
    """Load data including rainfall."""
    ts0, om, rain_p5 = tsm.load_inputs()
    net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date")
    rain = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_p5)
    soil = pd.read_csv(wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv",
                       parse_dates=["Date"]).set_index("Date")
    return ts0, om, rain, soil


def build_dry_day_frame(ts0, om, rain, soil):
    """Build feature frame for dry-day model."""
    _, flagged, _ = tsm.apply_screen(ts0, om, rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan

    feat = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])

    # Add soil antecedent for dryness metric
    soil_2d = soil[["sm_0_7cm"]].reindex(ts.index)
    feat["soil_dryness"] = 1.0 - soil_2d["sm_0_7cm"]  # Higher = drier

    return feat


def select_dry_days(feat, rain):
    """Select training data for dry conditions only."""
    # Dry: rainfall < 0.1 in and 5-day antecedent < 0.5 in
    dry_mask = (rain <= 0.1) & (feat["ante_5d"] <= 0.5)
    return feat[dry_mask].copy()


def fit_predict(f, cols, train_idx, pred_idx, params, n_estimators):
    """Fit dry-day model."""
    tr = f.loc[train_idx]
    # Less emphasis on wet-day weighting for dry model
    wet_mask = (tr["rain_wet_mask_t"] | tr["rain_wet_mask_t1"]).values
    sample_weight = np.where(wet_mask, 1.5, 1.0)  # Mild weighting

    m = xgb.XGBRegressor(**params, n_estimators=n_estimators,
                         objective="reg:squarederror", random_state=42)
    m.fit(tr[cols].values, tr["target_excess"].values, sample_weight=sample_weight)

    pr = f.loc[pred_idx]
    return pr["recent_level"].values + m.predict(pr[cols].values), m


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== DRY-DAY MODEL TRAINING ===\n")

    ts0, om, rain, soil = load_wet_weather_data()
    feat = build_dry_day_frame(ts0, om, rain, soil)

    # Select only dry days
    dry_feat = select_dry_days(feat, rain)
    print("Selected {} dry days from {} total".format(len(dry_feat), len(feat)))

    # Drop incomplete
    complete, n_dropped = tsm.drop_incomplete(dry_feat, cols=DRY_FEATURES)
    print("After dropping NaN: {} complete dry days\n".format(len(complete)))

    # Use simpler hyperparameters for dry model (less data, simpler problem)
    params = {
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 1.0,
        'min_child_weight': 2
    }
    n_est = 50

    dates = complete.index.to_series().reset_index(drop=True)
    feat_idx = complete.index
    ctx = scorecard.load_context()
    oos_parts = []

    print("Cross-validation on dry days:")
    for period, train_mask, val_mask in model_eval.get_cv_fold_masks(dates):
        train_idx = feat_idx[train_mask.values]
        val_idx = feat_idx[val_mask.values]
        pred, m = fit_predict(complete, DRY_FEATURES, train_idx, val_idx, params, n_est)
        s = model_eval.score(val_idx, complete.loc[val_idx, "flow_true"].values, pred)
        print("  {}: R²={:.3f}, MAE={:.3f}".format(period, s['R2'], s['MAE']))
        oos_parts.append(pd.DataFrame(
            {"Period": period, "y": complete.loc[val_idx, "flow_true"].values, "p": pred},
            index=val_idx))

    # 2024 test
    print("Hold-out test 2024:")
    train_mask, test_mask = model_eval.get_train_test_masks(dates)
    train_idx = feat_idx[train_mask.values]
    test_idx = feat_idx[test_mask.values]
    pred_test, m_test = fit_predict(complete, DRY_FEATURES, train_idx, test_idx, params, n_est)
    s_test = model_eval.score(test_idx, complete.loc[test_idx, "flow_true"].values, pred_test)
    print("  TEST 2024: R²={:.3f}, MAE={:.3f}\n".format(s_test['R2'], s_test['MAE']))
    oos_parts.append(pd.DataFrame(
        {"Period": "TEST 2024", "y": complete.loc[test_idx, "flow_true"].values, "p": pred_test},
        index=test_idx))

    # Combine and score
    oos = pd.concat(oos_parts)
    path = OUT_DIR / "oos_predictions_dry_days.csv"
    oos.to_csv(path, index_label="Date")
    pooled = scorecard.run(path, "DryDayModel", ctx)

    print("Pooled Performance (dry days only):")
    for k in ["R2_All", "MAE_All", "RMSE_All", "Bias_HighFlow"]:
        print("  {}: {:.3f}".format(k, pooled[k]))

    # Fit final model on all dry days
    print("\nTraining production model...")
    m_prod = xgb.XGBRegressor(**params, n_estimators=n_est,
                              objective="reg:squarederror", random_state=42)
    wet_all = (complete["rain_wet_mask_t"] | complete["rain_wet_mask_t1"]).values
    m_prod.fit(complete[DRY_FEATURES].values, complete["target_excess"].values,
              sample_weight=np.where(wet_all, 1.5, 1.0))

    # Save outputs
    with open(OUT_DIR / "production_model.json", "w") as f:
        json.dump(m_prod.get_booster().get_dump(dump_format="json"), f, indent=2)

    with open(OUT_DIR / "production_features.json", "w") as f:
        json.dump({
            "feature_set": "dry_day_baseline",
            "features": DRY_FEATURES,
            "params": params,
            "n_estimators": n_est,
            "trained_on_days": int(len(complete)),
            "condition": "rainfall < 0.1 in AND 5-day_antecedent < 0.5 in",
            "prediction": "flow_MGD = recent_level + model output"
        }, f, indent=1)

    with open(OUT_DIR / "model_metadata.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training: Dry days only from 2018-2024 (n={len(complete):,})\n")
        f.write(f"Condition: rainfall < 0.1 in AND 5-day antecedent < 0.5 in\n")
        f.write(f"Features: {len(DRY_FEATURES)}\n")
        f.write(f"  - Flow history: lags 1-3 days\n")
        f.write(f"  - Temperature: min temp (affects infiltration)\n")
        f.write(f"  - Seasonal: day-of-year sine/cosine\n")
        f.write(f"Hyperparameters: {params}\n")
        f.write(f"N_Estimators: {n_est}\n")
        f.write(f"\nPooled OOS Performance (dry days, n={len(oos)}):\n")
        for k in ["R2_All", "MAE_All", "RMSE_All", "Bias_HighFlow"]:
            f.write(f"  {k}: {pooled[k]:.3f}\n")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

"""
Verify: does R4S2 still beat Phase 6 when BOTH are trained on the full 2,017 rows
(not the 1,214 common rows from the earlier test)?
"""

import pandas as pd
import numpy as np
import xgboost as xgb

import model_eval
import scorecard
import train_phase6_model as p6
import train_screened_model as tsm
import wet_weather_experiments as wx

def fit_predict(f, cols, train_idx, pred_idx, params, n_estimators):
    """Fit XGBoost with wet-day weights and predict."""
    tr = f.loc[train_idx]
    wet = (tr["rain_wet_mask_t"] | tr["rain_wet_mask_t1"]).values
    m = xgb.XGBRegressor(**params, n_estimators=n_estimators, objective="reg:squarederror", random_state=42)
    m.fit(tr[cols].values, tr["target_excess"].values,
          sample_weight=np.where(wet, tsm.WET_DAY_SAMPLE_WEIGHT, 1.0))
    pr = f.loc[pred_idx]
    return pr["recent_level"].values + m.predict(pr[cols].values), m

# Load and build Phase 6 baseline
ts0, om, rain, screen_rain = p6.load_inputs()
feat_p6, _, _ = p6.build_frame(ts0, om, rain, screen_rain)
complete_p6, _ = tsm.drop_incomplete(feat_p6, cols=p6.FEATURE_SETS["gauge_rain_x_antecedent"])
print(f"Phase 6 complete rows: {len(complete_p6):,}")

# Load and build Phase 7 (R4S2)
soil = pd.read_csv(wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv", parse_dates=["Date"]).set_index("Date")
feat_p7, _, _ = p6.build_frame(ts0, om, rain, screen_rain)  # Start with Phase 6 frame
r = ts0.assign(Rainfall_in_Corrected=rain)["Rainfall_in_Corrected"]
feat_p7["ante_4d"] = r.rolling(4, min_periods=4).sum().shift(1)
feat_p7.drop(columns=["ante_30d"], inplace=True)
feat_p7["rain_x_ante4"] = feat_p7["rainfall_t"] * feat_p7["ante_4d"]
feat_p7["rain_lag1_x_ante4"] = feat_p7["rain_lag1"] * feat_p7["ante_4d"]
soil_2d = soil[["sm_0_7cm", "sm_7_28cm"]].reindex(ts0.index)
feat_p7["sm_0_7cm_ante2"] = soil_2d["sm_0_7cm"].rolling(2, min_periods=2).mean().shift(1)
feat_p7["sm_7_28cm_ante2"] = soil_2d["sm_7_28cm"].rolling(2, min_periods=2).mean().shift(1)

p7_cols = list(p6.FEATURE_SETS["gauge_rain_x_antecedent"])
if "ante_30d" in p7_cols:
    p7_cols.remove("ante_30d")
p7_cols.extend(["ante_4d", "rain_x_ante4", "rain_lag1_x_ante4", "sm_0_7cm_ante2", "sm_7_28cm_ante2"])

complete_p7, _ = tsm.drop_incomplete(feat_p7, cols=p7_cols)
print(f"Phase 7 complete rows: {len(complete_p7):,}")

# Find COMMON rows between both
common_idx = complete_p6.index.intersection(complete_p7.index)
print(f"Common rows between P6 and P7: {len(common_idx):,}")

# Fit on common rows
params, n_est = wx.shipped_params()
dates_common = pd.Series(range(len(common_idx)), index=common_idx)
ctx = scorecard.load_context()

print("\nScoring on common rows:")
oos_p6 = []
oos_p7 = []

for period, train_mask, val_mask in model_eval.get_cv_fold_masks(dates_common.reset_index(drop=True)):
    train_idx = common_idx[train_mask.values]
    val_idx = common_idx[val_mask.values]

    pred_p6, _ = fit_predict(complete_p6, p6.FEATURE_SETS["gauge_rain_x_antecedent"], train_idx, val_idx, params, n_est)
    pred_p7, _ = fit_predict(complete_p7, p7_cols, train_idx, val_idx, params, n_est)

    oos_p6.append(pd.DataFrame({"Period": period, "y": complete_p6.loc[val_idx, "flow_true"].values, "p": pred_p6}, index=val_idx))
    oos_p7.append(pd.DataFrame({"Period": period, "y": complete_p7.loc[val_idx, "flow_true"].values, "p": pred_p7}, index=val_idx))

# Test set
train_mask, test_mask = model_eval.get_train_test_masks(dates_common.reset_index(drop=True))
train_idx = common_idx[train_mask.values]
test_idx = common_idx[test_mask.values]

pred_p6, _ = fit_predict(complete_p6, p6.FEATURE_SETS["gauge_rain_x_antecedent"], train_idx, test_idx, params, n_est)
pred_p7, _ = fit_predict(complete_p7, p7_cols, train_idx, test_idx, params, n_est)

oos_p6.append(pd.DataFrame({"Period": "TEST 2024", "y": complete_p6.loc[test_idx, "flow_true"].values, "p": pred_p6}, index=test_idx))
oos_p7.append(pd.DataFrame({"Period": "TEST 2024", "y": complete_p7.loc[test_idx, "flow_true"].values, "p": pred_p7}, index=test_idx))

oos_p6 = pd.concat(oos_p6)
oos_p7 = pd.concat(oos_p7)

p6_path = "output/23_phase7_soil_antecedent/verify_p6_common_rows.csv"
p7_path = "output/23_phase7_soil_antecedent/verify_p7_common_rows.csv"
oos_p6.to_csv(p6_path, index_label="Date")
oos_p7.to_csv(p7_path, index_label="Date")

h_p6 = scorecard.run(p6_path, "Phase6_CommonRows", ctx)
h_p7 = scorecard.run(p7_path, "Phase7_CommonRows", ctx)

print(f"\nPhase 6 (common rows): R²={h_p6['R2_All']:.3f}, MAE={h_p6['MAE_All']:.3f}, Wet R²={h_p6['R2_Wet']:.3f}")
print(f"Phase 7 (common rows): R²={h_p7['R2_All']:.3f}, MAE={h_p7['MAE_All']:.3f}, Wet R²={h_p7['R2_Wet']:.3f}")
print(f"Difference:           R²={h_p7['R2_All']-h_p6['R2_All']:+.3f}, MAE={h_p7['MAE_All']-h_p6['MAE_All']:+.3f}")

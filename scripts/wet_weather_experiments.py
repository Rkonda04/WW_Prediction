"""
Wet-weather input experiments on the Phase 5 pipeline.

Tests the two inputs the internal review asked about -- rainfall from nearby
gauges and soil moisture -- by changing ONE thing at a time on the shipped
Phase 5 pipeline (scripts/train_screened_model.py):

  * the no-rain spike screen is fixed to the shipped model's flagged days,
    so every variant is trained and scored on the same days;
  * XGBoost hyperparameters and tree count are fixed to the shipped model's
    CV selection (output/13_screened_model/cv_selection.csv);
  * every variant is scored pooled out-of-sample (2020, 2021, 2022 by
    expanding-window CV + 2024 held out) through scripts/scorecard.py, on
    the intersection of rows every variant can predict.

Variant V0 reproduces the shipped model and is the reference.

Inputs: output/21_wet_weather_review/{gauge_network_daily,era5_soil_moisture_daily}.csv
        (run scripts/fetch_wet_weather_inputs.py first)
Outputs: output/21_wet_weather_review/experiments/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import model_eval
import scorecard
import train_screened_model as tsm

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEW_DIR = BASE_DIR / "output" / "21_wet_weather_review"
OUT_DIR = REVIEW_DIR / "experiments"
NETWORK_CSV = REVIEW_DIR / "gauge_network_daily.csv"
SOIL_CSV = REVIEW_DIR / "era5_soil_moisture_daily.csv"
CV_SELECTION_CSV = BASE_DIR / "output" / "13_screened_model" / "cv_selection.csv"

BASE_FEATURES = list(tsm.FEATURE_COLUMNS)
INTERACTION_FEATURES = ["rain_x_ante30", "rain_lag1_x_ante30"]
PARAM_KEYS = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "min_child_weight"]


def add_interactions(f: pd.DataFrame) -> pd.DataFrame:
    """Storm response scales with how wet the catchment already is; depth-2 trees struggle to learn that product."""
    f["rain_x_ante30"] = f["rainfall_t"] * f["ante_30d"]
    f["rain_lag1_x_ante30"] = f["rain_lag1"] * f["ante_30d"]
    return f


def shipped_params():
    cv = pd.read_csv(CV_SELECTION_CSV)
    best = cv.loc[cv["Avg_CV_RMSE"].idxmin()]
    params = {k: best[k] for k in PARAM_KEYS}
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    return params, int(best["N_Estimators"])


def build_variants():
    ts0, om, rain_base = tsm.load_inputs()
    ts, flagged, _ = tsm.apply_screen(ts0, om, rain_base)
    net = pd.read_csv(NETWORK_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts.index)
    soil = pd.read_csv(SOIL_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts.index)

    has_net = net["network_n_gauges"] > 0
    net_mean = net["network_mean_in"].where(has_net, rain_base)
    net_max = net["network_max_in"].where(has_net, rain_base)

    def frame(rain, extra=None):
        f = tsm.build_feature_frame(ts.assign(Rainfall_in_Corrected=rain), om, ts0["Total_Treated_MGD"])
        cols = list(BASE_FEATURES)
        for name, series in (extra or {}).items():
            f[name] = series
            cols.append(name)
        return f, cols

    soil_extra = {"sm_0_7cm": soil["sm_0_7cm"], "sm_7_28cm": soil["sm_7_28cm"]}
    variants = {
        "V0_shipped_rain": ("max(plant, Open-Meteo) -- shipped Phase 5 inputs", *frame(rain_base)),
        "V1_network_mean_rain": ("Nearby-gauge network mean replaces the rain input", *frame(net_mean)),
        "V2_max_plant_om_network": ("max(plant, Open-Meteo, network mean)", *frame(np.fmax(rain_base, net_mean))),
        "V3_add_network_max_feature": ("Shipped rain + network max (localized storm) as extra features",
                                       *frame(rain_base, {"net_max_t": net_max, "net_max_lag1": net_max.shift(1)})),
        "V4_add_soil_moisture": ("Shipped rain + ERA5 soil moisture (0-7, 7-28 cm) as extra features",
                                 *frame(rain_base, soil_extra)),
        "V5_network_mean_plus_soil": ("Network mean rain + ERA5 soil moisture", *frame(net_mean, soil_extra)),
    }
    f6, cols6 = frame(net_mean)
    variants["V6_network_mean_rain_x_antecedent"] = (
        "Network mean rain + rain x 30-day antecedent rain interactions", add_interactions(f6), cols6 + INTERACTION_FEATURES)
    coverage = {"days": int(len(ts)), "network_days": int(has_net.sum()),
                "network_share": float(has_net.loc["2018":"2024"].mean()), "screened_days": int(flagged.sum())}
    return variants, coverage


def fit_predict(f, cols, train_idx, pred_idx, params, n_estimators):
    tr = f.loc[train_idx]
    wet = (tr["rain_wet_mask_t"] | tr["rain_wet_mask_t1"]).values
    m = xgb.XGBRegressor(**params, n_estimators=n_estimators, objective="reg:squarederror", random_state=42)
    m.fit(tr[cols].values, tr["target_excess"].values,
          sample_weight=np.where(wet, tsm.WET_DAY_SAMPLE_WEIGHT, 1.0))
    pr = f.loc[pred_idx]
    return pr["recent_level"].values + m.predict(pr[cols].values), m


def pooled_oos(f, cols, rows, params, n_estimators):
    dates = rows.to_series().reset_index(drop=True)
    parts = []
    splits = [(f"{name} val {rows[vm.values].year.min()}", rows[tm.values], rows[vm.values])
              for name, tm, vm in model_eval.get_cv_fold_masks(dates)]
    tr, te = model_eval.get_train_test_masks(dates)
    splits.append(("TEST 2024", rows[tr.values], rows[te.values]))
    for period, tr_idx, va_idx in splits:
        p, _ = fit_predict(f, cols, tr_idx, va_idx, params, n_estimators)
        parts.append(pd.DataFrame({"Period": period, "y": f.loc[va_idx, "flow_true"].values, "p": p}, index=va_idx))
    return pd.concat(parts)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, n_estimators = shipped_params()
    print(f"Fixed params {params}, n_estimators={n_estimators}")
    variants, cov = build_variants()
    print(f"Gauge network reports on {cov['network_share']:.1%} of 2018-2024 days; "
          f"{cov['screened_days']} screened days held fixed\n")

    common = None
    for _, (_, f, cols) in variants.items():
        ok = f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)
        common = ok if common is None else common & ok
    rows = common[common].index
    print(f"Common complete rows across all variants: {len(rows):,}\n")

    context = scorecard.load_context()
    results = []
    for name, (desc, f, cols) in variants.items():
        oos = pooled_oos(f, cols, rows, params, n_estimators)
        path = OUT_DIR / f"oos_predictions_{name}.csv"
        oos.to_csv(path, index_label="Date")
        h = scorecard.run(path, name, context)
        results.append({"Variant": name, "Description": desc, "N_Features": len(cols), **h})
        print()

    res = pd.DataFrame(results)
    keep = ["Variant", "Description", "N_Features", "N_Days", "MAE_All", "RMSE_All", "R2_All", "MAE_Dry",
            "MAE_Wet", "Bias_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow", "Wet_Excess_Captured_Pct",
            "CDF_Gap_P90_Wet", "CDF_Gap_P99_Wet"]
    res = res[keep]
    ref = res.iloc[0]
    for k in ("MAE_All", "MAE_Wet", "MAE_HighFlow", "Bias_HighFlow", "R2_Wet", "CDF_Gap_P99_Wet"):
        res[f"d_{k}_vs_V0"] = res[k] - ref[k]
    res.to_csv(OUT_DIR / "experiment_comparison.csv", index=False)

    pd.set_option("display.width", 250)
    print(res[["Variant", "MAE_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow",
               "Wet_Excess_Captured_Pct", "CDF_Gap_P99_Wet"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

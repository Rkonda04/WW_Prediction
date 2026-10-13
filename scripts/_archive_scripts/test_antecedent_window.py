"""
Test antecedent rainfall windows: 2, 3, 4, 5 days vs. Phase 6's 30 days.

Hypothesis: recent saturation (2-5 days back) may be more predictive than
the full 30-day window for immediate flow response.

Keep everything else identical to Phase 6: gauge network rain, same hyperparameters,
same CV, same screen, same scoring.

Outputs: output/22_phase6_gauge_network/antecedent_window_comparison/
"""

import pandas as pd
import numpy as np

import scorecard
import train_phase6_model as p6
import train_screened_model as tsm
import wet_weather_experiments as wx

OUT_DIR = p6.OUT_DIR / "antecedent_window_comparison"


def build_features_with_ante(ts0, om, rain, ante_days):
    """Build features with specified antecedent window instead of 30 days."""
    _, flagged, _ = tsm.apply_screen(ts0, om, rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan

    # Build base feature frame
    feat = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])

    # Replace ante_30d with the custom window
    r = ts["Rainfall_in_Corrected"]
    feat["ante_ndays"] = r.rolling(ante_days, min_periods=ante_days).sum().shift(1)

    # Build interactions with the new antecedent
    feat["rain_x_ante"] = feat["rainfall_t"] * feat["ante_ndays"]
    feat["rain_lag1_x_ante"] = feat["rain_lag1"] * feat["ante_ndays"]

    return feat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, n_est = wx.shipped_params()
    ts0, om, rain, screen_rain = p6.load_inputs()

    # Load CV and test split (same as Phase 6)
    ctx = scorecard.load_context()

    variants = {}

    # Phase 6 baseline (30 days)
    feat_p6, _, _ = p6.build_frame(ts0, om, rain, screen_rain)
    variants["P6_ante30"] = ("Phase 6 baseline (30-day antecedent)", feat_p6, p6.FEATURE_SETS["gauge_rain_x_antecedent"])

    # Test 2, 3, 4, 5 days
    for ante_days in [2, 3, 4, 5]:
        feat = build_features_with_ante(ts0, om, rain, ante_days)
        cols = list(tsm.FEATURE_COLUMNS)
        # Replace ante_30d, and add rain x ante interactions
        if "ante_30d" in cols:
            cols.remove("ante_30d")
        cols.extend(["ante_ndays", "rain_x_ante", "rain_lag1_x_ante"])
        variants[f"ante{ante_days}"] = (f"Gauge network + {ante_days}-day antecedent interactions", feat, cols)

    # Find common rows
    ok = None
    for _, f, cols in variants.values():
        m = f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)
        ok = m if ok is None else ok & m
    rows = ok[ok].index
    print(f"Common rows: {len(rows):,}\n")

    results = []
    for name, (desc, f, cols) in variants.items():
        oos = wx.pooled_oos(f, cols, rows, params, n_est)
        path = OUT_DIR / f"oos_{name}.csv"
        oos.to_csv(path, index_label="Date")
        h = scorecard.run(path, name, ctx)
        results.append({"Variant": name, "Description": desc, "N_Features": len(cols), **h})
        print(f"{name}: R²={h['R2_All']:.3f}, MAE={h['MAE_All']:.3f}, Wet R²={h['R2_Wet']:.3f}, High-flow MAE={h['MAE_HighFlow']:.3f}")

    res = pd.DataFrame(results)
    keep = ["Variant", "N_Features", "N_Days", "MAE_All", "R2_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]
    res[keep].to_csv(OUT_DIR / "antecedent_window_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + res[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

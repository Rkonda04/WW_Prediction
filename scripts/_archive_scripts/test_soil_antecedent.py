"""
Test antecedent soil moisture windows (2-5 days) with antecedent rain windows (2-5 days).

Use the multi-source soil moisture (averaged over 3x3 grids from ERA5, ERA5-Land, ECMWF).
Instead of same-day soil, use rolling 2-5 day averages (antecedent soil saturation).

Combine with the winning antecedent rain configuration (5-day window from test_antecedent_window.py).

Outputs: output/22_phase6_gauge_network/soil_antecedent_comparison/
"""

import pandas as pd
import numpy as np

import scorecard
import train_phase6_model as p6
import train_screened_model as tsm
import wet_weather_experiments as wx

OUT_DIR = p6.OUT_DIR / "soil_antecedent_comparison"


def build_features_with_soil_ante(ts0, om, rain, soil_multi, rain_ante_days, soil_ante_days):
    """Build features with antecedent rain and antecedent soil moisture."""
    _, flagged, _ = tsm.apply_screen(ts0, om, rain)
    ts = ts0.assign(Rainfall_in_Corrected=rain)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan

    # Build base feature frame
    feat = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])

    # Replace ante_30d with custom rain antecedent window
    r = ts["Rainfall_in_Corrected"]
    feat["ante_rain"] = r.rolling(rain_ante_days, min_periods=rain_ante_days).sum().shift(1)

    # Build rain x antecedent interactions
    feat["rain_x_ante"] = feat["rainfall_t"] * feat["ante_rain"]
    feat["rain_lag1_x_ante"] = feat["rain_lag1"] * feat["ante_rain"]

    # Add antecedent soil moisture (rolling average of past N days)
    soil = soil_multi[["sm_0_7cm", "sm_7_28cm"]].reindex(ts.index)
    for c in soil:
        feat[f"{c}_ante{soil_ante_days}"] = soil[c].rolling(soil_ante_days, min_periods=soil_ante_days).mean().shift(1)

    return feat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, n_est = wx.shipped_params()
    ts0, om, rain, screen_rain = p6.load_inputs()

    # Load multi-source soil data
    soil_csv = wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv"
    soil_multi = pd.read_csv(soil_csv, parse_dates=["Date"]).set_index("Date")

    ctx = scorecard.load_context()
    variants = {}

    # Baseline: Phase 6 with 30-day rain antecedent, no soil
    feat_p6, _, _ = p6.build_frame(ts0, om, rain, screen_rain)
    variants["P6_baseline"] = ("Phase 6 (30-day rain ante, no soil)", feat_p6, p6.FEATURE_SETS["gauge_rain_x_antecedent"])

    # Test combinations: rain ante (2-5) × soil ante (2-5)
    for rain_ante in [2, 3, 4, 5]:
        for soil_ante in [2, 3, 4, 5]:
            feat = build_features_with_soil_ante(ts0, om, rain, soil_multi, rain_ante, soil_ante)
            cols = list(tsm.FEATURE_COLUMNS)
            if "ante_30d" in cols:
                cols.remove("ante_30d")
            cols.extend([
                "ante_rain", "rain_x_ante", "rain_lag1_x_ante",
                f"sm_0_7cm_ante{soil_ante}", f"sm_7_28cm_ante{soil_ante}"
            ])
            name = f"R{rain_ante}S{soil_ante}"
            desc = f"Rain ante {rain_ante}d + Soil ante {soil_ante}d"
            variants[name] = (desc, feat, cols)

    # Find common rows
    ok = None
    for _, f, cols in variants.values():
        m = f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)
        ok = m if ok is None else ok & m
    rows = ok[ok].index
    print(f"Common rows: {len(rows):,}\n")

    results = []
    for name, (desc, f, cols) in sorted(variants.items()):
        oos = wx.pooled_oos(f, cols, rows, params, n_est)
        path = OUT_DIR / f"oos_{name}.csv"
        oos.to_csv(path, index_label="Date")
        h = scorecard.run(path, name, ctx)
        results.append({"Variant": name, "Description": desc, "N_Features": len(cols), **h})
        print(f"{name}: R²={h['R2_All']:.3f}, MAE={h['MAE_All']:.3f}, Wet R²={h['R2_Wet']:.3f}, High-flow MAE={h['MAE_HighFlow']:.3f}")

    res = pd.DataFrame(results)
    keep = ["Variant", "N_Features", "N_Days", "MAE_All", "R2_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]
    res[keep].to_csv(OUT_DIR / "soil_antecedent_comparison.csv", index=False)
    pd.set_option("display.width", 300)
    print("\n" + res[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

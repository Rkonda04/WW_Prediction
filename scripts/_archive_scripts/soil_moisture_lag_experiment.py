"""
Re-test of ERA5 soil moisture using PRIOR-day values instead of same-day.

wet_weather_experiments.py (V4/V5) fed same-day soil moisture, which already contains
the day's rain and so is redundant with the rain features. Here soil moisture enters only
as of yesterday (and optionally 2 days back / a 7-day mean), on top of both the shipped
rain input and the Phase 6 inputs (network mean rain + rain x antecedent interactions).
Everything else (screen, hyperparameters, CV folds, scoring) is identical to
wet_weather_experiments.py.

Outputs: output/21_wet_weather_review/experiments_soil_lag/
"""

import pandas as pd
import requests

import scorecard
import wet_weather_experiments as wx

OUT_DIR = wx.REVIEW_DIR / "experiments_soil_lag"
MULTI_CSV = wx.REVIEW_DIR / "soil_moisture_multisource_daily.csv"
SM_MODELS = ["era5", "era5_land", "ecmwf_ifs"]
GRID_OFFSETS = [-0.1, 0.0, 0.1]  # 3x3 points ~6 mi apart around the plant, like the gauge network radius
CENTER_LAT, CENTER_LON = 29.5838, -95.7552


def fetch_multisource():
    """Soil moisture averaged over a 3x3 grid for each model, then across the three models."""
    if MULTI_CSV.exists():
        return pd.read_csv(MULTI_CSV, parse_dates=["Date"]).set_index("Date")
    lats = [CENTER_LAT + d for d in GRID_OFFSETS for _ in GRID_OFFSETS]
    lons = [CENTER_LON + d for _ in GRID_OFFSETS for d in GRID_OFFSETS]
    per_model = {}
    for model in SM_MODELS:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
                                 "start_date": "2018-01-01", "end_date": "2024-12-31", "models": model,
                                 "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm",
                                 "timezone": "America/Chicago"}, timeout=300)
        r.raise_for_status()
        frames = []
        for loc in r.json():
            h = loc["hourly"]
            frames.append(pd.DataFrame({"sm_0_7cm": h["soil_moisture_0_to_7cm"], "sm_7_28cm": h["soil_moisture_7_to_28cm"]},
                                       index=pd.to_datetime(h["time"])))
        grid_mean = pd.concat(frames).groupby(level=0).mean().resample("D").mean()
        print(f"  {model}: {len(r.json())} grid points")
        for c in grid_mean:
            per_model[f"{c}_{model}"] = grid_mean[c]
    df = pd.DataFrame(per_model)
    for c in ("sm_0_7cm", "sm_7_28cm"):
        df[c] = df[[f"{c}_{m}" for m in SM_MODELS]].mean(axis=1)
    df.to_csv(MULTI_CSV, index_label="Date")
    return df


def soil_sets(soil, multi):
    sm = soil[["sm_0_7cm", "sm_7_28cm"]]
    mm = multi[["sm_0_7cm", "sm_7_28cm"]]
    return {
        "multi_same_day": {f"{c}_multi": mm[c] for c in mm},
        "multi_lag1": {f"{c}_multi_lag1": mm[c].shift(1) for c in mm},
        "multi_avg_today_yday": {f"{c}_multi_avg_t_t1": (mm[c] + mm[c].shift(1)) / 2 for c in mm},
        "same_day": {"sm_0_7cm": sm["sm_0_7cm"], "sm_7_28cm": sm["sm_7_28cm"]},
        "lag1": {"sm_0_7cm_lag1": sm["sm_0_7cm"].shift(1), "sm_7_28cm_lag1": sm["sm_7_28cm"].shift(1)},
        "avg_today_yday": {f"{c}_avg_t_t1": (sm[c] + sm[c].shift(1)) / 2 for c in sm},
        "lag1_lag2": {**{f"{c}_lag{k}": sm[c].shift(k) for c in sm for k in (1, 2)}},
        "lag1_mean7": {"sm_0_7cm_lag1": sm["sm_0_7cm"].shift(1), "sm_7_28cm_lag1": sm["sm_7_28cm"].shift(1),
                       "sm_7_28cm_mean7_lag1": sm["sm_7_28cm"].shift(1).rolling(7).mean()},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params, n_est = wx.shipped_params()
    ts0, om, rain_base = wx.tsm.load_inputs()
    ts, _, _ = wx.tsm.apply_screen(ts0, om, rain_base)
    net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts.index)
    soil = pd.read_csv(wx.SOIL_CSV, parse_dates=["Date"]).set_index("Date").reindex(ts.index)
    net_mean = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_base)

    def frame(rain, extra=None, interactions=False):
        f = wx.tsm.build_feature_frame(ts.assign(Rainfall_in_Corrected=rain), om, ts0["Total_Treated_MGD"])
        cols = list(wx.BASE_FEATURES)
        if interactions:
            f = wx.add_interactions(f)
            cols += wx.INTERACTION_FEATURES
        for k, s in (extra or {}).items():
            f[k] = s
            cols.append(k)
        return f, cols

    variants = {"A0_shipped": ("Shipped rain", *frame(rain_base)),
                "B0_phase6": ("Phase 6 (net mean + interactions)", *frame(net_mean, interactions=True))}
    multi = fetch_multisource().reindex(ts.index)
    print("Correlation of each model with the multi-source mean (7-28 cm):")
    print(multi[[f"sm_7_28cm_{m}" for m in SM_MODELS]].corrwith(multi["sm_7_28cm"]).round(3).to_string())
    for name, extra in soil_sets(soil, multi).items():
        variants[f"A_{name}"] = (f"Shipped rain + soil {name}", *frame(rain_base, extra))
        variants[f"B_{name}"] = (f"Phase 6 + soil {name}", *frame(net_mean, extra, interactions=True))

    ok = None
    for _, f, cols in variants.values():
        m = f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)
        ok = m if ok is None else ok & m
    rows = ok[ok].index
    print(f"Common rows: {len(rows):,}")

    ctx = scorecard.load_context()
    out = []
    for name, (desc, f, cols) in variants.items():
        oos = wx.pooled_oos(f, cols, rows, params, n_est)
        p = OUT_DIR / f"oos_{name}.csv"
        oos.to_csv(p, index_label="Date")
        out.append({"Variant": name, "N_Features": len(cols), **scorecard.run(p, name, ctx)})
    res = pd.DataFrame(out)
    keep = ["Variant", "N_Features", "N_Days", "MAE_All", "R2_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]
    res[keep].to_csv(OUT_DIR / "soil_lag_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

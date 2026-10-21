"""
RRWWTF Flow Model -- does MRMS radar/gauge rainfall change Phase 6 skill?

MRMS daily rain (gauge-corrected multi-sensor QPE) comes from the Iowa
Environmental Mesonet IEMRE point service (`mrms_precip_in`), sampled at a
3 x 3 grid of points +/-0.1 deg around the plant and averaged. One request
per point per year, 2017-12 to 2024-08.

IEM reports MRMS by calendar day; the gauge network is already aligned to the
plant day. The day shift for MRMS is chosen by correlation with the gauge
network on 2018-2022 only -- never against flow.

Variants (everything except the rain input is Phase 6 as shipped: same spike
screen, same rows, same folds, hyperparameters held at the Phase 6 selection):
  P6   gauge-network mean rain (reproduces Phase 6)
  M1   MRMS area mean replaces the gauge network
  M2   average of gauge network and MRMS
  M3   gauge network rain + MRMS today / yesterday as extra features

Outputs: output/25_mrms_rain/
  mrms_daily_9pt.csv            -- raw point series + area mean
  mrms_alignment.csv            -- shift selection
  variant_scores.csv            -- pooled out-of-sample and per-year R2 / MAE
  oos_<variant>.csv             -- pooled out-of-sample predictions

Usage:
    python scripts/mrms_rain_experiment.py
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import train_phase6_model as p6
import train_screened_model as tsm
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "25_mrms_rain"
MRMS_CSV = OUT_DIR / "mrms_daily_9pt.csv"
ALIGN_CSV = OUT_DIR / "mrms_alignment.csv"
SCORES_CSV = OUT_DIR / "variant_scores.csv"

CENTER_LAT, CENTER_LON = 29.5838, -95.7552  # Richmond COOP station, ~1 mi from the plant
OFFSETS = [-0.1, 0.0, 0.1]
YEARS = range(2017, 2025)
IEM_URL = "https://mesonet.agron.iastate.edu/iemre/multiday/{a}/{b}/{lat:.4f}/{lon:.4f}/json"
SHIFTS = [-1, 0, 1]


def fetch_mrms():
    if MRMS_CSV.exists():
        return pd.read_csv(MRMS_CSV, parse_dates=["Date"]).set_index("Date")
    cols = {}
    for dy in OFFSETS:
        for dx in OFFSETS:
            lat, lon = CENTER_LAT + dy, CENTER_LON + dx
            parts = []
            for y in YEARS:
                a = "2017-12-01" if y == 2017 else f"{y}-01-01"
                b = "2017-12-31" if y == 2017 else f"{y}-12-31"
                with urllib.request.urlopen(IEM_URL.format(a=a, b=b, lat=lat, lon=lon), timeout=60) as r:
                    data = json.load(r)["data"]
                parts.append(pd.Series({pd.Timestamp(x["date"]): x["mrms_precip_in"] for x in data}, dtype=float))
            cols[f"pt_{dy:+.1f}_{dx:+.1f}"] = pd.concat(parts)
            print(f"  point {lat:.3f}, {lon:.3f}: {cols[f'pt_{dy:+.1f}_{dx:+.1f}'].sum():.1f} in total")
    df = pd.DataFrame(cols).sort_index()
    df["mrms_mean_in"] = df.mean(axis=1, skipna=True)
    df.round(3).to_csv(MRMS_CSV, index_label="Date")
    return df


def align(mrms, net):
    rows = []
    for s in SHIFTS:
        m = mrms.shift(s).reindex(net.index)
        both = pd.concat([m, net], axis=1).loc["2018":"2022"].dropna()
        rows.append({"Shift_days": s, "N": len(both), "Pearson_vs_network": both.iloc[:, 0].corr(both.iloc[:, 1]),
                     "MRMS_total_in": both.iloc[:, 0].sum(), "Network_total_in": both.iloc[:, 1].sum()})
    a = pd.DataFrame(rows)
    a.to_csv(ALIGN_CSV, index=False)
    return a, int(a.loc[a["Pearson_vs_network"].idxmax(), "Shift_days"])


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def score(o, wet):
    rows = []
    groups = [("Pooled", o.index), ("Wet days", o.index[wet.reindex(o.index).values]),
              ("Dry days", o.index[~wet.reindex(o.index).values])]
    groups += [(p, o.index[o["Period"] == p]) for p in o["Period"].unique()]
    for name, idx in groups:
        s = o.loc[idx]
        rows.append({"Subset": name, "N": len(s), "R2": r2(s["y"].values, s["p"].values),
                     "MAE": float(np.abs(s["y"] - s["p"]).mean()), "Bias": float((s["p"] - s["y"]).mean())})
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("MRMS RAINFALL EXPERIMENT (Phase 6 protocol, rain input only changes)\n")
    ts0, om, net_rain, screen_rain = p6.load_inputs()
    mrms = fetch_mrms()["mrms_mean_in"]

    a, shift = align(mrms, net_rain)
    print("Day alignment vs gauge network, 2018-2022:\n" + a.round(3).to_string(index=False))
    print(f" -> using shift {shift:+d} day\n")
    mrms_rain = mrms.shift(shift).reindex(ts0.index)
    print(f"MRMS days missing in model period: {int(mrms_rain.loc['2018':].isna().sum())}\n")

    meta = json.loads((p6.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols6, params, n_est = meta["features"], meta["params"], meta["n_estimators"]

    variants = {}
    for key, desc, rain in [("P6", "Gauge-network mean rain (Phase 6 as shipped)", net_rain),
                            ("M1", "MRMS area mean replaces gauge network", mrms_rain.fillna(net_rain)),
                            ("M2", "Average of gauge network and MRMS", (net_rain + mrms_rain.fillna(net_rain)) / 2)]:
        f, _, _ = p6.build_frame(ts0, om, rain, screen_rain)
        variants[key] = (desc, f, list(cols6))
    f3, _, _ = p6.build_frame(ts0, om, net_rain, screen_rain)
    f3["mrms_t"], f3["mrms_lag1"] = mrms_rain, mrms_rain.shift(1)
    variants["M3"] = ("Gauge-network rain + MRMS today/yesterday as extra features", f3, list(cols6) + ["mrms_t", "mrms_lag1"])

    # One row set for every variant: days complete in all of them.
    need = ["recent_level", "flow_true", "target_excess"]
    rows = None
    for _, f, cols in variants.values():
        ok = f.index[f[cols + need].notna().all(axis=1)]
        rows = ok if rows is None else rows.intersection(ok)
    base = variants["P6"][1]
    wet = (base["rainfall_t"] > tsm.WET) | (base["rain_lag1"] > tsm.WET)  # wet defined by gauge rain for all variants
    print(f"Scored rows: {len(rows):,} (same for every variant)\n")

    all_rows = []
    for key, (desc, f, cols) in variants.items():
        o = wx.pooled_oos(f, cols, rows, params, n_est)
        o.to_csv(OUT_DIR / f"oos_{key}.csv", index_label="Date")
        for r in score(o, wet):
            all_rows.append({"Variant": key, "Description": desc, **r})
    s = pd.DataFrame(all_rows)
    s.to_csv(SCORES_CSV, index=False)

    for metric in ["R2", "MAE"]:
        print(f"{metric} by variant")
        print(s.pivot(index="Subset", columns="Variant", values=metric)
              .reindex(s["Subset"].unique())[list(variants)].round(3).to_string())
        print()
    print(f"Saved {SCORES_CSV.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

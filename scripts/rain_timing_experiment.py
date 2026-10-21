"""
RRWWTF -- refinement 3: hourly rainfall timing.

Rain late in a plant day reaches the plant partly in the next day's flow. This
tests that with the hourly gauge at Sugar Land Regional Airport (KSGR, ~8 mi
from the plant), from the Iowa Environmental Mesonet hourlyprecip service.

Plant-day boundary: the hour h0 at which each plant day starts is chosen by
correlation between KSGR rain summed over [h0 on D, h0 on D+1) and the gauge
network's plant-day rain, 2018-2022 only -- never against flow.

Features added to Phase 7 + log-slip screen (hyperparameters held):
  T1  ksgr_late_frac_t, ksgr_late_frac_lag1 -- share of the plant day's KSGR
      rain that fell in its last LATE_H hours (0 on days under 0.1 in)
  T2  T1 + ksgr_maxhr_t -- the day's largest hourly KSGR rain (intensity)
Base vs T1 vs T2 chosen by average CV RMSE on the 2018-2022 folds, then scored
pooled out-of-sample on one row set.

Outputs: output/27_refinements/
  ksgr_hourly.csv, ksgr_plant_day.csv, rain_timing_scores.csv

Usage:
    python scripts/rain_timing_experiment.py
"""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import log_slip_screen as ls
import peak_weighting_experiment as pw
import train_phase7_model as p7
import train_screened_model as tsm

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "27_refinements"
HOURLY_CSV = OUT_DIR / "ksgr_hourly.csv"
DAILY_CSV = OUT_DIR / "ksgr_plant_day.csv"
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/hourlyprecip.py?network=TX_ASOS&station=SGR"
       "&year1={y}&month1=1&day1=1&year2={y2}&month2=1&day2=1&tz=America/Chicago")
LATE_H = 8


def fetch_hourly() -> pd.Series:
    if not HOURLY_CSV.exists():
        parts = []
        for y in range(2017, 2025):
            with urllib.request.urlopen(URL.format(y=y, y2=y + 1), timeout=120) as r:
                parts.append(pd.read_csv(io.StringIO(r.read().decode("utf-8"))))
        h = pd.concat(parts)
        h["valid"] = pd.to_datetime(h["valid"])
        h = h.drop_duplicates("valid").set_index("valid")["precip_in"].astype(float)
        h.to_csv(HOURLY_CSV, index_label="valid")
    s = pd.read_csv(HOURLY_CSV, parse_dates=["valid"]).set_index("valid")["precip_in"]
    return s.asfreq("h")  # missing hours stay NaN


def plant_days(hourly: pd.Series, h0: int) -> pd.DataFrame:
    shifted = hourly.copy()
    shifted.index = shifted.index - pd.Timedelta(hours=h0)  # plant day D = [h0 on D, h0 on D+1)
    day = shifted.index.normalize()
    hour_in_day = shifted.index.hour
    g = pd.DataFrame({"r": shifted.values, "day": day, "late": hour_in_day >= 24 - LATE_H})
    out = pd.DataFrame({
        "total": g.groupby("day")["r"].sum(min_count=18),
        "late": g[g["late"]].groupby("day")["r"].sum(min_count=1),
        "maxhr": g.groupby("day")["r"].max(),
        "n_hours": g.groupby("day")["r"].count(),
    })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = fetch_hourly()
    print(f"KSGR hourly: {hourly.index.min()} to {hourly.index.max()}, {int(hourly.isna().sum()):,} missing hours")

    inputs = p7.load_inputs()
    ts0, om, net, mrms, screen = inputs
    corr = {}
    for h0 in range(24):
        d = plant_days(hourly, h0)["total"]
        both = pd.concat([d, net], axis=1).loc["2018":"2022"].dropna()
        corr[h0] = both.iloc[:, 0].corr(both.iloc[:, 1])
    h0 = max(corr, key=corr.get)
    print("Plant-day start hour vs gauge network r: " + ", ".join(f"{h}:{c:.2f}" for h, c in corr.items()))
    print(f" -> plant day starts {h0:02d}:00 local (r={corr[h0]:.3f})\n")

    k = plant_days(hourly, h0).reindex(ts0.index)
    frac = (k["late"] / k["total"]).where(k["total"] > tsm.WET, 0.0)
    k.assign(late_frac=frac).round(3).to_csv(DAILY_CSV, index_label="Date")

    log = pd.read_csv(tsm.PLANT_LOG_CSV, parse_dates=["Date"]).set_index("Date")
    flags = ls.slip_flags(log)
    f, _ = ls.frame(*inputs, extra_flag=pd.Series(True, index=flags.index))
    # Missing KSGR hours -> no timing information (neutral 0), not a dropped row.
    f["ksgr_late_frac_t"] = frac.fillna(0.0)
    f["ksgr_late_frac_lag1"] = frac.shift(1).fillna(0.0)
    f["ksgr_maxhr_t"] = k["maxhr"].fillna(0.0)

    meta = json.loads((p7.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols, params, n = meta["features"], meta["params"], meta["n_estimators"]
    variants = {"Base (Phase7+slip)": cols,
                "T1 late-rain share": cols + ["ksgr_late_frac_t", "ksgr_late_frac_lag1"],
                "T2 late share + max hourly": cols + ["ksgr_late_frac_t", "ksgr_late_frac_lag1", "ksgr_maxhr_t"]}
    need = variants["T2 late share + max hourly"] + ["recent_level", "flow_true", "target_excess"]
    rows = f.index[f[need].notna().all(axis=1)]
    print(f"Rows: {len(rows):,}; KSGR hours missing on {int(k['n_hours'].reindex(rows).lt(18).sum())} of them\n")

    res, cv = [], {}
    for name, c in variants.items():
        o, _ = pw.run(f, c, rows, params, n)
        cvp = o[o["Period"] != "TEST 2024"]
        cv[name] = float(np.mean([np.sqrt(((g["p"] - g["y"]) ** 2).mean()) for _, g in cvp.groupby("Period")]))
        res += pw.score(o, f, name, {"CV_RMSE_2018_2022": cv[name]})
    print("CV RMSE (2018-2022 folds): " + ", ".join(f"{k_}: {v:.4f}" for k_, v in cv.items()))
    print(f" -> CV picks: {min(cv, key=cv.get)}\n")
    s = pd.DataFrame(res)
    s.to_csv(OUT_DIR / "rain_timing_scores.csv", index=False)
    pd.set_option("display.width", 220)
    for metric in ["R2", "MAE", "Bias", "Within_0.3MGD_Pct"]:
        print(metric)
        print(s.pivot_table(index="Subset", columns="Variant", values=metric, sort=False).round(3).to_string())
        print()


if __name__ == "__main__":
    main()

"""
RRWWTF -- refinement 4: weekend / weekday.

Phase 7 dry-day errors differ by weekday (Kruskal-Wallis p=0.002): Friday is
over-predicted, Sunday under-predicted. Part of it is the reading schedule --
the Friday interval averages 25.0 h and the Sunday interval 23.0 h while raw
volumes are flat, so the 24-hour rescaling moves them apart. Either way the
pattern repeats weekly and is known in advance, so it is a fair input.

Variants (hyperparameters held at the Phase 7 selection), on two bases --
Phase 7 as shipped and Phase 7 + log-slip screen:
  D1  is_weekend (Sat, Sun)
  D2  dow (0 = Mon ... 6 = Sun)
  D3  dow + US federal holiday flag
Chosen by average CV RMSE on the 2018-2022 folds; scored pooled out-of-sample
on each base's own row set (every variant on a base shares its rows).

Outputs: output/27_refinements/weekday_scores.csv

Usage:
    python scripts/weekday_experiment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

import log_slip_screen as ls
import peak_weighting_experiment as pw
import train_phase7_model as p7
import train_screened_model as tsm

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "27_refinements"


def add_calendar(f):
    f = f.copy()
    f["dow"] = f.index.dayofweek.astype(float)
    f["is_weekend"] = (f.index.dayofweek >= 5).astype(float)
    hol = USFederalHolidayCalendar().holidays(f.index.min(), f.index.max())
    f["holiday"] = f.index.isin(hol).astype(float)
    return f


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = p7.load_inputs()
    log = pd.read_csv(tsm.PLANT_LOG_CSV, parse_dates=["Date"]).set_index("Date")
    flags = ls.slip_flags(log)
    meta = json.loads((p7.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols, params, n = meta["features"], meta["params"], meta["n_estimators"]
    variants = {"Base": cols, "D1 is_weekend": cols + ["is_weekend"], "D2 dow": cols + ["dow"],
                "D3 dow + holiday": cols + ["dow", "holiday"]}

    res = []
    for base_name, extra in [("Phase7", None), ("Phase7+slip", pd.Series(True, index=flags.index))]:
        f, _ = ls.frame(*inputs, extra_flag=extra)
        f = add_calendar(f)
        rows = f.index[f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)]
        cv = {}
        for name, c in variants.items():
            o, _ = pw.run(f, c, rows, params, n)
            cvp = o[o["Period"] != "TEST 2024"]
            cv[name] = float(np.mean([np.sqrt(((g["p"] - g["y"]) ** 2).mean()) for _, g in cvp.groupby("Period")]))
            for r in pw.score(o, f, name, {"Base": base_name, "CV_RMSE_2018_2022": cv[name]}):
                res.append(r)
            dry = ~((f.loc[o.index, "rainfall_t"] > tsm.WET) | (f.loc[o.index, "rain_lag1"] > tsm.WET))
            e = (o["p"] - o["y"])[dry.values]
            by = e.groupby(o.index[dry.values].dayofweek).mean()
            res.append({"Variant": name, "Base": base_name, "Subset": "Dry-day bias spread (max-min weekday)",
                        "N": int(dry.sum()), "Bias": float(by.max() - by.min())})
        print(f"{base_name}: CV RMSE " + ", ".join(f"{k}: {v:.4f}" for k, v in cv.items())
              + f" -> {min(cv, key=cv.get)}")
    s = pd.DataFrame(res)
    s.to_csv(OUT_DIR / "weekday_scores.csv", index=False)
    pd.set_option("display.width", 240)
    for metric in ["R2", "MAE", "Within_0.3MGD_Pct", "Bias"]:
        print(f"\n{metric}")
        print(s.pivot_table(index="Subset", columns=["Base", "Variant"], values=metric, sort=False).round(3).to_string())


if __name__ == "__main__":
    main()

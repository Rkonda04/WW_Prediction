"""
RRWWTF -- refinement 1: screen plant-log reading-time slips.

Plant flow is the totalizer difference between two daily readings, rescaled to
24 hours by the logged elapsed time (train_screened_model.apply_24h_normalization).
When one reading is taken (or logged) hours off schedule, the interval before it
is short and the one after it long. The rescaled pair then swings in opposite
directions (e.g. 2022-03-10 / 11: 5.53 then 0.72 MGD) while the two-day volume
is ordinary -- the water is real, its split between the two days is not.

Rules, defined from the plant log only (never from model error), applied to
every year:
  * slip pair   -- day D elapsed <= 24 - SLIP_H and day D+1 >= 24 + SLIP_H, or
                   the reverse (within the 12-36 h range the normalization uses):
                   flag D and D+1.
  * stuck total -- totalizer unchanged (0.000 MG): flag D and D+1 (the next
                   reading carries both days).
Flagged days are treated exactly like the existing no-rain spike screen: flow
set to missing before features are built.

Comparison (Phase 7 configuration and hyperparameters held fixed):
  A  Phase 7 as shipped
  B  Phase 7 + log-slip screen
Scored like for like on B's rows (does training on clean days help?), and each
on its own rows (what the headline number does -- not a like-for-like gain).

Outputs: output/27_refinements/
  log_slip_flagged_days.csv, log_slip_scores.csv

Usage:
    python scripts/log_slip_screen.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import model_eval
import train_phase7_model as p7
import train_screened_model as tsm
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "27_refinements"
SLIP_H = 4.0
SET = "gauge_rain_plus_mrms"


def slip_flags(log: pd.DataFrame) -> pd.DataFrame:
    el = log["elapsed"].where(log["elapsed"].between(*tsm.ELAPSED_VALID_RANGE))
    short, long_ = el <= 24 - SLIP_H, el >= 24 + SLIP_H
    pair = (short & long_.shift(-1, fill_value=False)) | (long_ & short.shift(-1, fill_value=False))
    stuck = log["total_treated"] == 0
    rule = pd.Series("", index=log.index)
    for m, name in ((pair, "slip_pair"), (stuck, "stuck_total")):
        hit = m | m.shift(1, fill_value=False)
        rule[hit & (rule == "")] = name
    out = log.loc[rule != "", ["total_treated", "elapsed", "norm24"]].assign(Rule=rule[rule != ""])
    nxt = log["total_treated"].shift(-1) + log["total_treated"]
    out["Two_Day_Mean_MGD"] = (nxt / (log["elapsed"] + log["elapsed"].shift(-1)) * 24).reindex(out.index)
    return out


def frame(ts0, om, net, mrms, screen, extra_flag=None):
    _, flagged, _ = tsm.apply_screen(ts0, om, screen)
    if extra_flag is not None:
        flagged = flagged | extra_flag.reindex(flagged.index, fill_value=False)
    ts = ts0.assign(Rainfall_in_Corrected=net)
    ts.loc[flagged, "Total_Treated_MGD"] = np.nan
    f = wx.add_interactions(tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"]))
    f["mrms_t"], f["mrms_lag1"] = mrms, mrms.shift(1)
    return f, flagged


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def metrics(o, name, rows_label):
    rows = []
    for sub, s in [("Pooled", o)] + [(p, o[o["Period"] == p]) for p in o["Period"].unique()]:
        e = s["p"] - s["y"]
        rows.append({"Model": name, "Scored_on": rows_label, "Subset": sub, "N": len(s), "R2": r2(s["y"].values, s["p"].values),
                     "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "Bias": e.mean(),
                     "Within_0.3MGD_Pct": (e.abs() <= 0.3).mean() * 100,
                     "Correct_Band_Pct": ((e >= -0.2) & (e <= 0.3)).mean() * 100,
                     "HighFlow_Bias": e[s["y"] >= 2.5].mean()})
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(tsm.PLANT_LOG_CSV, parse_dates=["Date"]).set_index("Date")
    flags = slip_flags(log)
    inputs = p7.load_inputs()
    ts0, om, net, mrms, screen = inputs
    rain = net.reindex(flags.index)
    flags["Rain_t_in"], flags["Rain_t-1_in"] = rain, net.shift(1).reindex(flags.index)
    flags["Storm_Day"] = (flags["Rain_t_in"] > tsm.WET) | (flags["Rain_t-1_in"] > tsm.WET)
    flags.round(3).to_csv(OUT_DIR / "log_slip_flagged_days.csv", index_label="Date")
    in_model = flags.loc["2018":"2024-08"]
    print(f"Log-slip screen: {len(in_model)} days flagged 2018-2024 "
          f"({in_model['Rule'].value_counts().to_dict()}); {int(in_model['Storm_Day'].sum())} are rain days")

    meta = json.loads((p7.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols, params, n_est = meta["features"], meta["params"], meta["n_estimators"]
    fa, fl_a = frame(*inputs)
    fb, fl_b = frame(*inputs, extra_flag=pd.Series(True, index=flags.index))
    print(f"Screened days: A {int(fl_a.sum())}, B {int(fl_b.sum())}")

    need = cols + ["recent_level", "flow_true", "target_excess"]
    rows_a = fa.index[fa[need].notna().all(axis=1)]
    rows_b = fb.index[fb[need].notna().all(axis=1)]
    common = rows_a.intersection(rows_b)
    print(f"Rows: A {len(rows_a):,}, B {len(rows_b):,}, common {len(common):,}\n")

    res = []
    # Like for like: both trained on their own rows, both scored on the common (clean) rows.
    oa = wx.pooled_oos(fa, cols, rows_a, params, n_est)
    ob = wx.pooled_oos(fb, cols, rows_b, params, n_est)
    res += metrics(oa.loc[oa.index.intersection(common)], "A_Phase7", "common clean rows")
    res += metrics(ob.loc[ob.index.intersection(common)], "B_Phase7+slip_screen", "common clean rows")
    res += metrics(oa, "A_Phase7", "own rows")
    res += metrics(ob, "B_Phase7+slip_screen", "own rows")
    s = pd.DataFrame(res)
    s.to_csv(OUT_DIR / "log_slip_scores.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    removed = oa.index.difference(ob.index)
    print(f"\nScored days in A but not in B: {len(removed)}; their A error: MAE "
          f"{(oa.loc[removed, 'p'] - oa.loc[removed, 'y']).abs().mean():.3f}")


if __name__ == "__main__":
    main()

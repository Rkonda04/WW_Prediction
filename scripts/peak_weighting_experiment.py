"""
RRWWTF -- refinement 2: make the model care more about peaks.

Base: Phase 7 configuration + log-slip screen (scripts/log_slip_screen.py),
hyperparameters held at the Phase 7 selection.

  (a) Peak weight: training days whose flow is >= PEAK_RATIO x the dry-weather
      level get sample weight W (on top of the existing wet-day weight).
      W in {1 (base), 2, 3, 5} chosen by average CV RMSE on the 2018-2022 folds.
  (b) Storm correction: on big-rain days (>= BIG_RAIN_IN today or yesterday)
      the predicted excess is scaled by (1 + beta). beta is fitted per fold on
      that fold's TRAINING days only (in-sample, one parameter).

Scored pooled out-of-sample on one row set. "Normal days" are rain-defined
(no rain > 0.1 in today or yesterday), never flow-defined.

Outputs: output/27_refinements/peak_weighting_scores.csv

Usage:
    python scripts/peak_weighting_experiment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import log_slip_screen as ls
import model_eval
import train_phase7_model as p7
import train_screened_model as tsm

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "27_refinements"
PEAK_RATIO = 1.3
WEIGHTS = [1, 2, 3, 5]
BIG_RAIN_IN = 0.5


def weights(f, idx, w_peak):
    tr = f.loc[idx]
    wet = (tr["rain_wet_mask_t"] | tr["rain_wet_mask_t1"]).values
    w = np.where(wet, tsm.WET_DAY_SAMPLE_WEIGHT, 1.0)
    peak = (tr["flow_true"] >= PEAK_RATIO * tr["recent_level"]).values
    return w * np.where(peak, w_peak, 1.0)


def fit(f, cols, idx, params, n, w_peak):
    m = xgb.XGBRegressor(**params, n_estimators=n, objective="reg:squarederror", random_state=42)
    m.fit(f.loc[idx, cols].values, f.loc[idx, "target_excess"].values, sample_weight=weights(f, idx, w_peak))
    return m


def big_rain(f, idx):
    return ((f.loc[idx, "rainfall_t"] >= BIG_RAIN_IN) | (f.loc[idx, "rain_lag1"] >= BIG_RAIN_IN)).values


def splits(rows):
    dates = rows.to_series().reset_index(drop=True)
    out = [(f"{name} val {rows[vm.values].year.min()}", rows[tm.values], rows[vm.values])
           for name, tm, vm in model_eval.get_cv_fold_masks(dates)]
    tr, te = model_eval.get_train_test_masks(dates)
    return out + [("TEST 2024", rows[tr.values], rows[te.values])]


def run(f, cols, rows, params, n, w_peak=1, correct=False):
    parts, betas = [], []
    for period, tr, va in splits(rows):
        m = fit(f, cols, tr, params, n, w_peak)
        ex_va = m.predict(f.loc[va, cols].values)
        beta = 0.0
        if correct:
            ex_tr = m.predict(f.loc[tr, cols].values)
            b = big_rain(f, tr)
            x, y = ex_tr[b], f.loc[tr, "target_excess"].values[b]
            beta = float(np.sum(x * (y - x)) / np.sum(x * x))  # least squares for y ~ (1 + beta) x
            ex_va = np.where(big_rain(f, va), (1 + beta) * ex_va, ex_va)
        betas.append(beta)
        parts.append(pd.DataFrame({"Period": period, "y": f.loc[va, "flow_true"].values,
                                   "p": f.loc[va, "recent_level"].values + ex_va}, index=va))
    return pd.concat(parts), betas


def score(o, f, name, extra):
    dry = ~((f.loc[o.index, "rainfall_t"] > tsm.WET) | (f.loc[o.index, "rain_lag1"] > tsm.WET))
    big = big_rain(f, o.index)
    rows = []
    for sub, m in [("All", np.ones(len(o), bool)), ("Dry days (rain-defined)", dry.values), ("Big-rain days", big),
                   ("High flow >= 2.5 MGD", (o["y"] >= 2.5).values),
                   *[(p, (o["Period"] == p).values) for p in o["Period"].unique()]]:
        s = o[m]; e = s["p"] - s["y"]
        rows.append({"Variant": name, **extra, "Subset": sub, "N": len(s), "R2": ls.r2(s["y"].values, s["p"].values),
                     "MAE": e.abs().mean(), "RMSE": float(np.sqrt((e ** 2).mean())), "Bias": e.mean(),
                     "Within_0.3MGD_Pct": (e.abs() <= 0.3).mean() * 100,
                     "Correct_Band_Pct": ((e >= -0.2) & (e <= 0.3)).mean() * 100})
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(tsm.PLANT_LOG_CSV, parse_dates=["Date"]).set_index("Date")
    flags = ls.slip_flags(log)
    inputs = p7.load_inputs()
    f, _ = ls.frame(*inputs, extra_flag=pd.Series(True, index=flags.index))
    meta = json.loads((p7.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols, params, n = meta["features"], meta["params"], meta["n_estimators"]
    rows = f.index[f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)]

    print("(a) peak weight -- CV RMSE on 2018-2022 folds only")
    cv = {}
    for w in WEIGHTS:
        o, _ = run(f, cols, rows, params, n, w_peak=w)
        cvp = o[o["Period"] != "TEST 2024"]
        cv[w] = float(np.mean([np.sqrt(((g["p"] - g["y"]) ** 2).mean()) for _, g in cvp.groupby("Period")]))
        print(f"  W={w}: avg CV RMSE {cv[w]:.4f}")
    w_best = min(cv, key=cv.get)
    print(f"  -> W={w_best}\n")

    res = []
    for name, w, corr in [("Base (Phase7+slip)", 1, False), (f"(a) peak weight W={w_best}", w_best, False),
                          ("(a) peak weight W=3 (fixed, for reference)", 3, False),
                          ("(b) storm correction", 1, True)]:
        o, betas = run(f, cols, rows, params, n, w_peak=w, correct=corr)
        res += score(o, f, name, {"Betas": ";".join(f"{b:.3f}" for b in betas) if corr else ""})
    s = pd.DataFrame(res)
    s.to_csv(OUT_DIR / "peak_weighting_scores.csv", index=False)
    pd.set_option("display.width", 220)
    for metric in ["R2", "MAE", "Bias", "Within_0.3MGD_Pct"]:
        print(f"{metric}")
        print(s.pivot_table(index="Subset", columns="Variant", values=metric, sort=False).round(3).to_string())
        print()
    print("Storm-correction beta per fold:", s.loc[s["Betas"] != "", "Betas"].iloc[0])


if __name__ == "__main__":
    main()

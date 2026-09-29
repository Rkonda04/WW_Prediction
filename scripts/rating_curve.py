"""
Rainfall-to-flow rating curve distilled from the flow model.

Answers "for a forecast of X inches tomorrow, what flow should the plant
expect?" as a simple table and chart.

Method (model-derived, i.e. what the tool itself would predict):
  1. Use the Phase 6 production model (output/22_phase6_gauge_network/
     production_model.json), fitted on every available year 2018-2024 --
     the model a forward prediction would use.
  2. Take every real day with a dry lead-in (no rain > 0.1 in on that day or
     the day before) as a starting state: its dry-weather level, recent flow,
     temperature, season and 30-day antecedent rain are all real.
  3. Inject a storm of R inches on that day, predict the day's flow; then roll
     one day forward with no further rain and predict the day after.
  4. Report the median added flow (prediction minus dry-weather level) across
     starting days, split by antecedent wetness (30-day prior rain), with the
     25th-75th percentile spread.

The added-flow form is used because the dry-weather level drifts by year
(median 1.41 MGD in 2020, 1.88 in 2024): predicted flow = today's dry-weather
level + added flow.

Observed storms (median observed added flow per rain bin) are plotted beside
the curve as a check. Tree models are step functions and cannot extrapolate:
above the largest storms in the record the curve goes flat, so depths beyond
the 99th-percentile rain day are flagged.

Outputs (output/21_wet_weather_review/rating_curve/):
    rating_curve.csv, rating_curve.xlsx, rating_curve.png
Usage (after scripts/train_phase6_model.py):
    python scripts/rating_curve.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

import train_phase6_model as p6
import wet_weather_experiments as wx

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "21_wet_weather_review" / "rating_curve"
RAIN_STEPS = np.round(np.arange(0, 5.01, 0.25), 2)
ANTECEDENT = [("Dry (< 1 in prior 30 d)", 0, 1.0), ("Normal (1-4 in)", 1.0, 4.0), ("Wet (> 4 in prior 30 d)", 4.0, np.inf)]
COLORS = ["#8fbbeb", "#2a78d6", "#0f3d75"]
OBS_BINS = [(0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 5.0)]
INK, INK_2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e4e1", "#fcfcfb", "#8a8985"


def storm(base: pd.DataFrame, cols, model, r: float):
    x = base[cols].copy()
    x["rainfall_t"] = r
    x["rain_sqrt_t"] = np.sqrt(r)
    added_t = model.predict(wx.add_interactions(x)[cols].values)

    y = base[cols].copy()
    y["rainfall_t"] = 0.0
    y["rain_sqrt_t"] = 0.0
    y["rain_lag2"] = base["rain_lag1"]
    y["rain_lag1"] = r
    y["flow_lag1"] = base["recent_level"] + added_t
    y["excess_lag3"] = base["excess_lag2"]
    y["excess_lag2"] = base["excess_lag1"]
    y["excess_lag1"] = added_t
    y["excess_mean3"] = (y["excess_lag1"] + y["excess_lag2"] + y["excess_lag3"]) / 3
    y["ante_5d"] = base["ante_5d"] + r
    y["ante_30d"] = base["ante_30d"] + r
    y["days_since_rain"] = 0.0 if r > wx.tsm.WET else np.minimum(base["days_since_rain"] + 1, 30)
    added_t1 = model.predict(wx.add_interactions(y)[cols].values)
    return added_t, added_t1


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((p6.OUT_DIR / "production_features.json").read_text(encoding="utf-8"))
    cols = meta["features"]
    model = xgb.XGBRegressor()
    model.load_model(p6.OUT_DIR / "production_model.json")
    f, _, _ = p6.build_frame(*p6.load_inputs())
    f = f[f[cols + ["recent_level", "flow_true", "target_excess"]].notna().all(axis=1)]

    base = f[(f["rainfall_t"] <= wx.tsm.WET) & (f["rain_lag1"] <= wx.tsm.WET)]
    p99 = float(f.loc[f["rainfall_t"] > wx.tsm.WET, "rainfall_t"].quantile(0.99))
    rmax = float(f["rainfall_t"].max())

    rows = []
    for name, lo, hi in ANTECEDENT:
        b = base[(base["ante_30d"] >= lo) & (base["ante_30d"] < hi)]
        for r in RAIN_STEPS:
            a0, a1 = storm(b, cols, model, r)
            rows.append({"Antecedent": name, "N_Start_Days": len(b), "Rain_in": r,
                         "Added_Flow_DayOf_MGD": np.median(a0), "DayOf_P25": np.percentile(a0, 25),
                         "DayOf_P75": np.percentile(a0, 75),
                         "Added_Flow_DayAfter_MGD": np.median(a1), "DayAfter_P25": np.percentile(a1, 25),
                         "DayAfter_P75": np.percentile(a1, 75),
                         "Beyond_Record_P99": r > p99})
    curve = pd.DataFrame(rows)
    level_2024 = float(f.loc["2024", "recent_level"].median())
    curve["Example_Total_DayOf_MGD_at_2024_level"] = level_2024 + curve["Added_Flow_DayOf_MGD"]

    wet = f[f["rainfall_t"] > wx.tsm.WET]
    obs = []
    for name, lo, hi in ANTECEDENT:
        for blo, bhi in OBS_BINS:
            s = wet[(wet["ante_30d"] >= lo) & (wet["ante_30d"] < hi) & (wet["rainfall_t"] > blo) & (wet["rainfall_t"] <= bhi)]
            if len(s):
                obs.append({"Antecedent": name, "Rain_Bin_in": f"{blo:g}-{bhi:g}", "Rain_Median_in": s["rainfall_t"].median(),
                            "N_Days": len(s), "Observed_Added_Flow_Median_MGD": (s["flow_true"] - s["recent_level"]).median()})
    obs = pd.DataFrame(obs)

    curve.to_csv(OUT_DIR / "rating_curve.csv", index=False)
    notes = pd.DataFrame({"Note": [
        f"Model: Phase 6 production model ({meta['feature_set']}, {len(cols)} features), fitted {meta['trained_on']}.",
        f"Added flow = predicted flow minus the 60-day dry-weather level. Total flow = current dry-weather level + added flow "
        f"(2024 median dry-weather level {level_2024:.2f} MGD, used in the example column).",
        "Day of = the plant day (morning to morning reading) on which the rain falls. Day after assumes no further rain.",
        f"Starting states: {len(base):,} real days with no rain > 0.1 in on the day or day before; 30-day antecedent rain from the record.",
        f"99% of wet days had <= {p99:.2f} in; the largest day was {rmax:.2f} in. Beyond {p99:.2f} in the model has few examples "
        "and the curve flattens because tree models do not extrapolate -- treat it as a lower bound there.",
        "Rain is the nearby-gauge network mean (areal average), which is also what a gridded rainfall forecast represents.",
    ]})
    with pd.ExcelWriter(OUT_DIR / "rating_curve.xlsx", engine="openpyxl") as xw:
        wide = curve.pivot(index="Rain_in", columns="Antecedent", values="Added_Flow_DayOf_MGD")[[a[0] for a in ANTECEDENT]]
        wide.round(2).to_excel(xw, sheet_name="Day_Of_Added_MGD")
        curve.pivot(index="Rain_in", columns="Antecedent", values="Added_Flow_DayAfter_MGD")[[a[0] for a in ANTECEDENT]] \
            .round(2).to_excel(xw, sheet_name="Day_After_Added_MGD")
        curve.round(3).to_excel(xw, sheet_name="Full_Table", index=False)
        obs.round(3).to_excel(xw, sheet_name="Observed_Storms", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 110)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, title in ((axes[0], "DayOf", "Day of the rain"), (axes[1], "DayAfter", "Day after (no further rain)")):
        ax.axvspan(p99, RAIN_STEPS.max(), color=GRID, alpha=0.5, lw=0)
        ax.text((p99 + RAIN_STEPS.max()) / 2, 0.5, "sparse record: curve is a lower bound",
                transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="center", fontsize=8, color=INK_2)
        for (name, _, _), color in zip(ANTECEDENT, COLORS):
            c = curve[curve["Antecedent"] == name]
            col = "Added_Flow_DayOf_MGD" if key == "DayOf" else "Added_Flow_DayAfter_MGD"
            ax.fill_between(c["Rain_in"], c[f"{key}_P25"], c[f"{key}_P75"], color=color, alpha=0.12, lw=0)
            ax.plot(c["Rain_in"], c[col], color=color, lw=2, label=f"{name} (n={int(c['N_Start_Days'].iloc[0])} start days)")
            if key == "DayOf":
                o = obs[obs["Antecedent"] == name]
                ax.plot(o["Rain_Median_in"], o["Observed_Added_Flow_Median_MGD"], "o", color=color, ms=8,
                        mec=SURFACE, mew=2)
        ax.axhline(0, color=NEUTRAL, lw=1)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.set_xlabel("Daily rainfall (in, sewershed gauge average)", color=INK_2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, RAIN_STEPS.max())
    axes[0].set_ylabel("Flow added above dry-weather level (MGD)", color=INK_2)
    handles, _ = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="", marker="o", color=INK_2, ms=7,
                              label="Observed storms (median per rain bin)"))
    axes[0].legend(handles=handles, frameon=False, fontsize=8, loc="upper left",
                   title="Prior 30-day rainfall", title_fontsize=8)
    fig.suptitle("Rainfall-to-flow rating curve -- median added plant flow for a given day's rainfall",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "rating_curve.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(f"Phase 6 production model: {len(base):,} start days; rain P99 {p99:.2f} in, max {rmax:.2f} in")
    print(curve.pivot(index="Rain_in", columns="Antecedent", values="Added_Flow_DayOf_MGD")[[a[0] for a in ANTECEDENT]].round(2).to_string())
    print(obs.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

"""
Fixed model scorecard for the RRWWTF flow model.

Run it on every model update so improvement is tracked on one unchanging set
of metrics. Every definition here (subsets, bands, percentiles, tolerances)
is fixed on purpose: change one and the history stops being comparable.

Input: an out-of-sample predictions CSV with columns Date, y (observed MGD),
p (predicted MGD) and optionally Period -- the format written by
scripts/train_screened_model.py (out_of_sample_predictions_all_years.csv),
where every row was predicted by a model that never saw that year.

Day classification (rain depth, wet/dry, peaking factor) always uses the
Phase 5 context -- max(plant gauge, Open-Meteo) rainfall and the 60-day
dry-weather recent level -- regardless of which model is being scored, so a
model that uses a different rain source is still scored on the same days.

Outputs (output/21_wet_weather_review/scorecard/):
    scorecard_<label>.xlsx       all tables + per-day detail, for Excel review
    cdf_<label>.png              CDF, quantile gap and bias-by-band figure
    scorecard_history.csv        one row per label; rerunning a label replaces its row

Usage:
    python scripts/scorecard.py                                   # Phase 7 production model
    python scripts/scorecard.py --predictions path.csv --label my_variant
"""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_screened_model as tsm

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "21_wet_weather_review" / "scorecard"
HISTORY_CSV = OUT_DIR / "scorecard_history.csv"
DEFAULT_PREDICTIONS = BASE_DIR / "output" / "26_phase7_mrms_rain" / "out_of_sample_predictions_all_years.csv"
DEFAULT_LABEL = "Phase7_mrms_rain"

WET_IN = 0.1
TOL_MGD = 0.3
TOL_PCT = 0.20
BAND_UNDER_MGD, BAND_OVER_MGD = 0.2, 0.3  # agreed reporting tolerance: -0.2 / +0.3 MGD
HIGH_FLOW_MGD = 2.5
PERCENTILES = [5, 10, 25, 50, 75, 90, 95, 99]
FLOW_BANDS = [(0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, np.inf)]
PEAKING_BANDS = [(0, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, np.inf)]
RAIN_BANDS = [(0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]

OBSERVED = "#2a78d6"
PREDICTED = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e4e1"
NEUTRAL = "#8a8985"
SURFACE = "#fcfcfb"


def load_context() -> pd.DataFrame:
    ts0, om, rain = tsm.load_inputs()
    ts, _, _ = tsm.apply_screen(ts0, om, rain)
    f = tsm.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])
    return pd.DataFrame({"rain_in": f["rainfall_t"], "rain_lag1_in": f["rain_lag1"],
                         "recent_level": f["recent_level"]})


def band_label(lo, hi, unit):
    return f">= {lo:g} {unit}" if np.isinf(hi) else f"{lo:g}-{hi:g} {unit}"


def metrics(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {"N": 0}
    y, p = d["y"].to_numpy(float), d["p"].to_numpy(float)
    e = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "N": len(d),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "Bias": float(e.mean()),
        "R2": 1 - float((e ** 2).sum()) / ss_tot if len(d) > 2 and ss_tot > 0 else np.nan,
        "MAPE_Pct": float((np.abs(e) / y).mean() * 100),
        "Within_0.3MGD_Pct": float((np.abs(e) <= TOL_MGD).mean() * 100),
        "Correct_Band_Pct": float(((e >= -BAND_UNDER_MGD) & (e <= BAND_OVER_MGD)).mean() * 100),
        "Accuracy_Pct": float(100 - (np.abs(e) / y).mean() * 100),
        "Within_20Pct_Pct": float((np.abs(e) / y <= TOL_PCT).mean() * 100),
        "Underpredicted_Pct": float((e < 0).mean() * 100),
        "Mean_Observed": float(y.mean()),
        "Mean_Predicted": float(p.mean()),
    }


def table(d, groups) -> pd.DataFrame:
    return pd.DataFrame([{"Subset": name, **metrics(d[mask])} for name, mask in groups])


def quantile_table(d) -> pd.DataFrame:
    rows = []
    for subset, sub in (("All", d), ("Wet", d[d["wet"]]), ("Dry", d[~d["wet"]])):
        for q in PERCENTILES:
            o, p = np.percentile(sub["y"], q), np.percentile(sub["p"], q)
            rows.append({"Subset": subset, "Percentile": q, "Observed_MGD": o, "Predicted_MGD": p,
                         "Gap_MGD": p - o, "Gap_Pct": 100 * (p - o) / o})
    return pd.DataFrame(rows)


def wet_excess_capture(d) -> float:
    """Share of observed wet-day flow above the dry-weather level that the model reproduces."""
    w = d[d["wet"]]
    obs = (w["y"] - w["recent_level"]).sum()
    return float(100 * (w["p"] - w["recent_level"]).sum() / obs) if obs > 0 else np.nan


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def plot(d, qt, peak_tbl, label, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f"Observed vs predicted flow distribution -- {label} "
                 f"({d['Date'].min():%Y}-{d['Date'].max():%Y} out-of-sample, n={len(d):,})",
                 fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left")

    for ax, sub, name in ((axes[0, 0], d, "All days"), (axes[0, 1], d[d["wet"]], "Wet days (rain > 0.1 in on day or day before)")):
        for col, color, lab in (("y", OBSERVED, "Observed"), ("p", PREDICTED, "Predicted")):
            v = np.sort(sub[col].to_numpy())
            ax.plot(v, 100 * np.arange(1, len(v) + 1) / len(v), color=color, lw=2, label=lab)
        style(ax, f"CDF -- {name} (n={len(sub)})", "Daily flow (MGD)", "Days at or below this flow (%)")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.axhline(0, color=NEUTRAL, lw=1)
    for subset, color, marker in (("All", OBSERVED, "o"), ("Wet", PREDICTED, "s")):
        s = qt[qt["Subset"] == subset]
        ax.plot(s["Percentile"], s["Gap_MGD"], color=color, lw=2, marker=marker, ms=7,
                label=f"{subset} days")
    style(ax, "Quantile gap: predicted minus observed at each percentile",
          "Percentile of daily flow", "Predicted - observed (MGD)")
    ax.set_xticks(PERCENTILES)
    ax.legend(frameon=False)
    ax.text(0.01, 0.02, "Below zero = model's distribution is too low at that percentile",
            transform=ax.transAxes, color=INK_2, fontsize=8)

    ax = axes[1, 1]
    t = peak_tbl[peak_tbl["N"] > 0]
    x = np.arange(len(t))
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.bar(x, t["Bias"], width=0.6, color=PREDICTED)
    for i, (b, n) in enumerate(zip(t["Bias"], t["N"])):
        ax.text(i, b + (0.02 if b >= 0 else -0.02), f"{b:+.2f}\nn={n}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Peaking ", "") for s in t["Subset"]], fontsize=8)
    style(ax, "Mean error by observed peaking factor (flow / dry-weather level)",
          "Peaking factor band", "Mean error, predicted - observed (MGD)")
    ax.grid(axis="x", visible=False)
    lo = min(t["Bias"].min(), 0)
    ax.set_ylim(lo * 1.35 - 0.05, max(t["Bias"].max(), 0) * 1.35 + 0.1)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def run(predictions: Path, label: str, context: pd.DataFrame = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(predictions, parse_dates=["Date"]).dropna(subset=["y", "p"])
    ctx = load_context() if context is None else context
    d = d.join(ctx, on="Date")
    d["wet"] = (d["rain_in"] > WET_IN) | (d["rain_lag1_in"] > WET_IN)
    d["peaking_factor"] = d["y"] / d["recent_level"]
    d["error"] = d["p"] - d["y"]

    main = table(d, [("All", np.ones(len(d), bool)), ("Dry", ~d["wet"]), ("Wet", d["wet"]),
                     (f"High flow (>= {HIGH_FLOW_MGD:g} MGD)", d["y"] >= HIGH_FLOW_MGD)])
    flow_tbl = table(d, [(band_label(lo, hi, "MGD"), (d["y"] >= lo) & (d["y"] < hi)) for lo, hi in FLOW_BANDS])
    peak_tbl = table(d, [("Peaking " + band_label(lo, hi, "x"), (d["peaking_factor"] >= lo) & (d["peaking_factor"] < hi))
                         for lo, hi in PEAKING_BANDS])
    rain_groups = [("Dry (no rain t or t-1)", ~d["wet"]),
                   ("Day after rain only", (d["rain_in"] <= WET_IN) & (d["rain_lag1_in"] > WET_IN))]
    rain_groups += [(f"Rain today {band_label(lo, hi, 'in')}", (d["rain_in"] > lo) & (d["rain_in"] <= hi))
                    for lo, hi in RAIN_BANDS]
    rain_tbl = table(d, rain_groups)
    year_tbl = table(d, [(str(y), d["Date"].dt.year == y) for y in sorted(d["Date"].dt.year.unique())])
    qt = quantile_table(d)

    def q(subset, pct):
        return float(qt[(qt["Subset"] == subset) & (qt["Percentile"] == pct)]["Gap_MGD"].iloc[0])

    def m(tbl, subset, key):
        return float(tbl.loc[tbl["Subset"] == subset, key].iloc[0])

    high = f"High flow (>= {HIGH_FLOW_MGD:g} MGD)"
    headline = {
        "Label": label, "Run_Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Git_Commit": git_commit(),
        "Predictions_File": str(predictions.relative_to(BASE_DIR)) if predictions.is_relative_to(BASE_DIR) else str(predictions),
        "Period": f"{d['Date'].min():%Y-%m-%d} to {d['Date'].max():%Y-%m-%d}", "N_Days": len(d),
        "MAE_All": m(main, "All", "MAE"), "RMSE_All": m(main, "All", "RMSE"), "Bias_All": m(main, "All", "Bias"),
        "R2_All": m(main, "All", "R2"), "Within_0.3MGD_Pct": m(main, "All", "Within_0.3MGD_Pct"),
        "Correct_Band_Pct": m(main, "All", "Correct_Band_Pct"), "Accuracy_Pct": m(main, "All", "Accuracy_Pct"),
        "Correct_Band_Wet_Pct": m(main, "Wet", "Correct_Band_Pct"),
        "MAE_Dry": m(main, "Dry", "MAE"), "Bias_Dry": m(main, "Dry", "Bias"),
        "MAE_Wet": m(main, "Wet", "MAE"), "Bias_Wet": m(main, "Wet", "Bias"), "R2_Wet": m(main, "Wet", "R2"),
        "MAE_HighFlow": m(main, high, "MAE"), "Bias_HighFlow": m(main, high, "Bias"),
        "Wet_Excess_Captured_Pct": wet_excess_capture(d),
        "CDF_Gap_P50_All": q("All", 50), "CDF_Gap_P90_All": q("All", 90),
        "CDF_Gap_P95_All": q("All", 95), "CDF_Gap_P99_All": q("All", 99),
        "CDF_Gap_P90_Wet": q("Wet", 90), "CDF_Gap_P99_Wet": q("Wet", 99),
    }

    hist = pd.read_csv(HISTORY_CSV) if HISTORY_CSV.exists() else pd.DataFrame()
    if len(hist):
        hist = hist[hist["Label"] != label]
    hist = pd.concat([hist, pd.DataFrame([headline])], ignore_index=True)
    hist.to_csv(HISTORY_CSV, index=False)

    worst = d[d["wet"]].nsmallest(25, "error")
    daily = d[["Date"] + (["Period"] if "Period" in d else []) +
              ["y", "p", "error", "rain_in", "rain_lag1_in", "recent_level", "peaking_factor", "wet"]]
    daily = daily.rename(columns={"y": "Observed_MGD", "p": "Predicted_MGD", "error": "Error_MGD",
                                  "rain_in": "Rain_t_in", "rain_lag1_in": "Rain_t-1_in",
                                  "recent_level": "Dry_Level_MGD", "peaking_factor": "Peaking_Factor", "wet": "Wet_Day"})
    definitions = pd.DataFrame([
        ("Error", "Predicted minus observed. Negative = under-prediction."),
        ("MAE", "Mean absolute error, MGD. Primary tracking metric."),
        ("RMSE", "Root mean squared error, MGD. Penalises large misses (storm peaks) more than MAE."),
        ("Bias", "Mean error, MGD. Systematic over (+) or under (-) prediction."),
        ("R2", "Share of day-to-day variance explained. Dominated by the largest days."),
        ("Within 0.3 MGD / 20%", "Share of days whose prediction is within that tolerance."),
        ("Correct band", "Share of days predicted between 0.2 MGD under and 0.3 MGD over the observed flow "
                         "(agreed reporting tolerance, e.g. observed 1.7 -> 1.5 to 2.0 counts)."),
        ("Accuracy", "100% minus the mean absolute percent error (MAPE)."),
        ("Wet day", f"Rain > {WET_IN:g} in on the day or the day before (max of plant gauge and Open-Meteo)."),
        ("Peaking factor", "Observed flow / 60-day dry-weather median level. Removes year-to-year baseline drift."),
        ("CDF gap Pxx", "Predicted minus observed flow at the same percentile of each distribution. "
                        "Compares distributions, not days, so it is not affected by timing misses."),
        ("Wet excess captured", "Sum over wet days of (predicted - dry level) / sum of (observed - dry level). "
                                "100% = model reproduces the full wet-weather volume."),
        ("Out-of-sample", "Every day was predicted by a model trained only on earlier years (2020, 2021, 2022, 2024)."),
        ("Caution on observed-flow bands", "Grouping days by OBSERVED flow always shows under-prediction at the top "
                                           "(the highest days are partly the model's misses). Use the CDF gap for the "
                                           "distribution question; use the band tables to size the miss."),
    ], columns=["Term", "Definition"])

    xlsx = OUT_DIR / f"scorecard_{label}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame(list(headline.items()), columns=["Metric", "Value"]).to_excel(xw, sheet_name="Headline", index=False)
        main.to_excel(xw, sheet_name="Wet_Dry", index=False)
        rain_tbl.to_excel(xw, sheet_name="By_Rain_Depth", index=False)
        flow_tbl.to_excel(xw, sheet_name="By_Flow_Band", index=False)
        peak_tbl.to_excel(xw, sheet_name="By_Peaking_Factor", index=False)
        year_tbl.to_excel(xw, sheet_name="By_Year", index=False)
        qt.to_excel(xw, sheet_name="CDF_Quantiles", index=False)
        worst.rename(columns={"y": "Observed_MGD", "p": "Predicted_MGD", "error": "Error_MGD"}) \
            [["Date", "Observed_MGD", "Predicted_MGD", "Error_MGD", "rain_in", "rain_lag1_in", "peaking_factor"]] \
            .to_excel(xw, sheet_name="Worst_Wet_Days", index=False)
        daily.to_excel(xw, sheet_name="Daily", index=False)
        hist.to_excel(xw, sheet_name="History", index=False)
        definitions.to_excel(xw, sheet_name="Definitions", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.000"

    plot(d, qt, peak_tbl, label, OUT_DIR / f"cdf_{label}.png")

    print(f"SCORECARD -- {label}  ({headline['Period']}, n={len(d):,})")
    for k in ("MAE_All", "RMSE_All", "Bias_All", "R2_All", "MAE_Wet", "Bias_Wet", "Bias_HighFlow",
              "Wet_Excess_Captured_Pct", "CDF_Gap_P50_All", "CDF_Gap_P90_All", "CDF_Gap_P99_All"):
        print(f"  {k:<26} {headline[k]:+.3f}")
    print(f"  -> {xlsx.relative_to(BASE_DIR)}")
    return headline


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    a = ap.parse_args()
    run(a.predictions.resolve(), a.label)


if __name__ == "__main__":
    main()

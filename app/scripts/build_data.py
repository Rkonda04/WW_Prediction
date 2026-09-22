"""
Build the dashboard data bundle from Phase 5 model outputs.

Reads (relative to the WW project root, i.e. app/..):
  output/13_screened_model/test_predictions_2024.csv   - actual vs predicted flow
  output/13_screened_model/test_scores.csv             - the model's own scorecard
  output/14_saturation_model/saturation_timeseries.csv - saturation index + class

Writes:
  app/src/data/predictions.json

Metrics are recomputed here from the raw predictions rather than copied from
test_scores.csv, so the dashboard cannot silently drift from the data it plots.
The recomputed values are cross-checked against test_scores.csv on every run.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRED_CSV = ROOT / "output" / "13_screened_model" / "test_predictions_2024.csv"
SCORES_CSV = ROOT / "output" / "13_screened_model" / "test_scores.csv"
SAT_CSV = ROOT / "output" / "14_saturation_model" / "saturation_timeseries.csv"

OUT_JSON = Path(__file__).resolve().parents[1] / "src" / "data" / "predictions.json"

TOLERANCE_MGD = 0.3

# The model scores wet/dry by rainfall (scripts/train_screened_model.py): a day
# is Wet if rain today OR rain yesterday exceeds WET_DAY_RAIN_THRESHOLD_IN.
# The saturation model classifies by a 14-day water balance instead, so the two
# splits disagree and are reported separately rather than merged.
WET_DAY_RAIN_THRESHOLD_IN = 0.1


def r2(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2:
        return None
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


def score_block(df):
    """Metric bundle for a subset of rows. error = predicted - actual."""
    if df.empty:
        return {"n": 0}
    err = df["error"].to_numpy(dtype=float)
    actual = df["actual"].to_numpy(dtype=float)
    pred = df["predicted"].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "r2": r2(actual, pred),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "mape": float(np.mean(np.abs(err / actual)) * 100),
        "within_tolerance": float(np.mean(np.abs(err) <= TOLERANCE_MGD)),
        "within_0_1": float(np.mean(np.abs(err) <= 0.1)),
        "within_0_5": float(np.mean(np.abs(err) <= 0.5)),
    }


def main():
    preds = pd.read_csv(PRED_CSV, parse_dates=["Date"])
    sat = pd.read_csv(SAT_CSV, parse_dates=["Date"])

    preds = preds.rename(
        columns={
            "Observed_MGD": "actual",
            "Predicted_MGD": "predicted",
            "Screened_Out": "screened_out",
            "Rain_in": "rain_in",
            "Tmin_F": "tmin_f",
        }
    )
    preds["screened_out"] = preds["screened_out"].astype(str).str.lower().eq("true")

    sat = sat.rename(
        columns={
            "saturation_index": "saturation",
            "confidence_score": "sat_confidence",
            "precipitation_mm": "precip_mm",
            "evapotranspiration_mm": "et_mm",
        }
    )
    sat["classification"] = sat["classification"].str.upper()

    merged = preds.merge(
        sat[
            [
                "Date",
                "saturation",
                "classification",
                "sat_confidence",
                "precip_mm",
                "et_mm",
                "net_balance_mm",
            ]
        ],
        on="Date",
        how="left",
    )
    merged["error"] = merged["predicted"] - merged["actual"]

    # Reproduce the model's own rainfall-based wet-day flag so the dashboard can
    # show the same wet/dry breakdown the model report quotes.
    # The model builds this lag over its full continuous daily series. The test
    # CSV omits 21 days inside its own range and starts one day after the series
    # does, so a plain row-wise shift() would look back to the wrong day. Rebuild
    # the lag on a continuous daily index, filling days the test CSV omits from
    # the same precipitation record the model's Rain_in column comes from.
    rain_daily = (
        merged.set_index("Date")["rain_in"]
        .reindex(pd.date_range(merged["Date"].min() - pd.Timedelta(days=1),
                               merged["Date"].max(), freq="D"))
        .fillna(sat.set_index("Date")["precip_mm"])
    )
    wet_daily = (rain_daily > WET_DAY_RAIN_THRESHOLD_IN) | (
        rain_daily.shift(1) > WET_DAY_RAIN_THRESHOLD_IN
    )
    merged["rain_wet"] = wet_daily.reindex(merged["Date"]).fillna(False).to_numpy()

    unmatched = int(merged["saturation"].isna().sum())

    # Headline metrics use the screened protocol (flagged days excluded from
    # scoring), which is the protocol test_scores.csv reports as "Screened".
    scored = merged[~merged["screened_out"]].copy()

    metrics = {
        "all": score_block(scored),
        # Split A - rainfall, as the model itself scores. Matches test_scores.csv.
        "rain_wet": score_block(scored[scored["rain_wet"]]),
        "rain_dry": score_block(scored[~scored["rain_wet"]]),
        # Split B - 14-day saturation state, from the saturation model.
        "sat_wet": score_block(scored[scored["classification"] == "WET"]),
        "sat_dry": score_block(scored[scored["classification"] == "DRY"]),
        "tolerance_mgd": TOLERANCE_MGD,
        "wet_rain_threshold_in": WET_DAY_RAIN_THRESHOLD_IN,
    }

    # Cross-check against the model's own scorecard.
    published = pd.read_csv(SCORES_CSV)
    scr = published[published["Protocol"] == "Screened"]

    def pub(subset):
        return scr[scr["Subset"] == subset].iloc[0]

    checks = {}
    for subset, key in (("All", "all"), ("Wet", "rain_wet"), ("Dry", "rain_dry")):
        row = pub(subset)
        mine = metrics[key]
        checks[key] = {
            "published_r2": float(row["R2"]),
            "published_rmse": float(row["RMSE"]),
            "published_n": int(row["N"]),
            "recomputed_r2": mine["r2"],
            "recomputed_rmse": mine["rmse"],
            "recomputed_n": mine["n"],
        }
        drift = abs(float(row["R2"]) - (mine["r2"] or 0.0))
        if drift > 0.02 or int(row["N"]) != mine["n"]:
            print(
                f"  WARNING: {subset} subset differs from test_scores.csv "
                f"(R2 drift {drift:.4f}, n {mine['n']} vs {int(row['N'])})"
            )
    checks["published_within_0_3_pct"] = float(pub("All")["Pct_Days_Within_0.3_MGD"])
    checks["published_mape_accuracy_pct"] = float(pub("All")["Accuracy_1_minus_MAPE_Pct"])

    def num(v, nd):
        return None if pd.isna(v) else round(float(v), nd)

    records = []
    for _, r in merged.iterrows():
        records.append(
            {
                "date": r["Date"].strftime("%Y-%m-%d"),
                "actual": num(r["actual"], 4),
                "predicted": num(r["predicted"], 4),
                "error": num(r["error"], 4),
                "rain_in": num(r["rain_in"], 3),
                "tmin_f": num(r["tmin_f"], 1),
                "saturation": num(r["saturation"], 4),
                "classification": None if pd.isna(r["classification"]) else r["classification"],
                "sat_confidence": num(r["sat_confidence"], 4),
                "precip_mm": num(r["precip_mm"], 3),
                "screened_out": bool(r["screened_out"]),
                "rain_wet": bool(r["rain_wet"]),
            }
        )

    # Saturation history for the 90-day trend behind each selectable day. Only
    # the reachable span ships: the full 2018- series would roughly double the
    # bundle for rows no view can display.
    hist_from = merged["Date"].min() - pd.Timedelta(days=90)
    sat_hist = [
        {
            "date": r["Date"].strftime("%Y-%m-%d"),
            "saturation": round(float(r["saturation"]), 4),
            "classification": r["classification"],
        }
        for _, r in sat[sat["Date"] >= hist_from].iterrows()
    ]

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": {
            "predictions": PRED_CSV.relative_to(ROOT).as_posix(),
            "scores": SCORES_CSV.relative_to(ROOT).as_posix(),
            "saturation": SAT_CSV.relative_to(ROOT).as_posix(),
        },
        "model": {
            "name": "XGBoost Screened (Phase 5)",
            "protocol": "Screened - flagged days excluded from scoring",
            "test_start": records[0]["date"],
            "test_end": records[-1]["date"],
            "saturation_start": sat["Date"].min().strftime("%Y-%m-%d"),
            "saturation_end": sat["Date"].max().strftime("%Y-%m-%d"),
            "n_test_days": len(records),
            "n_screened_out": int(merged["screened_out"].sum()),
            "n_missing_saturation": unmatched,
        },
        "metrics": metrics,
        "validation": checks,
        "predictions": records,
        "saturation_history": sat_hist,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(bundle, indent=1), encoding="utf-8")

    m = metrics["all"]
    print(f"Wrote {OUT_JSON}")
    print(f"  test window     : {bundle['model']['test_start']} -> {bundle['model']['test_end']}")
    print(f"  days            : {len(records)} ({bundle['model']['n_screened_out']} screened out)")
    print(f"  no saturation   : {unmatched}")
    print(f"  R2 / RMSE / MAE : {m['r2']:.3f} / {m['rmse']:.3f} / {m['mae']:.3f} MGD")
    print(f"  within +/-0.3   : {m['within_tolerance'] * 100:.1f}%")
    print(f"  MAPE accuracy   : {100 - m['mape']:.1f}%")
    for key, label in (("rain", "rain split"), ("sat", "saturation split")):
        w, d = metrics[f"{key}_wet"], metrics[f"{key}_dry"]
        print(
            f"  {label:<16}: wet R2={w['r2']:+.3f} (n={w['n']})  "
            f"dry R2={d['r2']:+.3f} (n={d['n']})"
        )


if __name__ == "__main__":
    main()

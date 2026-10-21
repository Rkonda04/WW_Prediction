"""
RRWWTF -- cheap check of MRMS FLASH soil saturation before any full download.

Question: does MRMS soil saturation carry wet-weather signal that ERA5 soil
moisture (already tested, no gain) and 30-day antecedent rain do not?

Pulls one MRMS snapshot per date for ~45 dates only (both FLASH soil models,
CREST and SAC), averaged over a ~14 x 14 km box around the plant:
  * storm days since MRMS coverage began (2020-10-14): 0.75-2.5 in of
    gauge-network rain on a day after a dry day, split dry / normal / wet by
    30-day antecedent rain (same bins as rating_curve.py);
  * the 2024 no-rain spike days from output/12_r2_ceiling_diagnostic.

For storms the snapshot is 12:00 UTC (6 am CST) of the day before the storm
day, i.e. the soil state before the rain. For spike days it is 12:00 UTC of
the spike day itself.

Tests (all on the same storm rows):
  1. Group medians by antecedent class -- MRMS vs ERA5.
  2. Spearman correlation with the observed storm response (flow - recent
     level) and with the Phase 6 out-of-sample residual (observed - predicted).
     If MRMS is useful it should track the residual; ante_30d and ERA5 are the
     comparison.
  3. Spike days: MRMS saturation against the storm-day distribution.

Outputs: output/24_mrms_soil_check/
  mrms_soil_snapshots.csv   -- one row per date, both models + ERA5 + flow
  mrms_soil_check_summary.csv

Usage:
    python scripts/mrms_soil_check.py
"""

import gzip
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import train_phase6_model as p6

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "24_mrms_soil_check"
SNAPSHOT_CSV = OUT_DIR / "mrms_soil_snapshots.csv"
SUMMARY_CSV = OUT_DIR / "mrms_soil_check_summary.csv"
ERA5_CSV = BASE_DIR / "output" / "21_wet_weather_review" / "era5_soil_moisture_daily.csv"
OOS_CSV = p6.OUT_DIR / "out_of_sample_predictions_all_years.csv"

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"
PRODUCTS = {"crest": "FLASH_CREST_MAXSOILSAT_00.00", "sac": "FLASH_SAC_MAXSOILSAT_00.00"}
CENTER_LAT, CENTER_LON = 29.5838, -95.7552  # Richmond COOP station, ~1 mi from the plant
BOX_DEG = 0.065                             # half-width, ~7 km
MRMS_START = pd.Timestamp("2020-10-15")

STORM_RAIN = (0.75, 2.5)
ANTECEDENT = [("dry", 0, 1.0), ("normal", 1.0, 4.0), ("wet", 4.0, np.inf)]
SPIKE_DAYS = ["2024-01-07", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-13", "2024-02-14", "2024-04-02"]


def select_dates(f):
    w = f.loc[MRMS_START:]
    w = w[(w["rainfall_t"] > STORM_RAIN[0]) & (w["rainfall_t"] <= STORM_RAIN[1])
          & (w["rain_lag1"] <= p6.wx.tsm.WET) & w["flow_true"].notna()]
    rows = []
    for name, lo, hi in ANTECEDENT:
        for d in w[(w["ante_30d"] >= lo) & (w["ante_30d"] < hi)].index:
            rows.append({"Date": d, "Kind": "storm", "Antecedent": name, "Snapshot": d - pd.Timedelta(days=1)})
    for d in pd.to_datetime(SPIKE_DAYS):
        rows.append({"Date": d, "Kind": "no_rain_spike", "Antecedent": None, "Snapshot": d})
    return pd.DataFrame(rows)


def mrms_box_mean(product, snap, tmp):
    day = snap.strftime("%Y%m%d")
    url = f"{BUCKET}/{product}/{day}/MRMS_{product}_{day}-120000.grib2.gz"
    gz, raw = tmp / "f.grib2.gz", tmp / "f.grib2"
    try:
        urllib.request.urlretrieve(url, gz)
    except Exception as e:
        print(f"   missing {url.rsplit('/', 1)[-1]} ({e})")
        return np.nan
    with gzip.open(gz) as src, open(raw, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with xr.open_dataset(raw, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        a = ds[list(ds.data_vars)[0]].sel(
            latitude=slice(CENTER_LAT + BOX_DEG, CENTER_LAT - BOX_DEG),
            longitude=slice(360 + CENTER_LON - BOX_DEG, 360 + CENTER_LON + BOX_DEG)).values
    a = a[a >= 0]
    return float(a.mean()) if a.size else np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f, _, _ = p6.build_frame(*p6.load_inputs())
    dates = select_dates(f)
    print(f"{len(dates)} dates: " + ", ".join(f"{k}={v}" for k, v in
          dates.fillna({"Antecedent": "spike"})["Antecedent"].value_counts().items()))

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for key, product in PRODUCTS.items():
            print(f" pulling {product} ...")
            dates[f"mrms_{key}"] = [mrms_box_mean(product, s, tmp) for s in dates["Snapshot"]]

    era5 = pd.read_csv(ERA5_CSV, parse_dates=["Date"]).set_index("Date")["sm_0_7cm"]
    oos = pd.read_csv(OOS_CSV, parse_dates=["Date"]).set_index("Date")
    d = dates.set_index("Date")
    d["era5_sm_0_7cm"] = era5.reindex(d["Snapshot"]).values
    for c in ["rainfall_t", "ante_30d", "flow_true", "recent_level"]:
        d[c] = f[c].reindex(d.index)
    d["response_MGD"] = d["flow_true"] - d["recent_level"]
    d["phase6_pred"] = oos["p"].reindex(d.index)
    d["residual_MGD"] = d["flow_true"] - d["phase6_pred"]
    d.round(4).to_csv(SNAPSHOT_CSV, index_label="Date")

    storms = d[d["Kind"] == "storm"]
    soil_cols = ["mrms_crest", "mrms_sac", "era5_sm_0_7cm", "ante_30d"]
    print("\n1. STORM GROUP MEDIANS (soil state the morning before the storm)")
    g = storms.groupby("Antecedent")[soil_cols + ["rainfall_t", "response_MGD", "residual_MGD"]].median() \
        .reindex([a[0] for a in ANTECEDENT])
    g.insert(0, "N", storms["Antecedent"].value_counts())
    print(g.round(3).to_string())

    print(f"\n2. SPEARMAN CORRELATION across all {len(storms)} storms")
    rows = []
    for c in soil_cols:
        s = storms[[c, "response_MGD", "residual_MGD", "rainfall_t"]].dropna()
        rows.append({"Feature": c, "N": len(s),
                     "rho_vs_response": s[c].corr(s["response_MGD"], method="spearman"),
                     "rho_vs_phase6_residual": s[c].corr(s["residual_MGD"], method="spearman"),
                     "rho_vs_mrms_crest": s[c].corr(storms.loc[s.index, "mrms_crest"], method="spearman")})
    corr = pd.DataFrame(rows)
    print(corr.round(3).to_string(index=False))
    corr.to_csv(SUMMARY_CSV, index=False)

    print("\n3. NO-RAIN SPIKE DAYS (MRMS same day) vs storm-morning percentiles")
    spikes = d[d["Kind"] == "no_rain_spike"]
    for key in PRODUCTS:
        col = f"mrms_{key}"
        ref = storms[col].dropna()
        pct = [(ref < v).mean() * 100 for v in spikes[col]]
        print(f" {key}: " + ", ".join(f"{i:%m-%d} {v:.1f} (P{p:.0f})" for i, v, p in zip(spikes.index, spikes[col], pct)))
    print(f"\nSaved {SNAPSHOT_CSV.relative_to(BASE_DIR)} and {SUMMARY_CSV.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

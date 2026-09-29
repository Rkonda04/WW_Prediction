"""
Fetch the two candidate wet-weather inputs requested in the internal review:

  1. Nearby rain gauges (NOAA ACIS / GHCN-Daily: the Richmond COOP station,
     Sugar Land airport and CoCoRaHS volunteer gauges) within GAUGE_RADIUS_MI
     of Richmond. The plant gauge is a single point and under-reads the
     regional record 30-50% in most years, so localized sewershed rain can
     be missed entirely.
  2. Modelled soil moisture (ERA5-Land via the Open-Meteo archive API),
     0-7 cm and 7-28 cm, daily mean. The existing "saturation index" is a
     precipitation-minus-ET0 bookkeeping proxy, not soil moisture.

Date alignment: plant "day D" is the totalizer difference between the
morning reading on D and the morning reading on D+1. GHCN/CoCoRaHS gauges
are read ~07:00 and file the 24 h total under the reading date, so the gauge
value dated D+1 covers plant day D. The shift is not assumed: it is chosen
per gauge by which lag (-1/0/+1) correlates best with the plant gauge's
Rainfall_in_Corrected (already aligned to the plant day), and reported.

Outputs (output/21_wet_weather_review/):
    nearby_gauges_meta.csv       station list, distance, record length, chosen lag
    nearby_gauges_daily.csv      one column per gauge (inches, plant-day aligned)
    gauge_network_daily.csv      network mean / max / count per plant day
    era5_soil_moisture_daily.csv sm_0_7cm, sm_7_28cm (m3/m3)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "21_wet_weather_review"
CLEANED_CSV = BASE_DIR / "output" / "09_data_screening" / "richmond_daily_cleaned.csv"

CENTER_LAT, CENTER_LON = 29.5838, -95.7552  # Richmond COOP station, ~1 mi from the plant
GAUGE_RADIUS_MI = 6.0
START, END = "2017-11-01", "2024-12-31"
MIN_DAYS = 365
ACIS = "https://data.rcc-acis.org"
ERA5 = "https://archive-api.open-meteo.com/v1/archive"


def miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def list_stations():
    d = 0.12
    bbox = f"{CENTER_LON - d},{CENTER_LAT - d},{CENTER_LON + d},{CENTER_LAT + d}"
    r = requests.post(f"{ACIS}/StnMeta", json={
        "bbox": bbox, "elems": "pcpn", "sdate": START, "edate": END,
        "meta": "name,sids,ll,valid_daterange"}, timeout=60)
    r.raise_for_status()
    rows = []
    for m in r.json()["meta"]:
        lon, lat = m["ll"]
        dist = miles(CENTER_LAT, CENTER_LON, lat, lon)
        if dist <= GAUGE_RADIUS_MI:
            rows.append({"name": m["name"], "sid": m["sids"][0], "lat": lat, "lon": lon,
                         "distance_mi": round(float(dist), 2)})
    return pd.DataFrame(rows).sort_values("distance_mi").reset_index(drop=True)


def fetch_gauge(sid):
    r = requests.post(f"{ACIS}/StnData", json={
        "sid": sid, "sdate": START, "edate": END, "elems": [{"name": "pcpn"}]}, timeout=120)
    r.raise_for_status()
    data = r.json().get("data", [])
    s = pd.Series({pd.Timestamp(d): v for d, v in data}, dtype=object)
    # ACIS codes: "M" missing, "T" trace, "S" part of a multi-day accumulation, "A" accumulated total.
    # Accumulations cannot be split into days, so both are treated as missing.
    def parse(v):
        v = str(v)
        if v == "T":
            return 0.0
        if v in ("M", "S") or v.endswith("A"):
            return np.nan
        try:
            return float(v)
        except ValueError:
            return np.nan
    return s.map(parse).astype(float)


def best_lag(series, reference):
    scores = {}
    for lag in (-1, 0, 1):
        shifted = series.shift(-lag)  # lag=+1: gauge dated D+1 -> plant day D
        both = pd.concat([shifted, reference], axis=1).dropna()
        scores[lag] = both.corr().iloc[0, 1] if len(both) > 30 else np.nan
    lag = max(scores, key=lambda k: -np.inf if pd.isna(scores[k]) else scores[k])
    return lag, scores


def fetch_soil_moisture():
    r = requests.get(ERA5, params={
        "latitude": CENTER_LAT, "longitude": CENTER_LON, "start_date": START, "end_date": END,
        "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm", "timezone": "America/Chicago"}, timeout=180)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"]), "sm_0_7cm": h["soil_moisture_0_to_7cm"],
                       "sm_7_28cm": h["soil_moisture_7_to_28cm"]})
    # State at the start of the plant day (morning reading) is the value known before the rain falls.
    morning = df[df["time"].dt.hour == 7].assign(Date=lambda x: x["time"].dt.normalize())
    return morning.set_index("Date")[["sm_0_7cm", "sm_7_28cm"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plant = pd.read_csv(CLEANED_CSV, parse_dates=["Date"]).set_index("Date")["Rainfall_in_Corrected"]

    meta = list_stations()
    print(f"{len(meta)} gauges within {GAUGE_RADIUS_MI:g} mi of Richmond")
    cols, keep = {}, []
    for _, st in meta.iterrows():
        s = fetch_gauge(st["sid"])
        n = int(s.notna().sum())
        lag, sc = best_lag(s, plant)
        aligned = s.shift(-lag)
        label = f"{st['name']} [{st['sid']}]"
        meta.loc[meta["sid"] == st["sid"], ["n_days", "lag_days", "r_lag_m1", "r_lag_0", "r_lag_p1"]] = \
            [n, lag, sc[-1], sc[0], sc[1]]
        print(f"  {st['distance_mi']:4.1f} mi  {st['name']:<28} n={n:5d}  lag={lag:+d}  "
              f"r(-1/0/+1)={sc[-1]:.2f}/{sc[0]:.2f}/{sc[1]:.2f}")
        if n >= MIN_DAYS:
            cols[label] = aligned
            keep.append(st["sid"])
    meta["used"] = meta["sid"].isin(keep)
    meta.to_csv(OUT_DIR / "nearby_gauges_meta.csv", index=False)

    wide = pd.DataFrame(cols).loc["2018-01-01":"2024-12-31"]
    wide.index.name = "Date"
    wide.to_csv(OUT_DIR / "nearby_gauges_daily.csv")
    net = pd.DataFrame({
        "network_mean_in": wide.mean(axis=1),
        "network_median_in": wide.median(axis=1),
        "network_max_in": wide.max(axis=1),
        "network_n_gauges": wide.notna().sum(axis=1),
    })
    net.to_csv(OUT_DIR / "gauge_network_daily.csv")
    print(f"network: {len(wide.columns)} gauges used, median {net['network_n_gauges'].median():.0f} reporting per day")

    sm = fetch_soil_moisture().loc["2018-01-01":"2024-12-31"]
    sm.to_csv(OUT_DIR / "era5_soil_moisture_daily.csv", index_label="Date")
    print(f"soil moisture: {len(sm)} days, 0-7 cm range {sm['sm_0_7cm'].min():.3f}-{sm['sm_0_7cm'].max():.3f} m3/m3")


if __name__ == "__main__":
    main()

"""
Richmond, TX — Independent Historical Rainfall Download (Open-Meteo)

Downloads DAILY historical precipitation for Richmond, Texas from the
Open-Meteo Historical Weather (Archive) API, using a fixed coordinate and
the `precipitation_sum` daily variable (same terminology as the existing
Pearland forecasting app), converted to inches by the API itself.

This script is intentionally independent of the existing RRWWTF wastewater
dataset: it does NOT read, use, or compare the `Rain` field already present
in that dataset. The wastewater flow CSV is only used to look up the date
range to request (start/end date), not any flow or rainfall values.

No merging, correlation, lag analysis, or modeling is performed here —
this step only downloads, validates, and QA-plots an independent rainfall
series for later use.

Output:
    output/richmond_openmeteo_daily_rainfall.csv
    output/figures/openmeteo_richmond_rainfall_qa.png
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"

FLOW_CSV = OUTPUT_DIR / "richmond_total_treated_mgd.csv"  # date range reference ONLY
RAIN_CSV = OUTPUT_DIR / "richmond_openmeteo_daily_rainfall.csv"
QA_FIG_PATH = FIG_DIR / "openmeteo_richmond_rainfall_qa.png"

LATITUDE = 29.5818
LONGITUDE = -95.7608
TIMEZONE = "America/Chicago"

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Step 1: determine the date range to request (from the flow dataset's
# Date column only — no flow or rainfall values are read from it)
# ---------------------------------------------------------------------------

def get_target_date_range(flow_csv: Path) -> tuple[str, str]:
    flow_dates = pd.read_csv(flow_csv, usecols=["Date"], parse_dates=["Date"])
    start_date = flow_dates["Date"].min().strftime("%Y-%m-%d")
    end_date = flow_dates["Date"].max().strftime("%Y-%m-%d")
    return start_date, end_date


# ---------------------------------------------------------------------------
# Step 2: download daily precipitation from Open-Meteo Archive API
# ---------------------------------------------------------------------------

def fetch_openmeteo_daily_rainfall(latitude: float, longitude: float,
                                    start_date: str, end_date: str, timezone: str) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": timezone,
        "precipitation_unit": "inch",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{ARCHIVE_API_URL}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Open-Meteo request failed ({e.code}): {e.read().decode(errors='ignore')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach Open-Meteo Archive API: {e}") from e

    if "daily" not in payload or "time" not in payload["daily"]:
        raise RuntimeError(f"Unexpected Open-Meteo response shape: {payload}")

    daily = payload["daily"]
    df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]),
        "OpenMeteo_Rainfall_in": daily["precipitation_sum"],
    })
    df["OpenMeteo_Rainfall_in"] = pd.to_numeric(df["OpenMeteo_Rainfall_in"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 3: validation checks (duplicates, missing calendar dates)
# ---------------------------------------------------------------------------

def check_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Date"].value_counts()
    return counts[counts > 1].sort_index()


def check_missing_calendar_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    present = pd.DatetimeIndex(df["Date"].unique())
    return full_range.difference(present)


# ---------------------------------------------------------------------------
# Step 4: QA plot
# ---------------------------------------------------------------------------

def plot_qa(df: pd.DataFrame, out_path: Path):
    valid = df.dropna(subset=["OpenMeteo_Rainfall_in"])
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(valid["Date"], valid["OpenMeteo_Rainfall_in"], width=1.0,
           color="#63b3ed", alpha=0.8, label="Open-Meteo Daily Rainfall")
    ax.set_title("Richmond, TX — Open-Meteo Historical Daily Rainfall (QA)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (inches)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    start_date, end_date = get_target_date_range(FLOW_CSV)

    rain_df = fetch_openmeteo_daily_rainfall(LATITUDE, LONGITUDE, start_date, end_date, TIMEZONE)
    rain_df.to_csv(RAIN_CSV, index=False)

    dup_dates = check_duplicate_dates(rain_df)
    missing_dates = check_missing_calendar_dates(rain_df)

    n_days = len(rain_df)
    n_missing_values = int(rain_df["OpenMeteo_Rainfall_in"].isna().sum())
    n_rain_days = int((rain_df["OpenMeteo_Rainfall_in"] > 0).sum())
    max_rain = rain_df["OpenMeteo_Rainfall_in"].max()
    max_rain_date = rain_df.loc[rain_df["OpenMeteo_Rainfall_in"].idxmax(), "Date"] if pd.notna(max_rain) else None
    total_rain = rain_df["OpenMeteo_Rainfall_in"].sum()

    plot_qa(rain_df, QA_FIG_PATH)

    print("RICHMOND, TX - OPEN-METEO HISTORICAL DAILY RAINFALL")
    print(f"Coordinates:                 {LATITUDE}, {LONGITUDE}")
    print(f"Timezone:                    {TIMEZONE}")
    print(f"Start date:                  {rain_df['Date'].min().date()}")
    print(f"End date:                    {rain_df['Date'].max().date()}")
    print(f"Total number of days:        {n_days}")
    print(f"Missing rainfall values:     {n_missing_values}")
    print(f"Duplicate dates:             {len(dup_dates)}")
    print(f"Days with rainfall > 0:      {n_rain_days}")
    print(f"Maximum daily rainfall (in): {max_rain:.3f} on {max_rain_date.date() if max_rain_date is not None else 'N/A'}")
    print(f"Total rainfall over period (in): {total_rain:,.2f}")
    print(f"Missing calendar dates:      {len(missing_dates)}")
    if len(missing_dates) > 0:
        print(f"  First few missing: {[d.date().isoformat() for d in missing_dates[:5]]}")
    print()
    print("Output files:")
    print(f"  {RAIN_CSV}")
    print(f"  {QA_FIG_PATH}")


if __name__ == "__main__":
    main()

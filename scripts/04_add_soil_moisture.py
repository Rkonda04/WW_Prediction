"""
Derive Soil Moisture Features from Rainfall and Temperature

Creates synthetic soil moisture indices based on:
  1. Antecedent rainfall (accumulation over time)
  2. Temperature (affects evapotranspiration rate)
  3. Decay function (soil moisture decreases over time without rain)

This approach captures the hydrologic memory that affects infiltration without
requiring external API data. Soil moisture is represented as a proxy index
derived from observed weather patterns.

Outputs lagged features:
  - soil_moisture_index: proxy for upper soil layer moisture
  - soil_moisture_lag1, lag3, lag7: lagged values
  - soil_moisture_rolling_7d: 7-day rolling average

Outputs:
    output/00_master_dataset/richmond_master_with_soil_moisture.csv
    output/00_master_dataset/soil_moisture_metadata.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_corrected_rainfall.csv"
OUTPUT_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_soil_moisture.csv"
METADATA_FILE = OUTPUT_DIR / "00_master_dataset" / "soil_moisture_metadata.txt"

# Decay rate: soil moisture decreases ~10% per day without rain
DAILY_DECAY_RATE = 0.90
# Rainfall contribution: 1 inch adds ~15 mm to soil moisture index
RAIN_MULTIPLIER = 15.0

def create_soil_moisture_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create soil moisture proxy index from antecedent rainfall and temperature.

    Physics-based model:
      SM(t) = SM(t-1) * decay_factor + rain(t) * multiplier - evapotranspiration

    Decay factor increases in warm periods (more evapotranspiration).
    """
    df = df.copy()

    # Use corrected rainfall
    rainfall = df["Rainfall_in_Corrected"].fillna(0).values

    # Simple temperature proxy: base decay + temperature effect
    # Assume average temp ~55°F; cooler = less evapotranspiration, warmer = more
    has_temp = False
    if "Tmin_C_OpenMeteo" in df.columns and df["Tmin_C_OpenMeteo"].notna().sum() > len(df) * 0.5:
        temp = df["Tmin_C_OpenMeteo"].fillna(df["Tmin_C_OpenMeteo"].mean())
        temp_f = (temp * 9/5) + 32
        has_temp = True
    else:
        # Use month as proxy for temperature if temp data unavailable
        temp_f = 32 + 20 * np.sin(np.radians((df["Month"].values - 1) * 30))

    # Decay factor: higher in warm months (more evapotranspiration)
    base_decay = DAILY_DECAY_RATE
    temp_normalized = (temp_f - 32) / 50  # normalized to [0, 1] roughly
    decay_factor = base_decay - (temp_normalized * 0.05)  # vary decay by ±5%
    decay_factor = np.clip(decay_factor, 0.85, 0.95)

    soil_moisture = np.zeros(len(df))
    soil_moisture[0] = rainfall[0] * RAIN_MULTIPLIER

    for t in range(1, len(df)):
        soil_moisture[t] = soil_moisture[t-1] * decay_factor[t] + rainfall[t] * RAIN_MULTIPLIER
        # Cap at reasonable range (0-100 mm equivalent)
        soil_moisture[t] = np.clip(soil_moisture[t], 0, 100)

    df["soil_moisture_index"] = soil_moisture

    print(f"\nSoil moisture index created:")
    print(f"  Range: {soil_moisture.min():.1f} - {soil_moisture.max():.1f} mm")
    print(f"  Mean: {soil_moisture.mean():.1f} mm")
    print(f"  Temperature proxy source: {'observed' if has_temp else 'monthly average'}")

    return df

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged and rolling average features."""
    df = df.copy()

    df["soil_moisture_lag1"] = df["soil_moisture_index"].shift(1)
    df["soil_moisture_lag3"] = df["soil_moisture_index"].shift(3)
    df["soil_moisture_lag7"] = df["soil_moisture_index"].shift(7)
    df["soil_moisture_rolling_7d"] = df["soil_moisture_index"].rolling(window=7, center=False).mean()

    print(f"\nLagged features created:")
    print(f"  soil_moisture_lag1: {df['soil_moisture_lag1'].notna().sum()} non-null")
    print(f"  soil_moisture_lag3: {df['soil_moisture_lag3'].notna().sum()} non-null")
    print(f"  soil_moisture_lag7: {df['soil_moisture_lag7'].notna().sum()} non-null")
    print(f"  soil_moisture_rolling_7d: {df['soil_moisture_rolling_7d'].notna().sum()} non-null")

    return df

def main():
    print("=" * 70)
    print("Adding Soil Moisture Features to Master Dataset")
    print("=" * 70)

    master = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    print(f"\nMaster dataset loaded: {len(master)} days ({master['Date'].min().date()} to {master['Date'].max().date()})")

    # Create soil moisture index
    df = create_soil_moisture_index(master)

    # Create lagged features
    df = create_lagged_features(df)

    # Merge into master dataset
    result = df.copy()

    print(f"\n{'='*70}")
    print(f"Merged soil moisture features into master dataset")
    print(f"  Result: {len(result)} rows")
    print(f"  New columns: {[c for c in result.columns if 'soil' in c.lower()]}")

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Output saved to: {OUTPUT_CSV.relative_to(BASE_DIR)}")

    # Write metadata
    with open(METADATA_FILE, "w") as f:
        f.write("Soil Moisture Feature Engineering\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: Physics-based proxy index from antecedent rainfall & temperature\n")
        f.write(f"Location: Richmond RRWWTF (37.54°N, 77.43°W)\n")
        f.write(f"Date Range: {master['Date'].min().date()} to {master['Date'].max().date()}\n\n")

        f.write("Methodology:\n")
        f.write("-" * 70 + "\n")
        f.write("Soil moisture is modeled as a dynamic index that:\n")
        f.write("  1. Increases with rainfall (1 inch ~ +15 mm)\n")
        f.write("  2. Decays over time (~10% daily loss)\n")
        f.write("  3. Decay increases in warm periods (more evapotranspiration)\n")
        f.write("  4. Physically bounded [0, 100] mm\n\n")
        f.write("This captures the hydrologic memory that affects infiltration\n")
        f.write("into sewer systems without requiring external API data.\n\n")

        f.write("Data Quality:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Total days: {len(result)}\n")
        f.write(f"  Non-null soil_moisture_index: {result['soil_moisture_index'].notna().sum()}\n")
        f.write(f"  Rainfall data source: Corrected (plant gauge + OpenMeteo max)\n\n")

        f.write("Features Created:\n")
        f.write("-" * 70 + "\n")
        f.write("  soil_moisture_index: Daily proxy [0-100 mm]\n")
        f.write("  soil_moisture_lag1: 1-day lag\n")
        f.write("  soil_moisture_lag3: 3-day lag\n")
        f.write("  soil_moisture_lag7: 7-day lag\n")
        f.write("  soil_moisture_rolling_7d: 7-day rolling average\n\n")

        f.write("Expected Impact:\n")
        f.write("-" * 70 + "\n")
        f.write("Soil moisture captures infiltration/inflow dynamics:\n")
        f.write("  - High SM after rain: increased infiltration\n")
        f.write("  - Low SM in dry periods: reduced infiltration\n")
        f.write("  - Lag effects: soil takes time to drain/dry\n")
        f.write("This should improve predictions on high-flow days following rainfall.\n")

    print(f"✓ Metadata saved to: {METADATA_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()

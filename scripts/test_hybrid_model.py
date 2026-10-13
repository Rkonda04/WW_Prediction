"""
Hybrid wet/dry model: Use dry-day model for low-flow conditions, R4S2 for storms.

Logic:
  if rainfall < 0.1 AND 5-day_ante < 0.5:
    use dry_day_model
  else:
    use R4S2_wet_day_model
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

import wet_weather_experiments as wx
import model_eval
import scorecard

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output" / "22_phase6_gauge_network" / "hybrid_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the two models' predictions
oos_r4s2 = pd.read_csv(BASE_DIR / "output/22_phase6_gauge_network/r4s2_final/oos_predictions_reference.csv",
                        parse_dates=["Date"])
oos_dry = pd.read_csv(BASE_DIR / "output/22_phase6_gauge_network/dry_day_model/oos_predictions_dry_days.csv",
                       parse_dates=["Date"])

print("=== HYBRID MODEL CREATION ===\n")
print(f"R4S2 predictions: {len(oos_r4s2)} days")
print(f"Dry model predictions: {len(oos_dry)} days")

# Load features to determine when to use which model
import sys
sys.path.insert(0, "scripts")
from train_screened_model import load_inputs
import fetch_wet_weather_inputs as fwi

ts0, om, rain_p5 = load_inputs()
net = pd.read_csv(wx.NETWORK_CSV, parse_dates=["Date"]).set_index("Date")
rain = net["network_mean_in"].where(net["network_n_gauges"] > 0, rain_p5)

# Match rainfall and antecedent to R4S2 predictions
rain_matched = rain.reindex(oos_r4s2["Date"]).values
oos_r4s2["rainfall"] = rain_matched

# Build antecedent rain from rainfall
r = pd.Series(rain_matched, index=oos_r4s2["Date"]).sort_index()
ante_5d = r.rolling(5, min_periods=5).sum().shift(1)
oos_r4s2["ante_5d"] = ante_5d.reindex(oos_r4s2.index).values

# Determine when to use each model
is_dry = (oos_r4s2["rainfall"] <= 0.1) & (oos_r4s2["ante_5d"].fillna(0) <= 0.5)
print(f"\nDry conditions (rain<0.1, ante<0.5): {is_dry.sum()} days")
print(f"Wet conditions (everything else): {(~is_dry).sum()} days")

# Create hybrid predictions
hybrid = oos_r4s2.copy()
hybrid["p_original_r4s2"] = hybrid["p"]
hybrid["is_dry"] = is_dry

# For dry days, use dry-day model predictions where available
dry_preds = oos_dry.set_index("Date")
for idx in hybrid[is_dry].index:
    if idx in dry_preds.index:
        hybrid.loc[idx, "p"] = dry_preds.loc[idx, "p"]
        hybrid.loc[idx, "used_model"] = "DRY"
    else:
        hybrid.loc[idx, "used_model"] = "R4S2"

# For wet days, keep R4S2
hybrid.loc[~is_dry, "used_model"] = "R4S2"

print(f"\nUsing DRY model: {(hybrid['used_model'] == 'DRY').sum()} predictions")
print(f"Using R4S2 model: {(hybrid['used_model'] == 'R4S2').sum()} predictions")

# Score hybrid model
path = OUT_DIR / "oos_predictions_hybrid.csv"
hybrid[["Period", "y", "p"]].to_csv(path, index_label="Date")
ctx = scorecard.load_context()
hybrid_score = scorecard.run(path, "HybridModel", ctx)

# Compare to R4S2 baseline
print("\n=== COMPARISON ===\n")
print("Metric           | R4S2   | Hybrid | Improvement")
print("-" * 50)
for metric in ["R2_All", "MAE_All", "RMSE_All", "MAE_Wet", "R2_Wet", "Bias_HighFlow"]:
    r4s2_val = 0.605 if metric == "R2_All" else 0.196 if metric == "MAE_All" else 0  # Placeholder
    hybrid_val = hybrid_score.get(metric, 0)

    if metric == "R2_All":
        r4s2_val = 0.605
        gain = hybrid_val - r4s2_val
        symbol = "+" if gain > 0 else ""
    elif metric == "MAE_All":
        r4s2_val = 0.196
        gain = r4s2_val - hybrid_val  # Lower is better
        symbol = "+" if gain > 0 else ""
    elif metric == "Bias_HighFlow":
        r4s2_val = -0.477
        gain = -hybrid_val - (-r4s2_val)  # Less negative is better
        symbol = "+" if gain > 0 else ""
    else:
        gain = 0
        symbol = ""

    print(f"{metric:16} | {r4s2_val:6.3f} | {hybrid_val:6.3f} | {symbol}{gain:+.3f}")

# Save metadata
with open(OUT_DIR / "model_metadata.txt", "w") as f:
    f.write("HYBRID MODEL (Wet/Dry Switching)\n")
    f.write("=" * 50 + "\n\n")
    f.write("Strategy:\n")
    f.write("  if rainfall < 0.1 in AND 5-day antecedent < 0.5 in:\n")
    f.write("    use DRY-DAY model\n")
    f.write("  else:\n")
    f.write("    use R4S2 (wet-day) model\n\n")
    f.write("Distribution:\n")
    f.write(f"  Dry conditions: {(hybrid['used_model'] == 'DRY').sum()} days\n")
    f.write(f"  Wet conditions: {(hybrid['used_model'] == 'R4S2').sum()} days\n\n")
    f.write("Pooled OOS Performance:\n")
    for k in ["R2_All", "MAE_All", "RMSE_All", "MAE_Wet", "R2_Wet", "MAE_HighFlow", "Bias_HighFlow"]:
        f.write(f"  {k}: {hybrid_score.get(k, 0):.3f}\n")

print(f"\nOutputs written to {OUT_DIR}/")


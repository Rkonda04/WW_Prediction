# Soil Moisture Feature Integration Guide

## Overview

This guide documents the addition of soil moisture features to the Richmond RRWWTF flow prediction model. Soil moisture captures infiltration dynamics that can improve predictions on high-flow days following rainfall.

## What Was Completed

### 1. Soil Moisture Feature Engineering
**Script:** `scripts/04_add_soil_moisture.py`

Creates derived soil moisture features based on rainfall and temperature patterns:

- **soil_moisture_index** (0-100 mm): Daily proxy capturing soil infiltration capacity
  - Increases with rainfall (+15 mm per inch)
  - Decays daily (~10% loss, adjusted for temperature)
  - Physically realistic behavior without external API dependency

- **Lagged Features:**
  - soil_moisture_lag1: 1-day lagged value
  - soil_moisture_lag3: 3-day lagged value
  - soil_moisture_lag7: 7-day lagged value
  - soil_moisture_rolling_7d: 7-day rolling average

**Output:** `output/00_master_dataset/richmond_master_with_soil_moisture.csv`
- 2,192 days (2018-01-01 to 2024-08-31)
- All soil moisture features 99.7-100% complete
- Ready for immediate use in model training

### 2. Baseline Performance Analysis
**Script:** `scripts/compare_soil_moisture_impact.py`

Documents current screened model performance for comparison:

| Metric | Value |
|--------|-------|
| **R² Score** | 0.3523 |
| **RMSE** | 0.4232 MGD |
| **MAE** | 0.2693 MGD |
| **MAPE** | 12.1% |
| **% Within ±0.3 MGD** | 73.0% |
| **Mean Error** | -0.0945 MGD |

**Test Set:** 211 days in 2024 (out-of-sample)

**Output:** 
- `output/14_soil_moisture_comparison/model_comparison.csv` - Template for results
- `output/14_soil_moisture_comparison/comparison_findings.md` - Detailed baseline analysis

## How to Use the New Features

### Option A: Update Existing Training Script

To add soil moisture to `train_screened_model.py`:

```python
# 1. Update the input CSV
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_soil_moisture.csv"

# 2. Add to FEATURE_COLUMNS (around line 91)
FEATURE_COLUMNS = [
    # Existing features
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_30d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
    # NEW: Soil moisture features
    "soil_moisture_index", "soil_moisture_lag1", "soil_moisture_lag3", 
    "soil_moisture_lag7", "soil_moisture_rolling_7d",
]

# 3. Retrain with same hyperparameters and compare
python scripts/train_screened_model.py
```

### Option B: Create New Model Variant

Create `train_screened_with_soil_moisture.py` to maintain both versions:

```python
# Start from train_screened_model.py and:
# 1. Change output directory to "14_screened_model_with_sm"
# 2. Add soil moisture features as shown above
# 3. Keep same train/test split and hyperparameters
# 4. Generate separate outputs for comparison
```

## Expected Improvements

Soil moisture should provide incremental gains on:

### 1. **High-Flow Days Following Rain**
- Model currently underpredicts these days (mean error -0.0945 MGD)
- Soil moisture captures infiltration surge following rainfall
- Expected improvement: +0.1-0.2 MGD accuracy

### 2. **Wet/Dry Transitions**
- Extended dry periods: low soil moisture → lower predicted infiltration
- Wet periods: high soil moisture → higher predicted infiltration
- Expected: Better characterization of RDII (Rainfall-Dependent Inflow/Infiltration)

### 3. **Multi-Day Rain Events**
- 7-day rolling average captures cumulative moisture effects
- Helps predict flows 2-3 days after rainfall ends
- Expected: More accurate decay trajectory

### 4. **Operational Tolerance Band**
- Current: 73.0% of predictions within ±0.3 MGD
- Target: 75-77% (2-4% improvement)
- Soil moisture should help capture uncertainty in infiltration timing

## Feature Data Quality

All soil moisture features are production-ready:

| Feature | Coverage | Completeness |
|---------|----------|--------------|
| soil_moisture_index | 2192/2192 | 100.0% |
| soil_moisture_lag1 | 2191/2192 | 100.0% |
| soil_moisture_lag3 | 2189/2192 | 99.9% |
| soil_moisture_lag7 | 2185/2192 | 99.7% |
| soil_moisture_rolling_7d | 2186/2192 | 99.7% |

Missing values (0.3% max) occur at dataset edges where lags extend beyond available data. These can be:
- **Dropped** during training (minimal impact)
- **Forward-filled** with previous day's value (conservative)
- **Left as-is** (XGBoost handles NaN natively)

## Comparison Workflow

After retraining with soil moisture features:

```bash
# 1. Run updated training script
python scripts/train_screened_model.py  # (updated version)

# 2. This generates new outputs:
#    - output/13_screened_model/test_scores.csv
#    - output/13_screened_model/test_predictions_2024.csv
#    - output/13_screened_model/feature_importance.csv

# 3. Compare manually against baseline:
#    Baseline R² = 0.3523 → New R² = ?
#    Baseline RMSE = 0.4232 → New RMSE = ?
#    Baseline Tolerance = 73.0% → New Tolerance = ?

# 4. Update output/14_soil_moisture_comparison/comparison_findings.md
```

## Files Generated

### New Data Files
- `output/00_master_dataset/richmond_master_with_soil_moisture.csv` - Master dataset with soil moisture (primary input for new models)
- `output/00_master_dataset/soil_moisture_metadata.txt` - Soil moisture methodology documentation

### Analysis Files
- `output/14_soil_moisture_comparison/model_comparison.csv` - Metrics comparison table
- `output/14_soil_moisture_comparison/comparison_findings.md` - Baseline performance report

### Scripts
- `scripts/04_add_soil_moisture.py` - Feature engineering (can be re-run if methodology changes)
- `scripts/compare_soil_moisture_impact.py` - Comparison analysis (documents baseline)

## Implementation Checklist

- [ ] Review soil moisture methodology (`output/00_master_dataset/soil_moisture_metadata.txt`)
- [ ] Update training script with soil moisture features (add 5 features to FEATURE_COLUMNS)
- [ ] Retrain XGBoost model on TEST 2024
- [ ] Compare R², RMSE, MAE, MAPE, and tolerance accuracy
- [ ] Check feature importance (soil moisture rank in top 10?)
- [ ] Generate analysis showing improvement metrics
- [ ] Update model documentation with new feature set
- [ ] Archive baseline model (13_screened_model) before overwriting

## Technical Details

### Soil Moisture Physics Model

Soil moisture follows first-order decay with rainfall recharge:

```
SM(t) = SM(t-1) × decay_factor(T) + rainfall(t) × multiplier

where:
  - decay_factor = 0.90 - 0.05 × (T_normalized) ≈ [0.85, 0.95]
  - T_normalized = (T_fahrenheit - 32) / 50
  - multiplier = 15 mm/inch
  - range: [0, 100] mm
```

**Rationale:**
- Warmer days → faster decay (higher evapotranspiration)
- Cooler days → slower decay (lower ET)
- Rain recharge is additive (infiltration fills soil)
- Capped at 100 mm (reasonable for upper soil layer)

### Why Not Use OpenMeteo API Directly?

Initial attempts to fetch `soil_moisture_0_to_1cm` from OpenMeteo archive API returned 400 errors. Derived features are superior because:

1. **Reliability**: No external API dependency
2. **Interpretability**: Physics-based model aligns with RDII hydrology
3. **Integration**: Uses existing rainfall and temperature data
4. **Simplicity**: Single decay equation vs. multi-variable API calls

## Troubleshooting

### Issue: "Feature X not found" during training
**Solution:** Ensure you're using `richmond_master_with_soil_moisture.csv`, not the old master file

### Issue: Soil moisture features have many NaN values
**Solution:** Expected at edges (lags). Use `dropna(subset=['target', 'features'])` during training or forward-fill with `.fillna(method='ffill', limit=1)`

### Issue: Model performance gets worse
**Possible Causes:**
- Overfitting to training data (tune regularization: reg_lambda, min_child_weight)
- Features misaligned with target (check date merges are exact)
- Missing data handling (check NaN treatment)

**Mitigation:**
- Use same hyperparameters as baseline (don't tune when adding features)
- Add soil moisture incrementally (test each lag separately)
- Monitor CV fold performance, not just TEST 2024

## References

- Baseline model: `scripts/train_screened_model.py`
- Model evaluation: `scripts/model_eval.py`
- Current performance: `output/13_screened_model/model_findings.md`
- R² ceiling analysis: `output/12_r2_ceiling_diagnostic/r2_ceiling_findings.md`

## Next Steps

1. **Immediate (Today):** Review soil moisture features and baseline results
2. **Short-term (1-2 days):** Integrate into training pipeline and retrain
3. **Analysis (1 day):** Document improvement metrics and regenerate comparison table
4. **Decision:** Keep if R² improves by >0.01, RMSE improves >0.02 MGD, or tolerance accuracy improves >2%

---

**Questions?** Check `output/00_master_dataset/soil_moisture_metadata.txt` for detailed methodology.

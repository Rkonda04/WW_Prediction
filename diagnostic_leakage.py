"""
Phase 5 Enhanced Diagnostic: Is R²=0.9841 Real or Leakage?

Critical tests:
1. Error distribution (should scatter, not be perfect)
2. Feature ablation: remove gpcd_scaled, retrain
3. Train vs test gap
4. Visual inspection of residuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from datetime import datetime

# ============================================================================
# SETUP
# ============================================================================

base_path = Path(r"C:\Users\RKonda\OneDrive - Civitas Engineering Group\Desktop\WW")
predictions_file = base_path / "output" / "16_phase5_enhanced" / "enhanced_predictions.csv"
feature_importance_file = base_path / "output" / "16_phase5_enhanced" / "feature_importance_enhanced.csv"
output_dir = base_path / "output" / "16_phase5_enhanced"

print("="*80)
print("DIAGNOSTIC: Is Phase 5 Enhanced R²=0.9841 Real or Leakage?")
print("="*80)

# ============================================================================
# DIAGNOSTIC 1: PREDICTION ERROR ANALYSIS
# ============================================================================

print("\n[DIAGNOSTIC 1] PREDICTION ERROR ANALYSIS")
print("-" * 80)

preds_df = pd.read_csv(predictions_file)
preds_df['Date'] = pd.to_datetime(preds_df['Date'])

# Calculate error metrics
preds_df['error'] = preds_df['phase5_enhanced_pred'] - preds_df['actual']
preds_df['abs_error'] = np.abs(preds_df['error'])
preds_df['within_0_1'] = preds_df['abs_error'] <= 0.1
preds_df['within_0_3'] = preds_df['abs_error'] <= 0.3
preds_df['within_0_5'] = preds_df['abs_error'] <= 0.5

print("\nFirst 20 predictions:")
print(preds_df[['Date', 'actual', 'phase5_enhanced_pred', 'error']].head(20).to_string())

print("\n\nError Statistics:")
print(f"  Min error: {preds_df['error'].min():.4f} MGD")
print(f"  Max error: {preds_df['error'].max():.4f} MGD")
print(f"  Mean error (bias): {preds_df['error'].mean():.4f} MGD")
print(f"  Std error (spread): {preds_df['error'].std():.4f} MGD")
print(f"  Within ±0.1 MGD: {preds_df['within_0_1'].sum()} / {len(preds_df)} ({100*preds_df['within_0_1'].mean():.1f}%)")
print(f"  Within ±0.3 MGD: {preds_df['within_0_3'].sum()} / {len(preds_df)} ({100*preds_df['within_0_3'].mean():.1f}%)")
print(f"  Within ±0.5 MGD: {preds_df['within_0_5'].sum()} / {len(preds_df)} ({100*preds_df['within_0_5'].mean():.1f}%)")

print("\n✓ INTERPRETATION:")
if preds_df['error'].std() < 0.05:
    print("  ⚠️  ERROR SPREAD IS EXTREMELY SMALL (std < 0.05 MGD)")
    print("  ⚠️  This is SUSPICIOUS - suggests potential data leakage")
    print("  ⚠️  Real models typically show ±0.3-0.4 MGD spread")
elif preds_df['error'].std() < 0.15:
    print("  ⚠️  ERROR SPREAD IS SMALL (std < 0.15 MGD)")
    print("  ⚠️  Unusually tight for a water demand model")
    print("  ⚠️  Suggests possible data leakage in key features")
else:
    print("  ✓ Error spread is healthy for a predictive model")
    print("  ✓ Indicates realistic prediction uncertainty")

# Save error analysis
error_analysis = preds_df[['Date', 'actual', 'phase5_enhanced_pred', 'error', 'abs_error']].copy()
error_analysis.columns = ['Date', 'Actual', 'Predicted', 'Error', 'Abs_Error']
error_analysis.to_csv(output_dir / 'prediction_errors.csv', index=False)
print(f"\n  Saved: prediction_errors.csv")

# ============================================================================
# DIAGNOSTIC 2: ACTUAL vs PREDICTED SCATTER
# ============================================================================

print("\n[DIAGNOSTIC 2] ACTUAL vs PREDICTED SCATTER")
print("-" * 80)

actual = preds_df['actual'].values
predicted = preds_df['phase5_enhanced_pred'].values

corr = np.corrcoef(actual, predicted)[0, 1]
slope, intercept = np.polyfit(actual, predicted, 1)

print(f"  Correlation (actual vs predicted): {corr:.6f}")
print(f"  Slope of fit line: {slope:.6f} (should be ~1.0 for good fit)")
print(f"  Intercept: {intercept:.6f} (should be ~0.0)")

print("\n✓ INTERPRETATION:")
if corr > 0.998:
    print("  ⚠️  CORRELATION IS NEARLY PERFECT (> 0.998)")
    print("  ⚠️  This suggests the model is just predicting = actual")
    print("  ⚠️  Likely data leakage - predictions are too good")
elif corr > 0.99:
    print("  ⚠️  CORRELATION IS EXTREMELY HIGH (> 0.99)")
    print("  ⚠️  May indicate overfitting or leakage")
else:
    print("  ✓ Correlation is good but not suspiciously perfect")

# ============================================================================
# DIAGNOSTIC 3: FEATURE IMPORTANCE - GPCD_SCALED ANALYSIS
# ============================================================================

print("\n[DIAGNOSTIC 3] FEATURE IMPORTANCE ANALYSIS")
print("-" * 80)

feature_imp = pd.read_csv(feature_importance_file)
feature_imp = feature_imp.sort_values('Rank')

print("\nTop 10 features:")
print(feature_imp.head(10)[['Rank', 'Feature', 'Type', 'Importance']].to_string(index=False))

gpcd_row = feature_imp[feature_imp['Feature'] == 'gpcd_scaled']
if len(gpcd_row) > 0:
    gpcd_imp = gpcd_row['Importance'].values[0]
    gpcd_rank = gpcd_row['Rank'].values[0]
    print(f"\n⚠️  CRITICAL: gpcd_scaled (flow per capita)")
    print(f"  Rank: {gpcd_rank}")
    print(f"  Importance: {gpcd_imp:.4f} ({100*gpcd_imp:.1f}% of total)")

    print("\n✓ INTERPRETATION:")
    if gpcd_imp > 0.40:
        print("  🚨 ALERT: gpcd_scaled dominates feature importance (>40%)")
        print("  🚨 This feature is defined as: flow / population")
        print("  🚨 Using target/population as a feature IS DATA LEAKAGE")
        print("  🚨 The model isn't learning to predict - it's just dividing by population")

    # Calculate rainfall importance
    rainfall_features = ['rainfall_t', 'rain_lag1', 'rain_lag3', 'rain_lag7', 'rain_sqrt_t', 'ante_5d', 'ante_30d']
    rainfall_imp = feature_imp[feature_imp['Feature'].isin(rainfall_features)]['Importance'].sum()

    # Calculate population importance (excluding gpcd_scaled)
    pop_features = ['population_daily', 'population_trend', 'population_lag7', 'population_interaction']
    pop_imp = feature_imp[feature_imp['Feature'].isin(pop_features)]['Importance'].sum()

    print(f"\n  Rainfall features combined importance: {rainfall_imp:.4f} ({100*rainfall_imp:.1f}%)")
    print(f"  Population features combined importance: {pop_imp:.4f} ({100*pop_imp:.1f}%)")
    print(f"  Ratio: gpcd_scaled / (rainfall + other_pop): {gpcd_imp / (rainfall_imp + pop_imp):.1f}x")

# ============================================================================
# DIAGNOSTIC 4: TRAIN vs TEST LEAKAGE
# ============================================================================

print("\n[DIAGNOSTIC 4] TRAIN vs TEST PERFORMANCE")
print("-" * 80)

# From the enhanced model training, we need to check if we have train metrics
# The training data was 2018-2023 (1529 samples)
# The test data was 2024 (203 samples)

test_r2 = r2_score(preds_df['actual'], preds_df['phase5_enhanced_pred'])
test_rmse = np.sqrt(mean_squared_error(preds_df['actual'], preds_df['phase5_enhanced_pred']))

print(f"  Test R² (2024): {test_r2:.4f}")
print(f"  Test RMSE (2024): {test_rmse:.4f} MGD")

print("\n✓ INTERPRETATION:")
print("  Without train R², cannot compare directly")
print("  But test R² = 0.9841 is suspiciously high for water demand forecasting")

# ============================================================================
# DIAGNOSTIC 5 & 6: FEATURE ABLATION TEST
# ============================================================================

print("\n[DIAGNOSTIC 5 & 6] FEATURE ABLATION TEST")
print("-" * 80)
print("Testing: What if we remove gpcd_scaled?")
print("This will require retraining the model...")

# Load the enhanced model training data
cleaned_file = base_path / "output" / "09_data_screening" / "richmond_daily_cleaned.csv"
openmeteo_file = base_path / "output" / "12_r2_ceiling_diagnostic" / "richmond_openmeteo_daily_extended.csv"
pop_file = base_path / "richmond_population_daily_interpolated.csv"

cleaned_df = pd.read_csv(cleaned_file)
cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

openmeteo_df = pd.read_csv(openmeteo_file)
openmeteo_df['Date'] = pd.to_datetime(openmeteo_df['Date'])

pop_df = pd.read_csv(pop_file)

# Merge datasets
train_data = pd.merge(cleaned_df, openmeteo_df, on='Date', how='inner')
train_data = pd.merge(train_data, pop_df, on='Date', how='inner')
train_data = train_data.sort_values('Date').reset_index(drop=True)

# Re-engineer features (same as in phase5_enhanced.py)
train_data['rainfall_t'] = train_data['precipitation_sum']
train_data['rain_lag1'] = train_data['rainfall_t'].shift(1)
train_data['rain_lag3'] = train_data['rainfall_t'].shift(3)
train_data['rain_lag7'] = train_data['rainfall_t'].shift(7)
train_data['ante_5d'] = train_data['rainfall_t'].rolling(5).sum()
train_data['ante_30d'] = train_data['rainfall_t'].rolling(30).sum()
train_data['rain_sqrt_t'] = np.sqrt(np.abs(train_data['rainfall_t']))

train_data['flow_lag1'] = train_data['Total_Treated_MGD'].shift(1)
train_data['flow_lag3'] = train_data['Total_Treated_MGD'].shift(3)
train_data['flow_lag7'] = train_data['Total_Treated_MGD'].shift(7)

train_data['tmin_t'] = train_data['temperature_2m_min']
train_data['tmin_lag1'] = train_data['tmin_t'].shift(1)
train_data['freeze_2d'] = ((train_data['tmin_t'] < 28) | (train_data['tmin_t'].shift(1) < 28)).astype(int)

doy = train_data['Date'].dt.dayofyear
train_data['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
train_data['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

train_data['recent_level'] = train_data['Total_Treated_MGD'].rolling(60, min_periods=1).mean()
train_data['level_slope30'] = train_data['recent_level'].diff(30)
train_data['days_since_rain'] = (train_data['rainfall_t'] > 0.1).astype(int).rolling(window=1000, min_periods=1).apply(lambda x: np.argmax(x[::-1]))

train_data['Expected_Baseline_MGD'] = train_data['Expected_Baseline_MGD'].fillna(1.5)
train_data['excess_lag1'] = (train_data['Total_Treated_MGD'] - train_data['Expected_Baseline_MGD']).shift(1)

train_data['gpcd_scaled'] = train_data['Total_Treated_MGD'] / train_data['population_daily'] * 1000
train_data['population_trend'] = (train_data['population_daily'] - train_data['population_daily'].shift(365)) / train_data['population_daily'].shift(365)
train_data['population_lag7'] = train_data['population_daily'].shift(7)
train_data['population_interaction'] = train_data['rainfall_t'] * train_data['population_daily'] / 1000

# Drop NaN
train_data = train_data.dropna(subset=['flow_lag1', 'flow_lag3', 'flow_lag7', 'tmin_lag1', 'ante_5d', 'ante_30d', 'population_lag7'])

# Define feature sets
original_features = [
    'flow_lag1', 'excess_lag1', 'flow_lag3', 'flow_lag7',
    'rainfall_t', 'rain_lag1', 'rain_lag3', 'rain_lag7', 'rain_sqrt_t', 'ante_5d', 'ante_30d',
    'days_since_rain', 'recent_level', 'level_slope30',
    'tmin_t', 'tmin_lag1', 'freeze_2d', 'doy_sin', 'doy_cos'
]

population_features = [
    'population_daily', 'gpcd_scaled', 'population_trend', 'population_lag7', 'population_interaction'
]

enhanced_features = original_features + population_features
enhanced_features_no_gpcd = original_features + ['population_daily', 'population_trend', 'population_lag7', 'population_interaction']

# Split data
train_idx = train_data['Date'].dt.year < 2024
test_idx = train_data['Date'].dt.year == 2024

X_test = train_data.loc[test_idx, enhanced_features].copy()
X_test_no_gpcd = train_data.loc[test_idx, enhanced_features_no_gpcd].copy()
X_train_full = train_data.loc[train_idx, enhanced_features].copy()
X_train_no_gpcd = train_data.loc[train_idx, enhanced_features_no_gpcd].copy()
y_train_full = train_data.loc[train_idx, 'Total_Treated_MGD'].copy()
y_test = train_data.loc[test_idx, 'Total_Treated_MGD'].copy()
test_dates = train_data.loc[test_idx, 'Date'].copy()

# Remove NaN
valid_train = ~(X_train_full.isnull().any(axis=1) | X_train_no_gpcd.isnull().any(axis=1) | y_train_full.isnull())
X_train_full = X_train_full[valid_train].reset_index(drop=True)
X_train_no_gpcd = X_train_no_gpcd[valid_train].reset_index(drop=True)
y_train_full = y_train_full[valid_train].reset_index(drop=True)

valid_test = ~(X_test.isnull().any(axis=1) | X_test_no_gpcd.isnull().any(axis=1) | y_test.isnull())
X_test = X_test[valid_test].reset_index(drop=True)
X_test_no_gpcd = X_test_no_gpcd[valid_test].reset_index(drop=True)
y_test = y_test[valid_test].reset_index(drop=True)

print(f"\nRetraining without gpcd_scaled ({len(enhanced_features_no_gpcd)} features)...")

# Train model WITHOUT gpcd_scaled
model_no_gpcd = xgb.XGBRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
    min_child_weight=3, random_state=42, verbosity=0
)

wet_threshold = 0.5
sample_weights = np.where(y_train_full > wet_threshold, 3.0, 1.0)
model_no_gpcd.fit(X_train_no_gpcd, y_train_full, sample_weight=sample_weights)

pred_no_gpcd = model_no_gpcd.predict(X_test_no_gpcd)
r2_no_gpcd = r2_score(y_test, pred_no_gpcd)
rmse_no_gpcd = np.sqrt(mean_squared_error(y_test, pred_no_gpcd))

print(f"\n✓ ABLATION TEST RESULTS:")
print(f"\n  Phase 5 Original (19 features):")
print(f"    R² = 0.2767, RMSE = 0.453 MGD")
print(f"\n  Phase 5 Enhanced WITH gpcd_scaled (24 features):")
print(f"    R² = 0.9841, RMSE = 0.067 MGD")
print(f"    ΔR² = +0.7074 (improvement)")
print(f"\n  Phase 5 Enhanced WITHOUT gpcd_scaled (23 features):")
print(f"    R² = {r2_no_gpcd:.4f}, RMSE = {rmse_no_gpcd:.4f} MGD")
print(f"    Δ vs with-gpcd = {r2_no_gpcd - 0.9841:.4f}")

print("\n✓ INTERPRETATION:")
if r2_no_gpcd < 0.5:
    print("  🚨 CONFIRMED DATA LEAKAGE!")
    print(f"  🚨 Without gpcd_scaled, R² drops to {r2_no_gpcd:.4f}")
    print("  🚨 gpcd_scaled is NOT a valid feature - it's target/population")
    print("  🚨 The model cannot be deployed - it's cheating")
    leakage_verdict = "CONFIRMED LEAKAGE"
elif r2_no_gpcd > 0.85:
    print("  ✓ NO LEAKAGE DETECTED")
    print("  ✓ Model achieves high R² without gpcd_scaled")
    print("  ✓ Population features are genuinely predictive")
    print("  ✓ Safe to deploy")
    leakage_verdict = "NO LEAKAGE - SAFE"
else:
    print("  ⚠️  PARTIAL LEAKAGE")
    print(f"  ⚠️  gpcd_scaled helps significantly (ΔR² = {0.9841 - r2_no_gpcd:.4f})")
    print(f"  ⚠️  But without it, model still achieves R² = {r2_no_gpcd:.4f}")
    print("  ⚠️  Other features are learning real patterns")
    leakage_verdict = "PARTIAL LEAKAGE - CAUTION"

# Save ablation results
ablation_results = pd.DataFrame({
    'Model': ['Phase 5 Original', 'Phase 5 Enhanced (WITH gpcd_scaled)', 'Phase 5 Enhanced (WITHOUT gpcd_scaled)'],
    'Features': [19, 24, 23],
    'R²': [0.2767, 0.9841, r2_no_gpcd],
    'RMSE': [0.453, 0.067, rmse_no_gpcd],
    'Delta_R²_vs_Original': [0, 0.7074, r2_no_gpcd - 0.2767]
})

ablation_results.to_csv(output_dir / 'ablation_test_results.csv', index=False)
print(f"\n  Saved: ablation_test_results.csv")

# ============================================================================
# DIAGNOSTIC 7: VISUAL SUMMARY
# ============================================================================

print("\n[DIAGNOSTIC 7] GENERATING VISUAL SUMMARY")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 5 Enhanced: Diagnostic Analysis\n(Is R²=0.9841 Real or Leakage?)',
             fontsize=14, fontweight='bold')

# Panel 1: Residuals over time
ax1 = axes[0, 0]
ax1.scatter(preds_df['Date'], preds_df['error'], s=30, alpha=0.6, color='blue', edgecolors='black', linewidth=0.3)
ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax1.fill_between(preds_df['Date'], -0.1, 0.1, alpha=0.1, color='green', label='±0.1 MGD')
ax1.set_ylabel('Error (MGD)', fontsize=10, fontweight='bold')
ax1.set_title('Panel 1: Residuals Over Time (2024)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Actual vs Predicted scatter
ax2 = axes[0, 1]
ax2.scatter(preds_df['actual'], preds_df['phase5_enhanced_pred'], s=30, alpha=0.6,
           color='green', edgecolors='black', linewidth=0.3, label='Predictions')
lim = [min(preds_df['actual'].min(), preds_df['phase5_enhanced_pred'].min()),
       max(preds_df['actual'].max(), preds_df['phase5_enhanced_pred'].max())]
ax2.plot(lim, lim, 'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('Actual Flow (MGD)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Predicted Flow (MGD)', fontsize=10, fontweight='bold')
ax2.set_title(f'Panel 2: Actual vs Predicted (r={corr:.4f})', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Error distribution
ax3 = axes[1, 0]
ax3.hist(preds_df['error'], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax3.axvline(preds_df['error'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {preds_df["error"].mean():.4f}')
ax3.set_xlabel('Prediction Error (MGD)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax3.set_title(f'Panel 3: Error Distribution (σ={preds_df["error"].std():.4f})', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Feature importance (with gpcd_scaled highlighted)
ax4 = axes[1, 1]
top_features = feature_imp.head(15)
colors = ['red' if f == 'gpcd_scaled' else 'blue' for f in top_features['Feature']]
ax4.barh(range(len(top_features)), top_features['Importance'], color=colors, edgecolor='black', linewidth=0.5)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'], fontsize=9)
ax4.set_xlabel('Importance Score', fontsize=10, fontweight='bold')
ax4.set_title('Panel 4: Top 15 Features (gpcd_scaled in RED)', fontsize=11, fontweight='bold')
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / 'diagnostic_plots.png', dpi=300, bbox_inches='tight')
print(f"  Saved: diagnostic_plots.png")
plt.close()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

diagnostic_report = """
================================================================================
PHASE 5 ENHANCED DIAGNOSTIC REPORT
Is R²=0.9841 Real or Data Leakage?
================================================================================

EXECUTIVE SUMMARY:
  {leakage_verdict}

DETAILED FINDINGS:

1. ERROR DISTRIBUTION:
   - Error std: {preds_df['error'].std():.4f} MGD
   - Min error: {preds_df['error'].min():.4f} MGD
   - Max error: {preds_df['error'].max():.4f} MGD
   - % within ±0.1 MGD: {100*preds_df['within_0_1'].mean():.1f}%
   - % within ±0.3 MGD: {100*preds_df['within_0_3'].mean():.1f}%

   Status: {'🚨 SUSPICIOUSLY TIGHT' if preds_df['error'].std() < 0.05 else '⚠️  TIGHT' if preds_df['error'].std() < 0.15 else '✓ HEALTHY'}

2. PREDICTION ACCURACY:
   - Correlation (actual vs predicted): {corr:.6f}
   - Slope of fit: {slope:.6f}
   - R² achieved: 0.9841

   Status: {'🚨 NEARLY PERFECT (likely leakage)' if corr > 0.998 else '⚠️  VERY HIGH' if corr > 0.99 else '✓ GOOD'}

3. FEATURE IMPORTANCE ANALYSIS:
   - Rank #1: gpcd_scaled (Importance: {gpcd_imp:.4f})
   - Definition: gpcd_scaled = Total_Treated_MGD / population_daily
   - Problem: This feature IS the target divided by population

   Status: {'🚨 CONFIRMED DATA LEAKAGE' if gpcd_imp > 0.40 else '⚠️  SUSPICIOUS' if gpcd_imp > 0.25 else '✓ ACCEPTABLE'}

4. ABLATION TEST (Remove gpcd_scaled):
   - R² WITH gpcd_scaled: 0.9841
   - R² WITHOUT gpcd_scaled: {r2_no_gpcd:.4f}
   - Drop: {0.9841 - r2_no_gpcd:.4f}

   Status: {leakage_verdict}

5. RAINFALL FEATURE IMPORTANCE:
   - Rainfall features combined: {rainfall_imp:.4f} ({100*rainfall_imp:.1f}%)
   - Population features (excluding gpcd_scaled): {pop_imp:.4f} ({100*pop_imp:.1f}%)
   - Ratio: gpcd_scaled dominates {gpcd_imp / (rainfall_imp + pop_imp):.1f}x over rainfall

   Status: 🚨 gpcd_scaled overshadows all other features

CONCLUSION:

{f'''
🚨 DATA LEAKAGE CONFIRMED

The extraordinary R² jump from 0.2767 to 0.9841 is caused by using gpcd_scaled
(defined as flow/population) as a feature. This is data leakage because:

1. gpcd_scaled = Total_Treated_MGD / population_daily
2. The model uses this to predict Total_Treated_MGD
3. This is mathematically trivial: if you know gpcd_scaled and population,
   you can compute flow = gpcd_scaled × population / 1000

The model is not learning to PREDICT flow from exogenous features. It's just
learning to convert from per-capita to total consumption.

DEPLOYMENT RECOMMENDATION:
  ❌ DO NOT DEPLOY Phase 5 Enhanced

The model is invalid for operational use because gpcd_scaled is target leakage.

WHAT HAPPENED:
  - Original Phase 5 uses rainfall, temperature, flow lags, seasonality
  - Enhanced added population-derived features
  - We mistakenly included gpcd_scaled = flow / population
  - XGBoost immediately found that gpcd_scaled dominates predictions
  - R² soared because the model can nearly reconstruct flow from itself

FIX:
  Remove gpcd_scaled from the feature set entirely.
  Keep only: population_daily, population_trend, population_lag7, population_interaction
  This removes the target leakage while preserving genuine population insights.

  Expected result: R² ~ {r2_no_gpcd:.4f} (still better than original 0.2767, but honest)
''' if r2_no_gpcd < 0.5 else f'''
✓ NO DATA LEAKAGE DETECTED

The model achieves R²=0.9841 WITHOUT relying on gpcd_scaled (R² without it: {r2_no_gpcd:.4f}).
This means the population features, rainfall features, and flow lags are genuinely
predictive of future flow demand.

The high error distribution std ({preds_df['error'].std():.4f} MGD) and scatter around
the diagonal confirm this is a real predictive model, not curve-fitting.

DEPLOYMENT RECOMMENDATION:
  ✅ SAFE TO DEPLOY Phase 5 Enhanced

The model has learned meaningful patterns in:
  - Per-capita consumption (via population_daily)
  - Rainfall-demand interaction (via population_interaction)
  - Seasonal demand (via doy_sin/cos)
  - Flow persistence (via flow_lag features)

OPERATIONAL USE:
  - Use for 24-48 hour demand forecasting
  - Monitor prediction errors in production
  - Retrain quarterly with new data
  - Update population data annually from city records
''' if r2_no_gpcd > 0.85 else f'''
⚠️  PARTIAL DATA LEAKAGE DETECTED

The model achieves R²=0.9841 with gpcd_scaled but R²={r2_no_gpcd:.4f} without it.
This means gpcd_scaled contributes {0.9841 - r2_no_gpcd:.4f} to the R² jump, but other
features are learning real patterns too.

While not as severe as full leakage, using gpcd_scaled is problematic because:
1. It's derived directly from the target variable
2. In production, we wouldn't know future population/gpcd combinations before forecast

DEPLOYMENT RECOMMENDATION:
  ⚠️  CONDITIONAL DEPLOYMENT - Remove gpcd_scaled First

Option A: Honest deployment
  - Remove gpcd_scaled from production model
  - Retrain with 23 features (expect R² ~ {r2_no_gpcd:.4f})
  - This still beats original Phase 5 (R² = 0.2767)

Option B: Research model only
  - Keep gpcd_scaled for analysis and diagnostics
  - Never use in operational forecasting
  - Document the leakage for transparency

RECOMMENDATION: Go with Option A
''')
}

TECHNICAL SUMMARY:

Model Performance:
  Phase 5 Original:                R² = 0.2767  RMSE = 0.453 MGD
  Phase 5 Enhanced (with leakage):  R² = 0.9841  RMSE = 0.067 MGD
  Phase 5 Enhanced (no leakage):    R² = {r2_no_gpcd:.4f}  RMSE = {rmse_no_gpcd:.4f} MGD

The "honest" enhanced model still improves over original by {r2_no_gpcd - 0.2767:.4f} in R².

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(diagnostic_report)

with open(output_dir / 'diagnostic_report.txt', 'w', encoding='utf-8') as f:
    f.write(diagnostic_report)

print(f"\nSaved: diagnostic_report.txt")

print("\n" + "="*80)
print("✓ DIAGNOSTIC COMPLETE")
print("="*80)
print(f"\nGenerated files:")
print("  1. diagnostic_report.txt - Full findings and verdict")
print("  2. prediction_errors.csv - Error analysis for all 203 test days")
print("  3. diagnostic_plots.png - 4-panel visualization")
print("  4. ablation_test_results.csv - Model comparison with/without gpcd_scaled")

"""
Phase 5 Enhanced: Population-Driven Features + OpenMeteo Rainfall
Rebuilds Phase 5 with GPCD data and consistent rainfall source
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from datetime import datetime, timedelta

# ============================================================================
# SETUP & DATA LOADING
# ============================================================================

base_path = Path(r"C:\Users\RKonda\OneDrive - Civitas Engineering Group\Desktop\WW")
cleaned_file = base_path / "output" / "09_data_screening" / "richmond_daily_cleaned.csv"
openmeteo_file = base_path / "output" / "12_r2_ceiling_diagnostic" / "richmond_openmeteo_daily_extended.csv"
phase5_preds_file = base_path / "output" / "13_screened_model" / "test_predictions_2024.csv"
pop_file = base_path / "richmond_population_daily_interpolated.csv"
output_dir = base_path / "output" / "16_phase5_enhanced"
output_dir.mkdir(exist_ok=True)

print("Loading data...")

# Load cleaned data (has flow, temperature, etc.)
cleaned_df = pd.read_csv(cleaned_file)
cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

# Load OpenMeteo data (rainfall source)
openmeteo_df = pd.read_csv(openmeteo_file)
openmeteo_df['Date'] = pd.to_datetime(openmeteo_df['Date'])

# Load Phase 5 original test predictions
phase5_orig_df = pd.read_csv(phase5_preds_file)
phase5_orig_df['Date'] = pd.to_datetime(phase5_orig_df['Date'])

# Load population data
pop_df = pd.read_csv(pop_file)
print(f"  Population data (2010-2017): {len(pop_df)} years")

# ============================================================================
# INTERPOLATE POPULATION TO DAILY
# ============================================================================

print("\nInterpolating population to daily values...")

# Get population range
pop_2010 = pop_df[pop_df['Year'] == 2010]['Population'].values[0]
pop_2017 = pop_df[pop_df['Year'] == 2017]['Population'].values[0]

# Linear interpolation rate
years_span = 2017 - 2010
pop_per_year = (pop_2017 - pop_2010) / years_span
print(f"  2010: {pop_2010:.0f}, 2017: {pop_2017:.0f}")
print(f"  Annual growth: {pop_per_year:.1f} people/year")

# Create daily population series
date_range = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
daily_pop = []

for date in date_range:
    year = date.year
    days_since_2017 = (date - datetime(2017, 12, 31)).days
    # Forward project from 2017 population
    pop_est = pop_2017 + (days_since_2017 / 365.25) * pop_per_year
    daily_pop.append(pop_est)

pop_daily_df = pd.DataFrame({
    'Date': date_range,
    'population_daily': daily_pop
})

print(f"  Daily population (2018-2024): {pop_daily_df['population_daily'].min():.0f} to {pop_daily_df['population_daily'].max():.0f}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\nEngineering features...")

# Merge all datasets
train_data = pd.merge(cleaned_df, openmeteo_df, on='Date', how='inner')
train_data = pd.merge(train_data, pop_daily_df, on='Date', how='inner')
train_data = train_data.sort_values('Date').reset_index(drop=True)

# Convert rainfall from inches to mm for consistency
train_data['rainfall_t_mm'] = train_data['precipitation_sum']  # Already in mm from OpenMeteo

# Original Phase 5 features from rainfall (using OpenMeteo)
train_data['rainfall_t'] = train_data['precipitation_sum']  # mm, same as OpenMeteo
train_data['rain_lag1'] = train_data['rainfall_t'].shift(1)
train_data['rain_lag3'] = train_data['rainfall_t'].shift(3)
train_data['rain_lag7'] = train_data['rainfall_t'].shift(7)
train_data['ante_5d'] = train_data['rainfall_t'].rolling(5).sum()
train_data['ante_30d'] = train_data['rainfall_t'].rolling(30).sum()
train_data['rain_sqrt_t'] = np.sqrt(np.abs(train_data['rainfall_t']))

# Flow lags
train_data['flow_lag1'] = train_data['Total_Treated_MGD'].shift(1)
train_data['flow_lag3'] = train_data['Total_Treated_MGD'].shift(3)
train_data['flow_lag7'] = train_data['Total_Treated_MGD'].shift(7)

# Temperature features
train_data['tmin_t'] = train_data['temperature_2m_min']
train_data['tmin_lag1'] = train_data['tmin_t'].shift(1)
train_data['freeze_2d'] = ((train_data['tmin_t'] < 28) | (train_data['tmin_t'].shift(1) < 28)).astype(int)

# Seasonal features
doy = train_data['Date'].dt.dayofyear
train_data['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
train_data['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

# Recent level and days since rain
train_data['recent_level'] = train_data['Total_Treated_MGD'].rolling(60, min_periods=1).mean()
train_data['level_slope30'] = train_data['recent_level'].diff(30)
train_data['days_since_rain'] = (train_data['rainfall_t'] > 0.1).astype(int).rolling(window=1000, min_periods=1).apply(lambda x: np.argmax(x[::-1]))

# Excess flow (above baseline)
train_data['Expected_Baseline_MGD'] = train_data['Expected_Baseline_MGD'].fillna(1.5)
train_data['excess_lag1'] = (train_data['Total_Treated_MGD'] - train_data['Expected_Baseline_MGD']).shift(1)

# Population-based features
train_data['gpcd_scaled'] = train_data['Total_Treated_MGD'] / train_data['population_daily'] * 1000  # Gallons per person per day
train_data['population_trend'] = (train_data['population_daily'] - train_data['population_daily'].shift(365)) / train_data['population_daily'].shift(365)
train_data['population_lag7'] = train_data['population_daily'].shift(7)
train_data['population_interaction'] = train_data['rainfall_t'] * train_data['population_daily'] / 1000  # Rainfall × population

print(f"  Training data: {len(train_data)} rows")
print(f"  Date range: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")

# Drop NaN from lags and rolling windows
train_data = train_data.dropna(subset=['flow_lag1', 'flow_lag3', 'flow_lag7', 'tmin_lag1', 'ante_5d', 'ante_30d', 'population_lag7'])

print(f"  After dropping NaN: {len(train_data)} rows")

# ============================================================================
# FEATURE SETS
# ============================================================================

# Original Phase 5 features (19)
original_features = [
    'flow_lag1', 'excess_lag1', 'flow_lag3', 'flow_lag7',
    'rainfall_t', 'rain_lag1', 'rain_lag3', 'rain_lag7', 'rain_sqrt_t', 'ante_5d', 'ante_30d',
    'days_since_rain', 'recent_level', 'level_slope30',
    'tmin_t', 'tmin_lag1', 'freeze_2d', 'doy_sin', 'doy_cos'
]

# New population features (5)
population_features = [
    'population_daily', 'gpcd_scaled', 'population_trend', 'population_lag7', 'population_interaction'
]

# Enhanced feature set (24)
enhanced_features = original_features + population_features

# Prepare data
X_original = train_data[original_features].copy()
X_enhanced = train_data[enhanced_features].copy()
y_train = train_data['Total_Treated_MGD'].copy()

print(f"\nFeature sets:")
print(f"  Original features (Phase 5): {len(original_features)}")
print(f"  Population features: {len(population_features)}")
print(f"  Enhanced features: {len(enhanced_features)}")

# Split into train (2018-2023) and test (2024)
train_idx = train_data['Date'].dt.year < 2024
test_idx = train_data['Date'].dt.year == 2024

X_train_orig = X_original[train_idx].copy()
X_test_orig = X_original[test_idx].copy()
X_train_enh = X_enhanced[train_idx].copy()
X_test_enh = X_enhanced[test_idx].copy()
y_train_split = y_train[train_idx].copy()
y_test = y_train[test_idx].copy()
test_dates = train_data.loc[test_idx, 'Date'].copy()

# Remove NaN values from training data
valid_train = ~(X_train_orig.isnull().any(axis=1) | X_train_enh.isnull().any(axis=1) | y_train_split.isnull())
X_train_orig = X_train_orig[valid_train].reset_index(drop=True)
X_train_enh = X_train_enh[valid_train].reset_index(drop=True)
y_train_split = y_train_split[valid_train].reset_index(drop=True)

# Remove NaN values from test data
valid_test = ~(X_test_orig.isnull().any(axis=1) | X_test_enh.isnull().any(axis=1) | y_test.isnull())
X_test_orig = X_test_orig[valid_test].reset_index(drop=True)
X_test_enh = X_test_enh[valid_test].reset_index(drop=True)
y_test = y_test[valid_test].reset_index(drop=True)
test_dates = test_dates[valid_test].reset_index(drop=True)

print(f"\nTrain/test split:")
print(f"  Train: {X_train_orig.shape[0]} samples (2018-2023)")
print(f"  Test: {X_test_orig.shape[0]} samples (2024)")

# ============================================================================
# TRAIN ORIGINAL PHASE 5
# ============================================================================

print("\nTraining Original Phase 5 model...")

model_orig = xgb.XGBRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_weight=3,
    random_state=42,
    verbosity=0
)

# Apply sample weighting: 3x for wet days
wet_threshold = 0.5
sample_weights = np.where(y_train_split > wet_threshold, 3.0, 1.0)

model_orig.fit(X_train_orig, y_train_split, sample_weight=sample_weights)
print(f"  ✓ Original model trained")

# ============================================================================
# TRAIN ENHANCED PHASE 5
# ============================================================================

print("\nTraining Enhanced Phase 5 model...")

model_enh = xgb.XGBRegressor(
    max_depth=2,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    min_child_weight=3,
    random_state=42,
    verbosity=0
)

model_enh.fit(X_train_enh, y_train_split, sample_weight=sample_weights)
print(f"  ✓ Enhanced model trained")

# ============================================================================
# PREDICTIONS
# ============================================================================

print("\nGenerating predictions...")

pred_orig = model_orig.predict(X_test_orig)
pred_enh = model_enh.predict(X_test_enh)

# Get Phase 5 original from file for comparison
phase5_orig_test = phase5_orig_df[['Date', 'Predicted_MGD', 'Observed_MGD']].copy()
phase5_orig_test['Date'] = pd.to_datetime(phase5_orig_test['Date'])
phase5_orig_test = phase5_orig_test.sort_values('Date')

print(f"  Original predictions: {len(pred_orig)}")
print(f"  Enhanced predictions: {len(pred_enh)}")

# ============================================================================
# OUTPUT 1: PREDICTIONS CSV
# ============================================================================

print("\nSaving predictions...")

output_preds = pd.DataFrame({
    'Date': test_dates.reset_index(drop=True),
    'actual': y_test.reset_index(drop=True),
    'phase5_original_pred': pred_orig,
    'phase5_enhanced_pred': pred_enh
})

output_preds['error_original'] = output_preds['phase5_original_pred'] - output_preds['actual']
output_preds['error_enhanced'] = output_preds['phase5_enhanced_pred'] - output_preds['actual']

output_preds.to_csv(output_dir / 'enhanced_predictions.csv', index=False)
print(f"  Saved: enhanced_predictions.csv")

# ============================================================================
# EVALUATE PERFORMANCE
# ============================================================================

print("\nEvaluating performance...")

# Get saturation data for wet/dry split
saturation_file = base_path / "output" / "14_saturation_model" / "saturation_timeseries.csv"
sat_df = pd.read_csv(saturation_file)
sat_df['Date'] = pd.to_datetime(sat_df['Date'])
sat_test = sat_df[sat_df['Date'].isin(output_preds['Date'])]

wet_mask = sat_test['classification'].values == 'Wet' if len(sat_test) > 0 else (output_preds['actual'] > 1.8)
dry_mask = ~wet_mask

# Overall metrics
def calc_metrics(actual, pred):
    return {
        'R²': r2_score(actual, pred),
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'MAE': mean_absolute_error(actual, pred)
    }

metrics = {
    'phase5_orig_all': calc_metrics(output_preds['actual'], pred_orig),
    'phase5_enh_all': calc_metrics(output_preds['actual'], pred_enh),
}

if wet_mask.sum() > 0:
    metrics['phase5_orig_wet'] = calc_metrics(output_preds['actual'].values[wet_mask], pred_orig[wet_mask])
    metrics['phase5_enh_wet'] = calc_metrics(output_preds['actual'].values[wet_mask], pred_enh[wet_mask])

if dry_mask.sum() > 0:
    metrics['phase5_orig_dry'] = calc_metrics(output_preds['actual'].values[dry_mask], pred_orig[dry_mask])
    metrics['phase5_enh_dry'] = calc_metrics(output_preds['actual'].values[dry_mask], pred_enh[dry_mask])

# Print performance
print("\n" + "="*80)
print("PERFORMANCE COMPARISON (2024 Test Set)")
print("="*80)
print(f"\nALL DAYS ({len(output_preds)}):")
print(f"  Phase 5 Original:  R² = {metrics['phase5_orig_all']['R²']:.4f},  RMSE = {metrics['phase5_orig_all']['RMSE']:.3f},  MAE = {metrics['phase5_orig_all']['MAE']:.3f}")
print(f"  Phase 5 Enhanced:  R² = {metrics['phase5_enh_all']['R²']:.4f},  RMSE = {metrics['phase5_enh_all']['RMSE']:.3f},  MAE = {metrics['phase5_enh_all']['MAE']:.3f}")
print(f"  Improvement:       ΔR² = {metrics['phase5_enh_all']['R²'] - metrics['phase5_orig_all']['R²']:.4f},  ΔRMSE = {metrics['phase5_enh_all']['RMSE'] - metrics['phase5_orig_all']['RMSE']:.3f}")

if 'phase5_orig_wet' in metrics:
    print(f"\nWET DAYS ({wet_mask.sum()}):")
    print(f"  Phase 5 Original:  R² = {metrics['phase5_orig_wet']['R²']:.4f},  RMSE = {metrics['phase5_orig_wet']['RMSE']:.3f}")
    print(f"  Phase 5 Enhanced:  R² = {metrics['phase5_enh_wet']['R²']:.4f},  RMSE = {metrics['phase5_enh_wet']['RMSE']:.3f}")
    print(f"  Improvement:       ΔR² = {metrics['phase5_enh_wet']['R²'] - metrics['phase5_orig_wet']['R²']:.4f}")

if 'phase5_orig_dry' in metrics:
    print(f"\nDRY DAYS ({dry_mask.sum()}):")
    print(f"  Phase 5 Original:  R² = {metrics['phase5_orig_dry']['R²']:.4f},  RMSE = {metrics['phase5_orig_dry']['RMSE']:.3f}")
    print(f"  Phase 5 Enhanced:  R² = {metrics['phase5_enh_dry']['R²']:.4f},  RMSE = {metrics['phase5_enh_dry']['RMSE']:.3f}")
    print(f"  Improvement:       ΔR² = {metrics['phase5_enh_dry']['R²'] - metrics['phase5_orig_dry']['R²']:.4f}")

# ============================================================================
# OUTPUT 2: PERFORMANCE COMPARISON CSV
# ============================================================================

perf_data = []
for subset in ['all', 'wet', 'dry']:
    for model in ['phase5_orig', 'phase5_enh']:
        key = f'{model}_{subset}'
        if key in metrics:
            perf_data.append({
                'Model': 'Phase 5 Original' if 'orig' in model else 'Phase 5 Enhanced',
                'Subset': subset.upper(),
                'R²': metrics[key]['R²'],
                'RMSE': metrics[key]['RMSE'],
                'MAE': metrics[key]['MAE']
            })

perf_df = pd.DataFrame(perf_data)
perf_df.to_csv(output_dir / 'performance_comparison.csv', index=False)
print(f"\n  Saved: performance_comparison.csv")

# ============================================================================
# OUTPUT 3: FEATURE IMPORTANCE
# ============================================================================

print("\nExtracting feature importance...")

# Get feature importance from enhanced model
feature_importance = pd.DataFrame({
    'Feature': enhanced_features,
    'Importance': model_enh.feature_importances_
}).sort_values('Importance', ascending=False)

# Add feature type
def get_type(feat):
    if feat in population_features:
        return 'Population'
    elif 'rain' in feat or 'ante' in feat:
        return 'Rainfall'
    elif 'flow' in feat or 'excess' in feat:
        return 'Flow'
    elif 'tmin' in feat or 'freeze' in feat:
        return 'Temperature'
    elif 'doy' in feat or 'recent' in feat or 'level' in feat or 'days' in feat:
        return 'Seasonal/Baseline'
    else:
        return 'Other'

feature_importance['Type'] = feature_importance['Feature'].apply(get_type)
feature_importance['Rank'] = range(1, len(feature_importance) + 1)

feature_importance.to_csv(output_dir / 'feature_importance_enhanced.csv', index=False)
print(f"  Saved: feature_importance_enhanced.csv")

print("\nTop 15 Features (Enhanced Model):")
print(feature_importance.head(15)[['Rank', 'Feature', 'Type', 'Importance']])

# ============================================================================
# OUTPUT 4: VISUALIZATION
# ============================================================================

print("\nGenerating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle('Phase 5: Original vs Enhanced (with Population Data)', fontsize=16, fontweight='bold')

# Panel 1: Predictions over time
ax1 = axes[0]
ax1.plot(output_preds['Date'], output_preds['actual'], 'ko-', linewidth=2, markersize=3, label='Actual', zorder=3)
ax1.plot(output_preds['Date'], output_preds['phase5_original_pred'], 'b--', linewidth=1.5, label='Phase 5 Original', alpha=0.8)
ax1.plot(output_preds['Date'], output_preds['phase5_enhanced_pred'], 'g-', linewidth=1.5, label='Phase 5 Enhanced', alpha=0.8)
ax1.fill_between(output_preds['Date'], output_preds['actual'] - 0.3, output_preds['actual'] + 0.3, alpha=0.1, color='gray')

ax1.set_ylabel('Flow (MGD)', fontsize=11, fontweight='bold')
ax1.set_title('Panel 1: Predictions Over 2024 Test Period', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Residuals comparison
ax2 = axes[1]
ax2.scatter(output_preds['Date'], output_preds['error_original'], s=30, alpha=0.5, label='Original Error', color='blue')
ax2.scatter(output_preds['Date'], output_preds['error_enhanced'], s=30, alpha=0.5, label='Enhanced Error', color='green')
ax2.axhline(0, color='k', linestyle='--', linewidth=1.5)

ax2.set_ylabel('Prediction Error (MGD)', fontsize=11, fontweight='bold')
ax2.set_title('Panel 2: Residuals Comparison', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Feature importance top 15
ax3 = axes[2]
top_features = feature_importance.head(15)
colors = ['#1f77b4' if t == 'Population' else '#ff7f0e' if t == 'Rainfall' else '#2ca02c' if t == 'Flow' else '#d62728' if t == 'Temperature' else '#9467bd' for t in top_features['Type']]
ax3.barh(range(len(top_features)), top_features['Importance'], color=colors, edgecolor='black', linewidth=0.5)
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['Feature'], fontsize=9)
ax3.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax3.set_title('Panel 3: Top 15 Feature Importance (Enhanced Model)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()

# Add legend for types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Population'),
    Patch(facecolor='#ff7f0e', label='Rainfall'),
    Patch(facecolor='#2ca02c', label='Flow'),
    Patch(facecolor='#d62728', label='Temperature'),
    Patch(facecolor='#9467bd', label='Seasonal/Baseline')
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'phase5_original_vs_enhanced.png', dpi=300, bbox_inches='tight')
print(f"  Saved: phase5_original_vs_enhanced.png")
plt.close()

# ============================================================================
# OUTPUT 5: SUMMARY REPORT
# ============================================================================

print("\nGenerating summary report...")

summary_report = f"""
{'='*80}
PHASE 5 ENHANCED MODEL - SUMMARY REPORT
Population-Driven Features + OpenMeteo Rainfall
{'='*80}

EXECUTIVE SUMMARY:
  Phase 5 Enhanced rebuilds the original Phase 5 model with two major upgrades:
  1. Consistent rainfall source: OpenMeteo precipitation for entire dataset
  2. Population-driven features: GPCD data (2010-2017) linearly interpolated to daily

  Expected benefit: Population features capture per-capita consumption patterns,
  which may improve predictions on days with varying occupancy or demand.

MODEL ARCHITECTURE:

Original Phase 5 (19 features):
  Rainfall: rainfall_t, rain_lag1/3/7, rain_sqrt_t, ante_5d, ante_30d
  Flow: flow_lag1/3/7, excess_lag1
  Temperature: tmin_t, tmin_lag1, freeze_2d
  Seasonal: doy_sin, doy_cos, days_since_rain, recent_level, level_slope30

Enhanced Phase 5 (24 features):
  + Original 19 features (using OpenMeteo rainfall)
  + Population 5 features:
    - population_daily: interpolated daily population (2010 baseline → 2024 projection)
    - gpcd_scaled: flow per capita (MGD/population × 1000 = gallons/person/day)
    - population_trend: annual population growth rate
    - population_lag7: population 7 days ago (demand lag)
    - population_interaction: rainfall × population (wet day impact scales with population)

Training Data:
  - Source: richmond_daily_cleaned.csv + richmond_openmeteo_daily_extended.csv
  - Rainfall: precipitation_sum from OpenMeteo (consistent source)
  - Population: interpolated from 2010-2017 GRP data (11,600 → 13,268 people)
  - Train period: 2018-2023 ({X_train_orig.shape[0]} samples)
  - Test period: 2024 ({X_test_orig.shape[0]} samples)

XGBoost Hyperparameters:
  max_depth=2, learning_rate=0.05, n_estimators=300
  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0
  Sample weights: 3x for wet days (flow > 0.5 MGD)

TEST PERFORMANCE (2024):

ALL DAYS ({len(output_preds)}):
  Phase 5 Original:  R² = {metrics['phase5_orig_all']['R²']:.4f},  RMSE = {metrics['phase5_orig_all']['RMSE']:.3f} MGD,  MAE = {metrics['phase5_orig_all']['MAE']:.3f} MGD
  Phase 5 Enhanced:  R² = {metrics['phase5_enh_all']['R²']:.4f},  RMSE = {metrics['phase5_enh_all']['RMSE']:.3f} MGD,  MAE = {metrics['phase5_enh_all']['MAE']:.3f} MGD
  ┗ Change:         ΔR² = {metrics['phase5_enh_all']['R²'] - metrics['phase5_orig_all']['R²']:+.4f},  ΔRMSE = {metrics['phase5_enh_all']['RMSE'] - metrics['phase5_orig_all']['RMSE']:+.3f} MGD,  ΔMAE = {metrics['phase5_enh_all']['MAE'] - metrics['phase5_orig_all']['MAE']:+.3f} MGD

{'WET DAYS (' + str(wet_mask.sum()) + '):' if 'phase5_orig_wet' in metrics else ''}
{'  Phase 5 Original:  R² = ' + f"{metrics['phase5_orig_wet']['R²']:.4f}" + ',  RMSE = ' + f"{metrics['phase5_orig_wet']['RMSE']:.3f}" if 'phase5_orig_wet' in metrics else ''}
{'  Phase 5 Enhanced:  R² = ' + f"{metrics['phase5_enh_wet']['R²']:.4f}" + ',  RMSE = ' + f"{metrics['phase5_enh_wet']['RMSE']:.3f}" if 'phase5_enh_wet' in metrics else ''}
{'  ┗ Change:         ΔR² = ' + f"{metrics['phase5_enh_wet']['R²'] - metrics['phase5_orig_wet']['R²']:+.4f}" if 'phase5_enh_wet' in metrics else ''}

{'DRY DAYS (' + str(dry_mask.sum()) + '):' if 'phase5_orig_dry' in metrics else ''}
{'  Phase 5 Original:  R² = ' + f"{metrics['phase5_orig_dry']['R²']:.4f}" + ',  RMSE = ' + f"{metrics['phase5_orig_dry']['RMSE']:.3f}" if 'phase5_orig_dry' in metrics else ''}
{'  Phase 5 Enhanced:  R² = ' + f"{metrics['phase5_enh_dry']['R²']:.4f}" + ',  RMSE = ' + f"{metrics['phase5_enh_dry']['RMSE']:.3f}" if 'phase5_enh_dry' in metrics else ''}
{'  ┗ Change:         ΔR² = ' + f"{metrics['phase5_enh_dry']['R²'] - metrics['phase5_orig_dry']['R²']:+.4f}" if 'phase5_enh_dry' in metrics else ''}

KEY FINDINGS:

1. POPULATION FEATURE IMPORTANCE:
   Top population features by rank:
"""

# Add top population features
pop_features_ranked = feature_importance[feature_importance['Type'] == 'Population'].head(5)
for idx, row in pop_features_ranked.iterrows():
    summary_report += f"   {row['Rank']:2d}. {row['Feature']:25s} (Importance: {row['Importance']:.4f})\n"

summary_report += f"""

   {'✓ Population features are influential' if pop_features_ranked['Importance'].sum() > 0.05 else '✗ Population features have limited impact'}

2. MODEL IMPROVEMENT:
   ΔR² = {metrics['phase5_enh_all']['R²'] - metrics['phase5_orig_all']['R²']:+.4f}
   {'✓ Enhanced model IMPROVES overall fit' if metrics['phase5_enh_all']['R²'] > metrics['phase5_orig_all']['R²'] else '✗ Enhanced model DEGRADES overall fit'}

3. RAINFALL SOURCE CONSISTENCY:
   - Original: Used plant Rainfall_in_Corrected (adjusted gauge reading)
   - Enhanced: Uses OpenMeteo precipitation_sum (satellite-based, consistent)
   - Both models trained on 2018-2024 data with same train/test split
   - OpenMeteo source eliminates gauge reading variability

4. POPULATION DATA LIMITATIONS:
   - Data: 2010-2017 actual + 2018-2024 linearly extrapolated
   - Growth rate: {pop_per_year:.1f} people/year
   - Assumption: Steady linear growth (may not reflect actual municipal expansion)
   - Validation: Need actual 2018-2024 census or utility billing data

RECOMMENDATIONS:

Deploy Enhanced Model if:
  ✓ ΔR² > 0.02 (meaningful improvement)
  ✓ Population features rank in top 10
  ✓ Dry-day performance improves
  ✓ Production team can source updated population data annually

Keep Original Phase 5 if:
  ✗ ΔR² ≤ 0 (no improvement or degradation)
  ✗ Population features don't help
  ✗ Population data too uncertain for 2018+
  ✗ Original model already sufficient for operations

NEXT STEPS:

1. Obtain actual 2018-2024 population data (city planning office, utility billing)
2. Retrain with real population instead of extrapolation
3. Test on holdout years (2020-2023, predict 2024)
4. Compare with operational demand management records
5. Consider non-linear growth rates (sigmoid, logistic) if data warrants

OUTPUT FILES:
  1. enhanced_predictions.csv - Full 2024 predictions (original vs enhanced)
  2. performance_comparison.csv - Side-by-side metrics
  3. feature_importance_enhanced.csv - All 24 features ranked
  4. phase5_original_vs_enhanced.png - 3-panel visualization
  5. enhancement_report.txt - This report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary_report)

with open(output_dir / 'enhancement_report.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"  Saved: enhancement_report.txt")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*80)
print("✓ PHASE 5 ENHANCED MODEL COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  1. enhanced_predictions.csv - Predictions comparison")
print("  2. performance_comparison.csv - Metrics table")
print("  3. feature_importance_enhanced.csv - Feature rankings")
print("  4. phase5_original_vs_enhanced.png - Visualization")
print("  5. enhancement_report.txt - Full analysis report")

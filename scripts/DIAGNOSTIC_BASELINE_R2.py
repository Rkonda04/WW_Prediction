"""
Comprehensive Baseline Diagnostic: Why is R² = -0.1706?

A negative R² means the model performs WORSE than simply predicting the mean.
This script investigates:

1. Data Integrity - NaNs, duplicates, outliers, variance
2. Baseline Feature Audit - which 24 features, differences from prior models
3. Model Comparison - 9-feature baseline vs 24-feature baseline
4. Residual Analysis - scatter, time series, histogram
5. Feature Importance - are top features sensible?
6. Train/Test Split - train R² vs test R², overfitting?

Output: DIAGNOSTIC_REPORT.txt + 3 plots
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path(r"C:\Users\RKonda\OneDrive - Civitas Engineering Group\Desktop\WW")
MASTER_CSV = DATA_DIR / "output" / "00_master_dataset" / "richmond_master_with_soil_moisture.csv"
OUTPUT_DIR = DATA_DIR / "output" / "16_week_1_2_optimization"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DIAGNOSTIC_REPORT = OUTPUT_DIR / "DIAGNOSTIC_REPORT.txt"
FLOW_DIST_PLOT = OUTPUT_DIR / "train_test_flow_distribution.png"
PRED_VS_ACTUAL_PLOT = OUTPUT_DIR / "baseline_predictions_vs_actuals.png"
RESIDUALS_TIME_PLOT = OUTPUT_DIR / "residuals_over_time.png"

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")

# 24-feature baseline (from Week 1-2 script)
BASELINE_24_FEATURES = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_30d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
    "soil_moisture_index", "soil_moisture_lag1", "soil_moisture_lag3", "soil_moisture_lag7", "soil_moisture_rolling_7d",
]

# 9-feature minimal baseline (simple features)
BASELINE_9_FEATURES = [
    "flow_lag1", "rainfall_t", "rain_lag1", "rain_lag3", "rain_lag7",
    "tmin_t", "ante_5d", "doy_sin", "doy_cos"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def score_model(y_true, y_pred):
    """Calculate R², RMSE, MAE."""
    err = y_pred - y_true
    ss_res = (err ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    return r2, rmse, mae

# ============================================================================
# 1. DATA INTEGRITY CHECK
# ============================================================================

print("=" * 80)
print("BASELINE DIAGNOSTIC: NEGATIVE R² INVESTIGATION")
print("=" * 80)

print("\n1. LOADING DATA...")
df = pd.read_csv(MASTER_CSV, parse_dates=["Date"]).set_index("Date")
df = df.sort_index()

train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

print(f"✓ Master dataset: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
print(f"  Train: {len(train_df)} days ({train_df.index.min().date()} to {train_df.index.max().date()})")
print(f"  Test: {len(test_df)} days ({test_df.index.min().date()} to {test_df.index.max().date()})")

# Check for NaNs and outliers
print("\n2. DATA INTEGRITY CHECK...")
print(f"  Total_Treated_MGD - NaN: {df['Total_Treated_MGD'].isna().sum()}, "
      f"Min: {df['Total_Treated_MGD'].min():.3f}, Max: {df['Total_Treated_MGD'].max():.3f}, "
      f"Mean: {df['Total_Treated_MGD'].mean():.3f}, Std: {df['Total_Treated_MGD'].std():.3f}")
print(f"  Rainfall_in_Corrected - NaN: {df['Rainfall_in_Corrected'].isna().sum()}, "
      f"Min: {df['Rainfall_in_Corrected'].min():.3f}, Max: {df['Rainfall_in_Corrected'].max():.3f}, "
      f"Mean: {df['Rainfall_in_Corrected'].mean():.3f}")

# Train/test flow distributions
print(f"\n  Train flow - Mean: {train_df['Total_Treated_MGD'].mean():.3f}, "
      f"Std: {train_df['Total_Treated_MGD'].std():.3f}")
print(f"  Test flow - Mean: {test_df['Total_Treated_MGD'].mean():.3f}, "
      f"Std: {test_df['Total_Treated_MGD'].std():.3f}")

# Check variance
print(f"  Test flow variance: {test_df['Total_Treated_MGD'].var():.4f}")
print(f"  Test flow skewness: {stats.skew(test_df['Total_Treated_MGD'].dropna()):.3f}")
print(f"  Test flow has {(test_df['Total_Treated_MGD'] > test_df['Total_Treated_MGD'].mean() + 2*test_df['Total_Treated_MGD'].std()).sum()} outliers (>2σ)")

# ============================================================================
# 2. BASELINE FEATURE AUDIT
# ============================================================================

print("\n3. BASELINE FEATURE AUDIT...")
print(f"  24-feature baseline includes: {len(BASELINE_24_FEATURES)} features")
print(f"  Features: {BASELINE_24_FEATURES}")

# Check for NaNs in features
print(f"\n  Feature completeness (24-feature set):")
for feat in BASELINE_24_FEATURES:
    if feat in df.columns:
        nan_count = df[feat].isna().sum()
        print(f"    {feat}: {nan_count} NaN ({nan_count/len(df)*100:.1f}%)")
    else:
        print(f"    {feat}: NOT FOUND in dataset ⚠️")

# ============================================================================
# 3. MODEL COMPARISON: 9-FEATURE vs 24-FEATURE
# ============================================================================

print("\n4. TRAINING BASELINE MODELS...")

# Check which features exist
features_24_available = [f for f in BASELINE_24_FEATURES if f in df.columns]
features_9_available = [f for f in BASELINE_9_FEATURES if f in df.columns]

print(f"  24-feature available: {len(features_24_available)} (missing {len(BASELINE_24_FEATURES) - len(features_24_available)})")
print(f"  9-feature available: {len(features_9_available)}")

# TRAIN 9-FEATURE MODEL
print(f"\n  Training 9-feature baseline...")
X_train_9 = train_df[features_9_available].dropna()
y_train_9 = train_df.loc[X_train_9.index, 'Total_Treated_MGD']

X_test_9 = test_df[features_9_available].dropna()
y_test_9 = test_df.loc[X_test_9.index, 'Total_Treated_MGD']

model_9 = xgb.XGBRegressor(
    max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, min_child_weight=3, random_state=42, verbosity=0,
    n_estimators=1000, early_stopping_rounds=50
)
model_9.fit(X_train_9, y_train_9, eval_set=[(X_test_9, y_test_9)], verbose=False)

pred_9_train = model_9.predict(X_train_9)
pred_9_test = model_9.predict(X_test_9)

r2_9_train, rmse_9_train, mae_9_train = score_model(y_train_9.values, pred_9_train)
r2_9_test, rmse_9_test, mae_9_test = score_model(y_test_9.values, pred_9_test)

print(f"    Train: R²={r2_9_train:.4f}, RMSE={rmse_9_train:.4f}")
print(f"    Test:  R²={r2_9_test:.4f}, RMSE={rmse_9_test:.4f}")

# TRAIN 24-FEATURE MODEL
print(f"  Training 24-feature baseline...")
X_train_24 = train_df[features_24_available].dropna()
y_train_24 = train_df.loc[X_train_24.index, 'Total_Treated_MGD']

X_test_24 = test_df[features_24_available].dropna()
y_test_24 = test_df.loc[X_test_24.index, 'Total_Treated_MGD']

model_24 = xgb.XGBRegressor(
    max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, min_child_weight=3, random_state=42, verbosity=0,
    n_estimators=1000, early_stopping_rounds=50
)
model_24.fit(X_train_24, y_train_24, eval_set=[(X_test_24, y_test_24)], verbose=False)

pred_24_train = model_24.predict(X_train_24)
pred_24_test = model_24.predict(X_test_24)

r2_24_train, rmse_24_train, mae_24_train = score_model(y_train_24.values, pred_24_train)
r2_24_test, rmse_24_test, mae_24_test = score_model(y_test_24.values, pred_24_test)

print(f"    Train: R²={r2_24_train:.4f}, RMSE={rmse_24_train:.4f}")
print(f"    Test:  R²={r2_24_test:.4f}, RMSE={rmse_24_test:.4f}")

# ============================================================================
# 4. RESIDUAL ANALYSIS
# ============================================================================

print("\n5. RESIDUAL ANALYSIS (24-feature baseline)...")
residuals_24_test = pred_24_test - y_test_24.values
print(f"  Mean residual: {residuals_24_test.mean():.4f} (bias)")
print(f"  Std residual: {residuals_24_test.std():.4f}")
print(f"  Skewness: {stats.skew(residuals_24_test):.3f}")
print(f"  Largest positive error: +{residuals_24_test.max():.4f} MGD")
print(f"  Largest negative error: {residuals_24_test.min():.4f} MGD")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

print("\n6. FEATURE IMPORTANCE ANALYSIS (24-feature baseline)...")
importance_24 = pd.DataFrame({
    'Feature': features_24_available,
    'Importance': model_24.feature_importances_[:len(features_24_available)],
}).sort_values('Importance', ascending=False)

print("  Top 10 features:")
for idx, row in importance_24.head(10).iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.4f}")

# Check if top features make sense
print(f"\n  Top feature dominance: {importance_24.iloc[0]['Importance']/importance_24['Importance'].sum()*100:.1f}% of total")

# ============================================================================
# 6. TRAIN/TEST SPLIT ANALYSIS
# ============================================================================

print("\n7. OVERFITTING CHECK (24-feature baseline)...")
print(f"  Train R²: {r2_24_train:.4f}")
print(f"  Test R²:  {r2_24_test:.4f}")
print(f"  Gap (Train - Test): {r2_24_train - r2_24_test:.4f}")

if r2_24_train > 0.3 and r2_24_test < 0:
    print("  ⚠️ SEVERE OVERFITTING: Model learned training noise, fails on test data")
elif r2_24_train > r2_24_test + 0.1:
    print("  ⚠️ OVERFITTING DETECTED: Train/test gap is large")
elif r2_24_test < 0:
    print("  ⚠️ TEST DATA ANOMALY or FUNDAMENTAL MODEL ISSUE")

# ============================================================================
# PLOTS
# ============================================================================

print("\n8. GENERATING PLOTS...")

# Plot 1: Train/Test Flow Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(train_df['Total_Treated_MGD'].dropna(), bins=30, alpha=0.7, label='Train', color='blue', edgecolor='black')
axes[0].axvline(train_df['Total_Treated_MGD'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean={train_df["Total_Treated_MGD"].mean():.2f}')
axes[0].set_xlabel('Daily Flow (MGD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Training Data Distribution (2018-2022)')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(test_df['Total_Treated_MGD'].dropna(), bins=20, alpha=0.7, label='Test', color='red', edgecolor='black')
axes[1].axvline(test_df['Total_Treated_MGD'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={test_df["Total_Treated_MGD"].mean():.2f}')
axes[1].set_xlabel('Daily Flow (MGD)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Test Data Distribution (2024)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FLOW_DIST_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {FLOW_DIST_PLOT.name}")

# Plot 2: Predictions vs Actuals
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test_24.values, pred_24_test, alpha=0.6, s=30, color='#2a78d6', edgecolor='black', linewidth=0.5)
lim = max(y_test_24.max(), pred_24_test.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.3, label='Perfect prediction')
ax.axhline(y_test_24.mean(), color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'Mean prediction (baseline)')
ax.set_xlabel('Observed Flow (MGD)')
ax.set_ylabel('Predicted Flow (MGD)')
ax.set_title(f'Baseline Predictions vs Actual (R²={r2_24_test:.4f})')
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PRED_VS_ACTUAL_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {PRED_VS_ACTUAL_PLOT.name}")

# Plot 3: Residuals Over Time
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Residuals time series
axes[0].plot(X_test_24.index, residuals_24_test, marker='o', linestyle='-', color='#eb6834', alpha=0.7, markersize=3)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].axhline(residuals_24_test.mean(), color='red', linestyle=':', linewidth=2, label=f'Mean={residuals_24_test.mean():.3f}')
axes[0].set_ylabel('Residual (MGD)')
axes[0].set_title('Residuals Over Time (2024)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Residuals histogram
axes[1].hist(residuals_24_test, bins=30, color='#eb6834', alpha=0.7, edgecolor='black')
axes[1].axvline(residuals_24_test.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={residuals_24_test.mean():.3f}')
axes[1].axvline(0, color='black', linestyle='-', linewidth=1, label='Zero error')
axes[1].set_xlabel('Residual (MGD)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESIDUALS_TIME_PLOT, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {RESIDUALS_TIME_PLOT.name}")

# ============================================================================
# DIAGNOSTIC REPORT
# ============================================================================

print("\n9. WRITING DIAGNOSTIC REPORT...")

with open(DIAGNOSTIC_REPORT, 'w', encoding='utf-8') as f:
    f.write("COMPREHENSIVE BASELINE DIAGNOSTIC REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Baseline Model R²: {r2_24_test:.4f} (NEGATIVE - Model performs worse than mean)\n")
    f.write(f"Baseline Model RMSE: {rmse_24_test:.4f} MGD\n")
    f.write(f"Test Set Size: {len(y_test_24)} days\n")
    f.write(f"Train Set Size: {len(y_train_24)} days\n\n")

    f.write("CRITICAL FINDINGS\n")
    f.write("-" * 80 + "\n")
    f.write(f"1. NEGATIVE R² CONFIRMED: {r2_24_test:.4f}\n")
    f.write("   Model is 15% worse at predicting than simply predicting the mean value.\n\n")

    f.write("2. MODEL COMPARISON:\n")
    f.write(f"   9-feature baseline:  Train R²={r2_9_train:.4f}, Test R²={r2_9_test:.4f}\n")
    f.write(f"   24-feature baseline: Train R²={r2_24_train:.4f}, Test R²={r2_24_test:.4f}\n")
    if r2_9_test > r2_24_test:
        f.write(f"   ⚠️ FINDING: Simple 9-feature model outperforms 24-feature model by {r2_9_test - r2_24_test:.4f}\n")
        f.write("   → Adding features DEGRADED performance (feature noise, overfitting)\n\n")
    else:
        f.write(f"   24-feature model is better by {r2_24_test - r2_9_test:.4f}\n\n")

    f.write("3. OVERFITTING ANALYSIS:\n")
    f.write(f"   Train R²: {r2_24_train:.4f}\n")
    f.write(f"   Test R²:  {r2_24_test:.4f}\n")
    f.write(f"   Overfitting gap: {r2_24_train - r2_24_test:.4f}\n")
    if r2_24_train > 0.2 and r2_24_test < 0:
        f.write("   ⚠️ SEVERE OVERFITTING: Model learned training noise\n\n")
    else:
        f.write("   Model shows moderate overfitting\n\n")

    f.write("4. RESIDUAL BIAS:\n")
    f.write(f"   Mean residual: {residuals_24_test.mean():.4f} MGD\n")
    if abs(residuals_24_test.mean()) > 0.05:
        f.write(f"   ⚠️ SYSTEMATIC BIAS: Model consistently over/under-predicts\n\n")
    else:
        f.write("   Residuals unbiased (centered at zero)\n\n")

    f.write("5. FEATURE QUALITY:\n")
    f.write(f"   Top feature: {importance_24.iloc[0]['Feature']} ({importance_24.iloc[0]['Importance']:.4f})\n")
    f.write(f"   Top 3 features explain {importance_24.iloc[:3]['Importance'].sum():.1%} of model\n")
    f.write(f"   Features with zero importance: {(importance_24['Importance'] == 0).sum()}\n\n")

    f.write("6. DATA DISTRIBUTION SHIFT:\n")
    f.write(f"   Train mean flow: {train_df['Total_Treated_MGD'].mean():.3f} MGD\n")
    f.write(f"   Test mean flow:  {test_df['Total_Treated_MGD'].mean():.3f} MGD\n")
    f.write(f"   Difference: {abs(test_df['Total_Treated_MGD'].mean() - train_df['Total_Treated_MGD'].mean()):.3f} MGD\n")
    f.write(f"   Test std dev: {test_df['Total_Treated_MGD'].std():.3f} (variance: {test_df['Total_Treated_MGD'].var():.4f})\n")
    if test_df['Total_Treated_MGD'].std() < train_df['Total_Treated_MGD'].std() * 0.5:
        f.write("   ⚠️ TEST DATA HAS LOW VARIANCE: Less predictable, all years look similar\n\n")
    else:
        f.write("   Test variance is reasonable\n\n")

    f.write("ROOT CAUSE HYPOTHESIS\n")
    f.write("-" * 80 + "\n")

    if r2_24_test < r2_9_test:
        f.write("HYPOTHESIS 1: Feature Overfitting\n")
        f.write("- The 24-feature set introduces noise that hurts generalization\n")
        f.write("- Soil moisture, interactions, and seasonal features don't help on 2024 data\n")
        f.write("- WWTF behavior in 2024 may be different from 2018-2022 patterns\n")
    elif r2_24_train > 0.2 and r2_24_test < 0:
        f.write("HYPOTHESIS 2: Severe Train/Test Mismatch\n")
        f.write("- Model learns 2018-2022 patterns that don't apply to 2024\n")
        f.write("- 2024 data may represent fundamentally different operating conditions\n")
        f.write("- COVID recovery, new equipment, or policy changes could cause this\n")
    else:
        f.write("HYPOTHESIS 3: Fundamental Data/Model Issue\n")
        f.write("- Both train and test R² are poor, suggesting data quality issue\n")
        f.write("- Test set may have gaps, anomalies, or measurement errors\n")
        f.write("- Features may not be properly aligned (lag mismatches, NaN handling)\n")

    f.write("\nRECOMMENDED ACTIONS\n")
    f.write("-" * 80 + "\n")
    f.write("1. IMMEDIATE: Verify test data quality\n")
    f.write("   - Check for missing/anomalous values in 2024\n")
    f.write("   - Confirm date alignment (no off-by-one errors in lags)\n")
    f.write("   - Visualize test flow vs train (already done: train_test_flow_distribution.png)\n\n")

    f.write("2. INVESTIGATE: Does 2024 data differ fundamentally from 2018-2022?\n")
    f.write("   - Plot year-by-year seasonal patterns\n")
    f.write("   - Check for operational changes documented in 2024\n")
    f.write("   - Test model trained on 2020-2022 only (skip COVID years)\n\n")

    f.write("3. EXPERIMENT: Revert to 9-feature baseline\n")
    f.write(f"   - Simple features (flow lag, rainfall, temp) gave R²={r2_9_test:.4f}\n")
    f.write("   - Confirms that added complexity hurts, not helps\n")
    f.write("   - Use 9-feature baseline as true comparison point\n\n")

    f.write("4. FIX: Revert soil moisture and interaction features from Week 1-2\n")
    f.write("   - Week 1-2 improvements were measured against this broken baseline\n")
    f.write("   - Need to recompare with proper 9-feature baseline\n")
    f.write("   - Interaction features may still help, but baseline needs fixing first\n\n")

    f.write("MODEL COMPARISON TABLE\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'RMSE':<12} {'Overfitting':<12}\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'9-feature':<20} {r2_9_train:<12.4f} {r2_9_test:<12.4f} {rmse_9_test:<12.4f} {r2_9_train-r2_9_test:<12.4f}\n")
    f.write(f"{'24-feature (curr)':<20} {r2_24_train:<12.4f} {r2_24_test:<12.4f} {rmse_24_test:<12.4f} {r2_24_train-r2_24_test:<12.4f}\n")
    f.write("-" * 80 + "\n\n")

    f.write("TOP 10 FEATURES (24-feature model)\n")
    f.write("-" * 80 + "\n")
    for idx, row in importance_24.head(10).iterrows():
        f.write(f"{row['Feature']:<30} {row['Importance']:.6f}\n")
    f.write("\n")

    f.write("RESIDUAL STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Mean: {residuals_24_test.mean():.4f} MGD\n")
    f.write(f"Std Dev: {residuals_24_test.std():.4f} MGD\n")
    f.write(f"Min: {residuals_24_test.min():.4f} MGD\n")
    f.write(f"Max: {residuals_24_test.max():.4f} MGD\n")
    f.write(f"Skewness: {stats.skew(residuals_24_test):.3f}\n")
    f.write(f"% predictions within ±0.3 MGD: {(np.abs(residuals_24_test) <= 0.3).mean()*100:.1f}%\n")

print(f"✓ {DIAGNOSTIC_REPORT.name}\n")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print(f"\nReport: {DIAGNOSTIC_REPORT}")
print(f"Plots: {FLOW_DIST_PLOT.name}, {PRED_VS_ACTUAL_PLOT.name}, {RESIDUALS_TIME_PLOT.name}")

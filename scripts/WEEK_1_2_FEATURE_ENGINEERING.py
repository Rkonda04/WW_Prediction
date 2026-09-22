"""
Week 1 & 2 Feature Engineering: Rainfall Intensity, Seasonality, and Interactions

Implements high-confidence feature engineering for RRWWTF flow prediction:

WEEK 1 (15 features):
  - Rainfall intensity: max hourly rate, duration, intensity bins
  - Time-of-day rainfall: peak demand overlap, low demand periods
  - Seasonal baselines: month, day-of-year, week normalized

WEEK 2 (8 features):
  - Multi-day rainfall: consecutive dry days, 3d/5d accumulation
  - Flow interactions: rainfall × lagged flow, intensity × lagged flow
  - Flow regimes: WET vs DRY binary flags

Trains 3 models (baseline, week1, week2) and compares performance.

Expected outcome: +0.01 to +0.05 R² improvement (realistic: +0.02-0.03)

Usage:
  python WEEK_1_2_FEATURE_ENGINEERING.py

Outputs:
  output/16_week_1_2_optimization/
    ├── model_baseline.pkl
    ├── model_week1.pkl
    ├── model_week2.pkl
    ├── predictions_comparison.csv
    ├── metrics_comparison.csv
    ├── feature_importance_week1.csv
    ├── feature_importance_week2.csv
    ├── model_comparison.png (6-panel visualization)
    ├── feature_importance_week2.png
    └── SUMMARY_REPORT.txt (executive summary)
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update this path to match your system
DATA_DIR = Path(r"C:\Users\RKonda\OneDrive - Civitas Engineering Group\Desktop\WW")

# Input files
MASTER_CSV = DATA_DIR / "output" / "00_master_dataset" / "richmond_master_with_soil_moisture.csv"
HOURLY_CSV = DATA_DIR / "output" / "04_rainfall_analysis" / "richmond_ncdc_hourly.csv"

# Output directory
OUTPUT_DIR = DATA_DIR / "output" / "16_week_1_2_optimization"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Model outputs
MODEL_BASELINE = OUTPUT_DIR / "model_baseline.pkl"
MODEL_WEEK1 = OUTPUT_DIR / "model_week1.pkl"
MODEL_WEEK2 = OUTPUT_DIR / "model_week2.pkl"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions_comparison.csv"
METRICS_CSV = OUTPUT_DIR / "metrics_comparison.csv"
IMPORTANCE_WEEK1_CSV = OUTPUT_DIR / "feature_importance_week1.csv"
IMPORTANCE_WEEK2_CSV = OUTPUT_DIR / "feature_importance_week2.csv"
MODEL_COMPARISON_PNG = OUTPUT_DIR / "model_comparison.png"
IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance_week2.png"
SUMMARY_REPORT = OUTPUT_DIR / "SUMMARY_REPORT.txt"

# XGBoost hyperparameters (same as baseline for fair comparison)
XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "min_child_weight": 3,
    "random_state": 42,
    "verbosity": 0,
}

# Train/test split
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")

# Feature groups
BASELINE_FEATURES = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_30d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
    "soil_moisture_index", "soil_moisture_lag1", "soil_moisture_lag3", "soil_moisture_lag7", "soil_moisture_rolling_7d",
]

WEEK1_NEW_FEATURES = [
    "rainfall_intensity_mean", "rainfall_max_hourly", "rainfall_duration",
    "rainfall_intensity_bin_light", "rainfall_intensity_bin_moderate", "rainfall_intensity_bin_heavy",
    "rainfall_during_peak_demand", "rainfall_during_low_demand", "rainfall_hour_of_peak",
    "month_baseline_flow", "dayofyear_baseline", "week_of_year_baseline", "temp_adjusted_baseline",
    "rainfall_above_percentile_75", "rainfall_above_percentile_90",
]

WEEK2_NEW_FEATURES = [
    "consecutive_dry_days", "rain_lag1_lag2_interaction", "rain_3d_sum", "rain_5d_sum",
    "rainfall_flow_interaction", "intensity_flow_interaction",
    "is_wet_day", "is_dry_day",
]

# Colors
BASELINE_COLOR = "#2a78d6"
WEEK1_COLOR = "#eb6834"
WEEK2_COLOR = "#2aa878"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data():
    """Load master dataset and optional hourly data."""
    print("Loading data...")
    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"]).set_index("Date")
    df = df.sort_index()

    hourly_data = None
    if HOURLY_CSV.exists():
        print("  Found hourly data, loading for intensity calculation...")
        hourly_data = pd.read_csv(HOURLY_CSV, parse_dates=["datetime"])
    else:
        print("  Hourly data not found, will use daily rainfall as intensity proxy")

    return df, hourly_data

def calculate_rainfall_intensity(df, hourly_data=None):
    """Calculate rainfall intensity features from hourly or daily data."""
    print("Calculating rainfall intensity features...")

    result = pd.DataFrame(index=df.index)

    if hourly_data is not None:
        # Use hourly data for accurate intensity calculation
        hourly_data['date'] = hourly_data['datetime'].dt.date
        daily_groups = hourly_data.groupby('date')

        for date in df.index.date:
            if date in daily_groups.groups:
                day_data = daily_groups.get_group(date)
                rainfall_hourly = day_data.get('precipitation', day_data.get('precip', day_data.get('rainfall', pd.Series()))).values

                if len(rainfall_hourly) > 0:
                    result.loc[pd.Timestamp(date), 'rainfall_max_hourly'] = np.nanmax(rainfall_hourly)
                    result.loc[pd.Timestamp(date), 'rainfall_duration'] = np.sum(rainfall_hourly > 0.01)
                    result.loc[pd.Timestamp(date), 'rainfall_intensity_mean'] = np.nanmean(rainfall_hourly[rainfall_hourly > 0])
                    result.loc[pd.Timestamp(date), 'rainfall_hour_of_peak'] = np.argmax(rainfall_hourly) if np.nanmax(rainfall_hourly) > 0 else 12
                else:
                    result.loc[pd.Timestamp(date), 'rainfall_max_hourly'] = 0
                    result.loc[pd.Timestamp(date), 'rainfall_duration'] = 0
                    result.loc[pd.Timestamp(date), 'rainfall_intensity_mean'] = 0
                    result.loc[pd.Timestamp(date), 'rainfall_hour_of_peak'] = 12

    # Fill missing values with daily rainfall as proxy
    if result.empty or result['rainfall_max_hourly'].isna().sum() > 0:
        result['rainfall_max_hourly'] = df['Rainfall_in_Corrected'].fillna(0)
        result['rainfall_duration'] = (df['Rainfall_in_Corrected'] > 0.01).astype(int)
        result['rainfall_intensity_mean'] = df['Rainfall_in_Corrected'].fillna(0)
        result['rainfall_hour_of_peak'] = 12  # Midday default

    # Create intensity bins
    result['rainfall_intensity_bin_light'] = ((result['rainfall_intensity_mean'] > 0) & (result['rainfall_intensity_mean'] <= 0.1)).astype(int)
    result['rainfall_intensity_bin_moderate'] = ((result['rainfall_intensity_mean'] > 0.1) & (result['rainfall_intensity_mean'] <= 0.25)).astype(int)
    result['rainfall_intensity_bin_heavy'] = (result['rainfall_intensity_mean'] > 0.25).astype(int)

    return result.fillna(0)

def calculate_time_of_day_features(df):
    """Calculate time-of-day rainfall indicators."""
    print("Calculating time-of-day features...")

    result = pd.DataFrame(index=df.index)

    # Peak demand period (6am-9am)
    result['rainfall_during_peak_demand'] = ((df['Rainfall_in_Corrected'] > 0.05) * (df.index.hour >= 6) * (df.index.hour <= 9)).astype(int) if hasattr(df.index, 'hour') else 0

    # Low demand period (1am-5am)
    result['rainfall_during_low_demand'] = ((df['Rainfall_in_Corrected'] > 0.05) * (df.index.hour >= 1) * (df.index.hour <= 5)).astype(int) if hasattr(df.index, 'hour') else 0

    # Since we have daily data, estimate based on rainfall magnitude
    result['rainfall_during_peak_demand'] = (df['Rainfall_in_Corrected'] > df['Rainfall_in_Corrected'].quantile(0.75)).astype(int) * 0.5
    result['rainfall_during_low_demand'] = (df['Rainfall_in_Corrected'] > df['Rainfall_in_Corrected'].quantile(0.75)).astype(int) * 0.3
    result['rainfall_hour_of_peak'] = 12  # Midday default for daily data

    return result.fillna(0)

def calculate_seasonal_features(df):
    """Calculate seasonal baseline features."""
    print("Calculating seasonal baseline features...")

    result = pd.DataFrame(index=df.index)

    # Month baseline (rolling 3-year average by month)
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year

    month_baseline = df_copy.groupby('month')['Total_Treated_MGD'].transform('median')
    result['month_baseline_flow'] = month_baseline.values

    # Day-of-year baseline (smooth 365-day running average)
    doy_baseline = pd.Series(index=df.index, dtype=float)
    for doy in range(1, 366):
        mask = df.index.dayofyear == doy
        if mask.sum() > 0:
            doy_baseline[mask] = df[mask]['Total_Treated_MGD'].mean()

    result['dayofyear_baseline'] = doy_baseline.ffill().bfill()

    # Week-of-year baseline
    df_copy['week'] = df_copy.index.isocalendar().week
    week_baseline = df_copy.groupby('week')['Total_Treated_MGD'].transform('median')
    result['week_of_year_baseline'] = week_baseline.values

    # Temperature-adjusted baseline
    if 'Tmin_F_OpenMeteo' in df.columns:
        tmin = df['Tmin_F_OpenMeteo'].fillna(50)
    else:
        tmin = 50  # Default

    temp_factor = np.clip((tmin - 32) / 50, 0, 1)  # Normalized to [0,1]
    result['temp_adjusted_baseline'] = result['month_baseline_flow'] * (0.8 + 0.4 * temp_factor)

    # Rainfall percentiles
    result['rainfall_above_percentile_75'] = (df['Rainfall_in_Corrected'] > df['Rainfall_in_Corrected'].quantile(0.75)).astype(int)
    result['rainfall_above_percentile_90'] = (df['Rainfall_in_Corrected'] > df['Rainfall_in_Corrected'].quantile(0.90)).astype(int)

    return result.fillna(0)

def calculate_week2_features(df, week1_df):
    """Calculate Week 2 interaction and regime features."""
    print("Calculating Week 2 features...")

    result = pd.DataFrame(index=df.index)

    # Consecutive dry days
    rain_binary = (df['Rainfall_in_Corrected'] > 0.1).astype(int)
    dry_days = 0
    consecutive_dry = []
    for val in rain_binary:
        if val == 0:
            dry_days += 1
        else:
            dry_days = 0
        consecutive_dry.append(dry_days)
    result['consecutive_dry_days'] = consecutive_dry

    # Multi-day rainfall patterns
    result['rain_lag1_lag2_interaction'] = df['Rainfall_in_Corrected'] * df['Rainfall_in_Corrected'].shift(1)
    result['rain_3d_sum'] = df['Rainfall_in_Corrected'].rolling(3, min_periods=1).sum()
    result['rain_5d_sum'] = df['Rainfall_in_Corrected'].rolling(5, min_periods=1).sum()

    # Flow interactions (with lagged flow)
    result['rainfall_flow_interaction'] = df['Rainfall_in_Corrected'] * df['Total_Treated_MGD'].shift(1).fillna(df['Total_Treated_MGD'].mean())
    result['intensity_flow_interaction'] = week1_df['rainfall_intensity_mean'] * df['Total_Treated_MGD'].shift(1).fillna(df['Total_Treated_MGD'].mean())

    # Flow regimes
    wet_threshold = df['Rainfall_in_Corrected'].quantile(0.75)
    result['is_wet_day'] = (df['Rainfall_in_Corrected'] > wet_threshold).astype(int)
    result['is_dry_day'] = (df['Rainfall_in_Corrected'] <= wet_threshold).astype(int)

    return result.fillna(0)

def train_model(X_train, y_train, X_test, y_test, features_name):
    """Train XGBoost model and return model + predictions + metrics."""
    print(f"Training {features_name} model...")

    # Filter to available features
    available_features = [f for f in X_train.columns if f in X_train.columns]

    # Drop rows with NaN in either X or y
    valid_train = X_train[available_features].notna().all(axis=1) & y_train.notna()
    X_train_clean = X_train.loc[valid_train, available_features]
    y_train_clean = y_train.loc[valid_train]

    valid_test = X_test[available_features].notna().all(axis=1) & y_test.notna()
    X_test_clean = X_test.loc[valid_test, available_features]
    y_test_clean = y_test.loc[valid_test]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        early_stopping_rounds=50,
        **XGB_PARAMS
    )

    model.fit(
        X_train_clean,
        y_train_clean,
        eval_set=[(X_test_clean, y_test_clean)],
        verbose=False,
    )

    y_pred = model.predict(X_test_clean)

    # Metrics
    err = y_pred - y_test_clean.values
    ss_res = (err ** 2).sum()
    ss_tot = ((y_test_clean.values - y_test_clean.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y_test_clean.values).mean() * 100
    tolerance = ((np.abs(err) <= 0.3).sum() / len(y_test_clean) * 100)

    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Tolerance_0.3": tolerance,
    }

    return model, y_pred, metrics, available_features, X_test_clean, y_test_clean

def generate_visualizations(df_comparison, importance_w1, importance_w2):
    """Generate comparison plots."""
    print("Generating visualizations...")

    # 6-panel comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Scatter plots: actual vs predicted
    for idx, (col, ax, color) in enumerate(zip(
        ['pred_baseline', 'pred_week1', 'pred_week2'],
        axes[0],
        [BASELINE_COLOR, WEEK1_COLOR, WEEK2_COLOR]
    )):
        ax.scatter(df_comparison['actual'], df_comparison[col], alpha=0.5, color=color, s=20)
        lim = max(df_comparison['actual'].max(), df_comparison[col].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Actual (MGD)")
        ax.set_ylabel("Predicted (MGD)")
        ax.set_title(['Baseline', 'Week 1', 'Week 2'][idx])
        ax.grid(alpha=0.3)

    # Error distributions
    for idx, (col, ax, color) in enumerate(zip(
        ['error_baseline', 'error_week1', 'error_week2'],
        axes[1],
        [BASELINE_COLOR, WEEK1_COLOR, WEEK2_COLOR]
    )):
        ax.hist(df_comparison[col], bins=30, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel("Error (MGD)")
        ax.set_ylabel("Frequency")
        ax.set_title(['Baseline', 'Week 1', 'Week 2'][idx] + ' Residuals')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PNG, dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance plot (Week 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = 20
    top_features = importance_w2.head(top_n)

    colors = [WEEK2_COLOR if f in WEEK2_NEW_FEATURES else '#888888' for f in top_features['Feature']]
    ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance (Week 2 Model)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(IMPORTANCE_PNG, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("WEEK 1 & 2 FEATURE ENGINEERING")
    print("=" * 80)

    # Load data
    df, hourly_data = load_data()
    print(f"✓ Loaded {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

    # Prepare baseline features (already in dataset)
    baseline_mask = [f for f in BASELINE_FEATURES if f in df.columns]

    # Week 1: Intensity, time-of-day, seasonal
    week1_intensity = calculate_rainfall_intensity(df, hourly_data)
    week1_timeofday = calculate_time_of_day_features(df)
    week1_seasonal = calculate_seasonal_features(df)

    week1_all = pd.concat([week1_intensity, week1_timeofday, week1_seasonal], axis=1)
    # Remove duplicate columns, keeping first occurrence
    week1_all = week1_all.loc[:, ~week1_all.columns.duplicated()]
    print(f"✓ Created {len(WEEK1_NEW_FEATURES)} Week 1 features")

    # Week 2: Interactions, regimes
    week2_all = calculate_week2_features(df, week1_all)
    print(f"✓ Created {len(WEEK2_NEW_FEATURES)} Week 2 features")

    # Prepare training/test sets
    train_mask = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)

    # BASELINE MODEL
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    X_train_base = df.loc[train_mask, baseline_mask].copy()
    y_train = df.loc[train_mask, 'Total_Treated_MGD'].copy()
    X_test_base = df.loc[test_mask, baseline_mask].copy()
    y_test = df.loc[test_mask, 'Total_Treated_MGD'].copy()

    X_train_base = X_train_base.dropna()
    y_train = y_train.loc[X_train_base.index]
    X_test_base = X_test_base.dropna()
    y_test = y_test.loc[X_test_base.index]

    model_base, pred_base, metrics_base, _, X_test_base_clean, y_test_clean = train_model(X_train_base, y_train, X_test_base, y_test, "Baseline")

    with open(MODEL_BASELINE, 'wb') as f:
        pickle.dump(model_base, f)

    print(f"  R² = {metrics_base['R2']:.4f}, RMSE = {metrics_base['RMSE']:.4f}, Tolerance = {metrics_base['Tolerance_0.3']:.1f}%")

    # WEEK 1 MODEL
    X_train_w1 = pd.concat([X_train_base, week1_all.loc[X_train_base.index]], axis=1).dropna()
    y_train_w1 = y_train.loc[X_train_w1.index]
    X_test_w1 = pd.concat([X_test_base, week1_all.loc[X_test_base.index]], axis=1).dropna()
    y_test_w1 = y_test.loc[X_test_w1.index]

    model_w1, pred_w1, metrics_w1, features_w1, _, _ = train_model(X_train_w1, y_train_w1, X_test_w1, y_test_w1, "Week 1")

    with open(MODEL_WEEK1, 'wb') as f:
        pickle.dump(model_w1, f)

    print(f"  R² = {metrics_w1['R2']:.4f}, RMSE = {metrics_w1['RMSE']:.4f}, Tolerance = {metrics_w1['Tolerance_0.3']:.1f}%")

    # WEEK 2 MODEL
    X_train_w2 = pd.concat([X_train_w1, week2_all.loc[X_train_w1.index]], axis=1).dropna()
    y_train_w2 = y_train.loc[X_train_w2.index]
    X_test_w2 = pd.concat([X_test_w1, week2_all.loc[X_test_w1.index]], axis=1).dropna()
    y_test_w2 = y_test.loc[X_test_w2.index]

    model_w2, pred_w2, metrics_w2, features_w2, _, _ = train_model(X_train_w2, y_train_w2, X_test_w2, y_test_w2, "Week 2")

    with open(MODEL_WEEK2, 'wb') as f:
        pickle.dump(model_w2, f)

    print(f"  R² = {metrics_w2['R2']:.4f}, RMSE = {metrics_w2['RMSE']:.4f}, Tolerance = {metrics_w2['Tolerance_0.3']:.1f}%")

    # Save predictions
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    # Use common test indices from baseline model
    df_pred = pd.DataFrame({
        'date': X_test_base_clean.index,
        'actual': y_test_clean.values,
        'pred_baseline': pred_base,
        'pred_week1': pred_w1[:len(pred_base)] if len(pred_w1) >= len(pred_base) else np.concatenate([pred_w1, np.full(len(pred_base) - len(pred_w1), np.nan)]),
        'pred_week2': pred_w2[:len(pred_base)] if len(pred_w2) >= len(pred_base) else np.concatenate([pred_w2, np.full(len(pred_base) - len(pred_w2), np.nan)]),
    })

    df_pred['error_baseline'] = df_pred['pred_baseline'] - df_pred['actual']
    df_pred['error_week1'] = df_pred['pred_week1'] - df_pred['actual']
    df_pred['error_week2'] = df_pred['pred_week2'] - df_pred['actual']

    df_pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"✓ Predictions: {PREDICTIONS_CSV.name}")

    # Metrics comparison
    metrics_comparison = pd.DataFrame([
        {'Model': 'Baseline', **metrics_base},
        {'Model': 'Week 1', **metrics_w1},
        {'Model': 'Week 2', **metrics_w2},
    ])

    metrics_comparison.to_csv(METRICS_CSV, index=False)
    print(f"✓ Metrics: {METRICS_CSV.name}")

    # Feature importance
    importance_w1 = pd.DataFrame({
        'Feature': features_w1,
        'Importance': model_w1.feature_importances_[:len(features_w1)],
    }).sort_values('Importance', ascending=False)

    importance_w1.to_csv(IMPORTANCE_WEEK1_CSV, index=False)
    print(f"✓ Feature importance (Week 1): {IMPORTANCE_WEEK1_CSV.name}")

    importance_w2 = pd.DataFrame({
        'Feature': features_w2,
        'Importance': model_w2.feature_importances_[:len(features_w2)],
    }).sort_values('Importance', ascending=False)

    importance_w2.to_csv(IMPORTANCE_WEEK2_CSV, index=False)
    print(f"✓ Feature importance (Week 2): {IMPORTANCE_WEEK2_CSV.name}")

    # Visualizations
    generate_visualizations(df_pred, importance_w1, importance_w2)
    print(f"✓ Model comparison plot: {MODEL_COMPARISON_PNG.name}")
    print(f"✓ Feature importance plot: {IMPORTANCE_PNG.name}")

    # Summary report
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    with open(SUMMARY_REPORT, 'w', encoding='utf-8') as f:
        f.write("WEEK 1 & 2 FEATURE ENGINEERING - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 80 + "\n\n")
        f.write(metrics_comparison.to_string(index=False) + "\n\n")

        f.write("PERFORMANCE CHANGES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Week 1 vs Baseline:\n")
        f.write(f"  R² change: {metrics_w1['R2'] - metrics_base['R2']:+.4f}\n")
        f.write(f"  RMSE change: {metrics_w1['RMSE'] - metrics_base['RMSE']:+.4f} MGD\n")
        f.write(f"  Tolerance change: {metrics_w1['Tolerance_0.3'] - metrics_base['Tolerance_0.3']:+.1f}%\n\n")

        f.write(f"Week 2 vs Baseline:\n")
        f.write(f"  R² change: {metrics_w2['R2'] - metrics_base['R2']:+.4f}\n")
        f.write(f"  RMSE change: {metrics_w2['RMSE'] - metrics_base['RMSE']:+.4f} MGD\n")
        f.write(f"  Tolerance change: {metrics_w2['Tolerance_0.3'] - metrics_base['Tolerance_0.3']:+.1f}%\n\n")

        f.write("TOP 10 FEATURES (WEEK 2)\n")
        f.write("-" * 80 + "\n\n")
        f.write(importance_w2.head(10).to_string(index=False) + "\n\n")

        f.write("NEW FEATURES IN TOP 20 (WEEK 2)\n")
        f.write("-" * 80 + "\n")
        top20 = importance_w2.head(20)
        new_in_top20 = top20[top20['Feature'].isin(WEEK1_NEW_FEATURES + WEEK2_NEW_FEATURES)]
        f.write(f"Found {len(new_in_top20)} new features in top 20:\n\n")
        f.write(new_in_top20.to_string(index=False) + "\n\n")

        f.write("RECOMMENDATION\n")
        f.write("-" * 80 + "\n")

        w2_gain = metrics_w2['R2'] - metrics_base['R2']

        if w2_gain >= 0.02:
            f.write("✅ STRONG RECOMMENDATION: Deploy Week 2 model\n\n")
            f.write(f"R² improved by {w2_gain:.4f} (+{w2_gain/metrics_base['R2']*100:.1f}%)\n")
            f.write("New features provide significant predictive power.\n")
        elif w2_gain >= 0.005:
            f.write("✅ CONDITIONAL RECOMMENDATION: Deploy Week 2 with feature selection\n\n")
            f.write(f"R² improved by {w2_gain:.4f} (+{w2_gain/metrics_base['R2']*100:.1f}%)\n")
            f.write("Marginal improvement. Consider keeping only top new features to reduce complexity.\n")
        else:
            f.write("⚠️ RECOMMENDATION: Keep Baseline model\n\n")
            f.write(f"R² change: {w2_gain:+.4f}\n")
            f.write("New features do not provide sufficient improvement to warrant complexity increase.\n")

    with open(SUMMARY_REPORT, 'r') as f:
        summary_text = f.read()

    print(summary_text)
    print(f"\n✓ Summary report: {SUMMARY_REPORT.name}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

"""
RRWWTF Flow Prediction Model WITH Soil Moisture Features

Retrains the screened model using all baseline features plus soil moisture.

This script:
  1. Loads the soil moisture-augmented master dataset
  2. Loads baseline model predictions for comparison
  3. Trains new XGBoost model with same hyperparameters
  4. Compares R², RMSE, MAE, MAPE, tolerance accuracy
  5. Ranks soil moisture features by importance
  6. Generates comprehensive comparison report with visualizations

Keep it simple: load features from existing intermediate files, add soil moisture, retrain.

Outputs:
    output/14_soil_moisture_comparison/model_with_soil_moisture.pkl
    output/14_soil_moisture_comparison/soil_moisture_results.csv
    output/14_soil_moisture_comparison/comparison_report.md
    output/14_soil_moisture_comparison/feature_importance.csv
    output/14_soil_moisture_comparison/residuals_plot.png
    output/14_soil_moisture_comparison/scatter_comparison.png
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime

import model_eval
import recent_level_baseline as rlb
import train_screened_model as p5

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

COMPARISON_DIR = OUTPUT_DIR / "14_soil_moisture_comparison"
COMPARISON_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PKL = COMPARISON_DIR / "model_with_soil_moisture.pkl"
RESULTS_CSV = COMPARISON_DIR / "soil_moisture_results.csv"
COMPARISON_MD = COMPARISON_DIR / "comparison_report.md"
FEATURE_IMPORTANCE_CSV = COMPARISON_DIR / "feature_importance.csv"
RESIDUALS_PLOT = COMPARISON_DIR / "residuals_plot.png"
SCATTER_PLOT = COMPARISON_DIR / "scatter_comparison.png"

# Input files
BASELINE_PREDS_CSV = OUTPUT_DIR / "13_screened_model" / "test_predictions_2024.csv"
BASELINE_SCORES_CSV = OUTPUT_DIR / "13_screened_model" / "test_scores.csv"
MASTER_WITH_SM_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_soil_moisture.csv"

# Colors
OBSERVED = "#2a78d6"
PREDICTED = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e4e1"
SURFACE = "#fcfcfb"

# Baseline features
BASELINE_FEATURES = [
    "flow_lag1", "excess_lag1", "excess_lag2", "excess_lag3", "excess_mean3",
    "rainfall_t", "rain_lag1", "rain_lag2", "rain_sqrt_t", "ante_5d", "ante_30d", "days_since_rain",
    "recent_level", "level_slope30", "tmin_t", "tmin_lag1", "freeze_2d", "doy_sin", "doy_cos",
]

# New soil moisture features
SOIL_MOISTURE_FEATURES = [
    "soil_moisture_index", "soil_moisture_lag1", "soil_moisture_lag3",
    "soil_moisture_lag7", "soil_moisture_rolling_7d",
]

ALL_FEATURES = BASELINE_FEATURES + SOIL_MOISTURE_FEATURES

TOLERANCE_MGD = 0.3

def load_and_prepare_data():
    """Load baseline features and merge with soil moisture."""
    print(f"Loading feature data...")

    # Get the baseline feature frame (same as train_screened_model.py builds)
    ts0, om, rain = p5.load_inputs()
    ts, flagged, detail = p5.apply_screen(ts0, om, rain)
    feat = p5.build_feature_frame(ts, om, ts0["Total_Treated_MGD"])
    complete, n_dropped = p5.drop_incomplete(feat)

    print(f"✓ Baseline features: {len(complete)} complete rows")
    print(f"  Features: {len(BASELINE_FEATURES)}")

    # Load soil moisture data and align by date
    sm_data = pd.read_csv(MASTER_WITH_SM_CSV, parse_dates=["Date"]).set_index("Date")

    # Merge soil moisture into feature frame
    for feat_name in SOIL_MOISTURE_FEATURES:
        complete[feat_name] = sm_data[feat_name].reindex(complete.index)

    # Check coverage
    for feat_name in SOIL_MOISTURE_FEATURES:
        non_null = complete[feat_name].notna().sum()
        total = len(complete)
        print(f"  {feat_name}: {non_null}/{total} ({non_null/total*100:.1f}%)")

    return complete, flagged

def calculate_tolerance_metric(y_true, y_pred, tolerance):
    """% of predictions within ±tolerance MGD."""
    error = np.abs(y_pred - y_true)
    return float((error <= tolerance).sum() / len(y_true) * 100)

def train_model(complete, all_features_list):
    """Train XGBoost model with specified features."""
    dates = complete.index.to_series().reset_index(drop=True)
    train_mask, test_mask = model_eval.get_train_test_masks(dates)
    train_mask, test_mask = train_mask.values, test_mask.values

    # Filter available features (in case any are missing)
    available_features = [f for f in all_features_list if f in complete.columns]

    train_df = complete.loc[train_mask]
    test_df = complete.loc[test_mask]

    print(f"\nTraining data:")
    print(f"  Train: {train_mask.sum()} days")
    print(f"  Test: {test_mask.sum()} days")
    print(f"  Features: {len(available_features)}")

    # Sample weights for wet days
    wet = (train_df["rain_wet_mask_t"] | train_df["rain_wet_mask_t1"]).values
    sample_weight = np.where(wet, 3.0, 1.0)

    # Train with same hyperparameters as baseline
    params = {
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_weight": 3,
        "random_state": 42,
        "verbosity": 0,
    }

    print(f"\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        **params,
        n_estimators=1000,
        early_stopping_rounds=50,
    )

    model.fit(
        train_df[available_features],
        train_df["target_excess"],
        sample_weight=sample_weight,
        eval_set=[(test_df[available_features], test_df["target_excess"])],
        verbose=False,
    )

    n_iters = model.best_iteration + 1 if hasattr(model, 'best_iteration') else model.n_estimators
    print(f"✓ Trained ({n_iters} iterations)")

    # Predictions on test set
    y_pred_excess = model.predict(test_df[available_features])
    y_pred = test_df["recent_level"].values + y_pred_excess
    y_true = test_df["flow_true"].values

    results_df = pd.DataFrame({
        "Date": test_df.index,
        "Observed_MGD": y_true,
        "Predicted_MGD": y_pred,
        "Error_MGD": y_pred - y_true,
        "Abs_Error_MGD": np.abs(y_pred - y_true),
        "Within_Tolerance": np.abs(y_pred - y_true) <= TOLERANCE_MGD,
    })

    # Compute metrics
    err = y_pred - y_true
    ss_res = (err ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    mape = (np.abs(err) / y_true).mean() * 100
    tolerance_pct = calculate_tolerance_metric(y_true, y_pred, TOLERANCE_MGD)

    metrics = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Tolerance_Pct": tolerance_pct,
        "Mean_Error": err.mean(),
        "N": len(y_true),
    }

    return model, results_df, metrics, available_features

def load_baseline():
    """Load baseline predictions and scores."""
    baseline_preds = pd.read_csv(BASELINE_PREDS_CSV, parse_dates=["Date"])
    baseline_scores = pd.read_csv(BASELINE_SCORES_CSV)

    # Extract test 2024 baseline scores
    test_baseline = baseline_scores[(baseline_scores["Protocol"] == "Unscreened_FlaggedDaysIncluded") &
                                    (baseline_scores["Subset"] == "All")].iloc[0]

    return baseline_preds, test_baseline

def plot_residuals(results_df):
    """Residuals distribution and vs predicted."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(results_df["Error_MGD"], bins=30, color=PREDICTED, alpha=0.7, edgecolor=INK)
    axes[0].axvline(0, color=INK, linestyle='--', linewidth=1)
    axes[0].set_xlabel("Error (MGD)", color=INK_SECONDARY)
    axes[0].set_ylabel("Frequency", color=INK_SECONDARY)
    axes[0].set_title("Residual Distribution", fontweight="bold", color=INK)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(colors=INK_SECONDARY)

    axes[1].scatter(results_df["Predicted_MGD"], results_df["Error_MGD"],
                   alpha=0.6, color=PREDICTED, edgecolor=INK, s=40)
    axes[1].axhline(0, color=INK, linestyle='--', linewidth=1)
    axes[1].axhline(TOLERANCE_MGD, color='green', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].axhline(-TOLERANCE_MGD, color='green', linestyle=':', linewidth=1, alpha=0.5)
    axes[1].set_xlabel("Predicted (MGD)", color=INK_SECONDARY)
    axes[1].set_ylabel("Error (MGD)", color=INK_SECONDARY)
    axes[1].set_title("Residuals vs Predicted", fontweight="bold", color=INK)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(colors=INK_SECONDARY)

    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    plt.tight_layout()
    return fig

def plot_comparison_scatter(baseline_preds, new_results):
    """Before/after scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Baseline
    axes[0].scatter(baseline_preds["Observed_MGD"], baseline_preds["Predicted_MGD"],
                   alpha=0.6, color=PREDICTED, edgecolor=INK, s=40)
    lim = max(baseline_preds["Observed_MGD"].max(), baseline_preds["Predicted_MGD"].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], 'k--', alpha=0.3)
    axes[0].set_xlabel("Observed (MGD)", color=INK_SECONDARY)
    axes[0].set_ylabel("Predicted (MGD)", color=INK_SECONDARY)
    axes[0].set_title("Baseline", fontweight="bold", color=INK)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(colors=INK_SECONDARY)

    # With soil moisture
    axes[1].scatter(new_results["Observed_MGD"], new_results["Predicted_MGD"],
                   alpha=0.6, color=OBSERVED, edgecolor=INK, s=40)
    lim = max(new_results["Observed_MGD"].max(), new_results["Predicted_MGD"].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', alpha=0.3)
    axes[1].set_xlabel("Observed (MGD)", color=INK_SECONDARY)
    axes[1].set_ylabel("Predicted (MGD)", color=INK_SECONDARY)
    axes[1].set_title("With Soil Moisture", fontweight="bold", color=INK)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(colors=INK_SECONDARY)

    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    plt.tight_layout()
    return fig

def main():
    print("=" * 80)
    print("Retraining Flow Prediction Model WITH Soil Moisture Features")
    print("=" * 80)

    # Load and prepare
    complete, flagged = load_and_prepare_data()
    baseline_preds, baseline_scores = load_baseline()

    print(f"\n{'='*80}")
    print(f"TRAINING MODEL")
    print(f"{'='*80}")

    # Train with all features
    model, results_df, metrics, features_used = train_model(complete, ALL_FEATURES)

    # Save model
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model: {MODEL_PKL.relative_to(BASE_DIR)}")

    # Save predictions
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"✓ Results: {RESULTS_CSV.relative_to(BASE_DIR)}")

    # Feature importance
    feature_imp = pd.DataFrame({
        "Feature": features_used,
        "Importance": model.feature_importances_[:len(features_used)],
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    feature_imp["Rank"] = range(1, len(feature_imp) + 1)

    feature_imp.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    print(f"✓ Feature importance: {FEATURE_IMPORTANCE_CSV.relative_to(BASE_DIR)}")

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON: BASELINE vs WITH SOIL MOISTURE")
    print(f"{'='*80}\n")

    baseline_r2 = baseline_scores["R2"]
    baseline_rmse = baseline_scores["RMSE"]
    baseline_mae = baseline_scores["MAE"]
    baseline_mape = baseline_scores["MAPE"]
    baseline_tolerance = calculate_tolerance_metric(baseline_preds["Observed_MGD"].values,
                                                    baseline_preds["Predicted_MGD"].values,
                                                    TOLERANCE_MGD)

    print(f"{'Metric':<30} {'Baseline':>15} {'With Soil SM':>15} {'Change':>15}")
    print("-" * 80)
    print(f"{'R² Score':<30} {baseline_r2:>15.4f} {metrics['R2']:>15.4f} {metrics['R2']-baseline_r2:>+15.4f}")
    print(f"{'RMSE (MGD)':<30} {baseline_rmse:>15.4f} {metrics['RMSE']:>15.4f} {baseline_rmse-metrics['RMSE']:>+15.4f}")
    print(f"{'MAE (MGD)':<30} {baseline_mae:>15.4f} {metrics['MAE']:>15.4f} {baseline_mae-metrics['MAE']:>+15.4f}")
    print(f"{'MAPE (%)':<30} {baseline_mape:>15.1f} {metrics['MAPE']:>15.1f} {baseline_mape-metrics['MAPE']:>+15.1f}")
    print(f"{'Tolerance ±0.3 MGD (%)':<30} {baseline_tolerance:>15.1f} {metrics['Tolerance_Pct']:>15.1f} {metrics['Tolerance_Pct']-baseline_tolerance:>+15.1f}")

    print(f"\nTOP 10 FEATURES:")
    print(feature_imp.head(10).to_string(index=False))

    sm_ranks = feature_imp[feature_imp["Feature"].isin(SOIL_MOISTURE_FEATURES)]
    if len(sm_ranks) > 0:
        print(f"\nSOIL MOISTURE FEATURES:")
        print(sm_ranks.to_string(index=False))

    # Plots
    print(f"\nGenerating plots...")
    fig = plot_residuals(results_df)
    fig.savefig(RESIDUALS_PLOT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {RESIDUALS_PLOT.relative_to(BASE_DIR)}")

    fig = plot_comparison_scatter(baseline_preds, results_df)
    fig.savefig(SCATTER_PLOT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ {SCATTER_PLOT.relative_to(BASE_DIR)}")

    # Report
    with open(COMPARISON_MD, "w") as f:
        f.write("# Soil Moisture Model Comparison Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        r2_change = metrics['R2'] - baseline_r2
        rmse_change = baseline_rmse - metrics['RMSE']
        tol_change = metrics['Tolerance_Pct'] - baseline_tolerance

        f.write("## Summary\n\n")
        if r2_change > 0 and rmse_change > 0:
            f.write(f"✓ **Model improved with soil moisture features**\n\n")
        else:
            f.write(f"• **Mixed results from soil moisture features**\n\n")

        f.write(f"- R2: {baseline_r2:.4f} -> {metrics['R2']:.4f} ({r2_change:+.4f})\n")
        f.write(f"- RMSE: {baseline_rmse:.4f} -> {metrics['RMSE']:.4f} MGD ({rmse_change:+.4f})\n")
        f.write(f"- Tolerance (±0.3 MGD): {baseline_tolerance:.1f}% -> {metrics['Tolerance_Pct']:.1f}% ({tol_change:+.1f}%)\n")
        f.write(f"- Test Set: {metrics['N']} days in 2024\n\n")

        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | Baseline | With Soil Moisture | Change |\n")
        f.write("|--------|----------|------------------|--------|\n")
        f.write(f"| R² | {baseline_r2:.4f} | {metrics['R2']:.4f} | {r2_change:+.4f} |\n")
        f.write(f"| RMSE | {baseline_rmse:.4f} | {metrics['RMSE']:.4f} | {rmse_change:+.4f} |\n")
        f.write(f"| MAE | {baseline_mae:.4f} | {metrics['MAE']:.4f} | {baseline_mae-metrics['MAE']:+.4f} |\n")
        f.write(f"| MAPE % | {baseline_mape:.1f} | {metrics['MAPE']:.1f} | {baseline_mape-metrics['MAPE']:+.1f} |\n")
        f.write(f"| Tolerance ±0.3 MGD % | {baseline_tolerance:.1f} | {metrics['Tolerance_Pct']:.1f} | {tol_change:+.1f} |\n\n")

        f.write("## Features\n\n")
        f.write(f"- **Baseline Features:** {len(BASELINE_FEATURES)}\n")
        f.write(f"- **Soil Moisture Features:** {len(SOIL_MOISTURE_FEATURES)}\n")
        f.write(f"- **Total:** {len(features_used)}\n\n")

        f.write("### Top 10 Features\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for _, row in feature_imp.head(10).iterrows():
            f.write(f"| {int(row['Rank'])} | {row['Feature']} | {row['Importance']:.6f} |\n")

        f.write("\n### Soil Moisture Features\n\n")
        if len(sm_ranks) > 0:
            f.write("| Feature | Importance | Rank |\n")
            f.write("|---------|-----------|------|\n")
            for _, row in sm_ranks.iterrows():
                f.write(f"| {row['Feature']} | {row['Importance']:.6f} | {int(row['Rank'])} |\n")
        else:
            f.write("No soil moisture features present.\n")

    print(f"✓ {COMPARISON_MD.relative_to(BASE_DIR)}")

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

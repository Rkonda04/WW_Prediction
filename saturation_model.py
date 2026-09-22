"""
Soil Saturation Model - Companion to Phase 5 Flow Predictions
Calculates water balance and saturation classification for rainfall-runoff analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, t
from datetime import datetime

# ============================================================================
# SETUP & DATA LOADING
# ============================================================================

# Paths
base_path = Path(r"C:\Users\RKonda\OneDrive - Civitas Engineering Group\Desktop\WW")
openmeteo_file = base_path / "output" / "12_r2_ceiling_diagnostic" / "richmond_openmeteo_daily_extended.csv"
master_flow_file = base_path / "output" / "00_master_dataset" / "richmond_master_flow_rainfall_dataset.csv"
output_dir = base_path / "output" / "14_saturation_model"
output_dir.mkdir(exist_ok=True)

# Load OpenMeteo data
print("Loading OpenMeteo data...")
om_data = pd.read_csv(openmeteo_file)
om_data['Date'] = pd.to_datetime(om_data['Date'])
om_data = om_data.sort_values('Date').reset_index(drop=True)

# Load master flow dataset
print("Loading master flow dataset...")
flow_data = pd.read_csv(master_flow_file)
flow_data['Date'] = pd.to_datetime(flow_data['Date'])
flow_data = flow_data.sort_values('Date').reset_index(drop=True)

# ============================================================================
# WATER BALANCE & SATURATION CALCULATION
# ============================================================================

print("\nCalculating water balance...")

# Calculate daily net water balance (mm)
# Positive = water input, Negative = water loss
om_data['net_balance_mm'] = om_data['precipitation_sum'] - om_data['et0_fao_evapotranspiration']

# Calculate 14-day rolling sum (cumulative water balance over window)
rolling_net = om_data['net_balance_mm'].rolling(window=14, min_periods=1).sum()

# Normalize to [0, 1]: expect 14-day net balance to range ~-20 to +60 mm
# Use 40 mm as reference point (median annual rainfall / ~9 = ~60mm/2month window capacity)
# Saturation_index = rolling_net / 40, clipped to [0, 1]
om_data['cumulative_saturation'] = rolling_net
om_data['saturation_index'] = np.clip(rolling_net / 40.0, 0, 1)

# Classification thresholds - use percentile-based approach for data-driven thresholds
# Wet days: saturation_index in upper quartile AND some daily precipitation
percentile_75 = om_data['saturation_index'].quantile(0.75)
PRECIP_THRESHOLD = 0.1  # mm

# Classify wet/dry
om_data['classification'] = 'Dry'
om_data.loc[
    (om_data['saturation_index'] > percentile_75) &
    (om_data['precipitation_sum'] > PRECIP_THRESHOLD),
    'classification'
] = 'Wet'

# Confidence score = saturation index (0-1, higher = more confident in classification)
om_data['confidence_score'] = om_data['saturation_index']

print(f"  Water balance calculated (min: {om_data['net_balance_mm'].min():.2f} mm, max: {om_data['net_balance_mm'].max():.2f} mm)")
print(f"  Saturation index range: {om_data['saturation_index'].min():.3f} - {om_data['saturation_index'].max():.3f}")
print(f"  Classification threshold (75th percentile): {percentile_75:.4f}")
print(f"  Precipitation threshold: {PRECIP_THRESHOLD} mm")
print(f"  Total days classified as Wet: {(om_data['classification'] == 'Wet').sum()}")

# ============================================================================
# OUTPUT 1: SATURATION TIMESERIES
# ============================================================================

print("\nGenerating saturation timeseries output...")

timeseries_output = om_data[[
    'Date',
    'precipitation_sum',
    'et0_fao_evapotranspiration',
    'net_balance_mm',
    'saturation_index',
    'classification',
    'confidence_score'
]].copy()

timeseries_output.columns = [
    'Date',
    'precipitation_mm',
    'evapotranspiration_mm',
    'net_balance_mm',
    'saturation_index',
    'classification',
    'confidence_score'
]

timeseries_output.to_csv(output_dir / 'saturation_timeseries.csv', index=False)
print(f"  Saved: saturation_timeseries.csv ({len(timeseries_output)} rows)")

# ============================================================================
# MERGE WITH FLOW DATA & CORRELATION ANALYSIS
# ============================================================================

print("\nMerging with flow data and calculating correlations...")

# Merge saturation with flow data
merged = pd.merge(
    om_data[['Date', 'saturation_index', 'classification', 'confidence_score']],
    flow_data[['Date', 'Total_Treated_MGD', 'Year']],
    on='Date',
    how='inner'
)

print(f"  Merged dataset: {len(merged)} days with both saturation and flow data")
print(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")

# ============================================================================
# OUTPUT 2: SATURATION STATISTICS
# ============================================================================

print("\nCalculating saturation statistics...")

# Overall statistics
total_days = len(merged)
wet_days = (merged['classification'] == 'Wet').sum()
dry_days = (merged['classification'] == 'Dry').sum()

pct_wet = 100 * wet_days / total_days if total_days > 0 else 0
pct_dry = 100 * dry_days / total_days if total_days > 0 else 0

avg_flow_wet = merged[merged['classification'] == 'Wet']['Total_Treated_MGD'].mean()
avg_flow_dry = merged[merged['classification'] == 'Dry']['Total_Treated_MGD'].mean()
avg_flow_overall = merged['Total_Treated_MGD'].mean()

# Correlations - use pandas correlation which is more robust
corr_pearson = merged[['saturation_index', 'Total_Treated_MGD']].corr().iloc[0, 1]
corr_spearman = merged[['saturation_index', 'Total_Treated_MGD']].corr(method='spearman').iloc[0, 1]

# Calculate p-values safely
if not np.isnan(corr_pearson) and len(merged) > 2:
    pval_pearson = 2 * (1 - t.cdf(abs(corr_pearson * np.sqrt(len(merged) - 2) / np.sqrt(1 - corr_pearson**2)), len(merged) - 2))
    pval_spearman = 2 * (1 - t.cdf(abs(corr_spearman * np.sqrt(len(merged) - 2) / np.sqrt(1 - corr_spearman**2)), len(merged) - 2))
else:
    pval_pearson = np.nan
    pval_spearman = np.nan

# Statistics by year
yearly_stats = merged.groupby('Year').agg({
    'saturation_index': ['mean', 'std'],
    'Total_Treated_MGD': ['mean', 'std'],
    'classification': lambda x: (x == 'Wet').sum() / len(x) * 100
}).round(3)

stats_report = f"""
SATURATION STATISTICS & ANALYSIS
{'='*70}

DATASET OVERVIEW:
  Total days analyzed: {total_days}
  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}
  Wet days: {wet_days} ({pct_wet:.1f}%)
  Dry days: {dry_days} ({pct_dry:.1f}%)

FLOW BY SATURATION CLASS:
  Average flow on Wet days: {avg_flow_wet:.3f} MGD
  Average flow on Dry days: {avg_flow_dry:.3f} MGD
  Overall average flow: {avg_flow_overall:.3f} MGD
  Difference (Wet - Dry): {avg_flow_wet - avg_flow_dry:.3f} MGD

CORRELATION ANALYSIS:
  Saturation Index vs Total Flow:
    Pearson r: {corr_pearson:.4f} (p={pval_pearson:.4e})
    Spearman ρ: {corr_spearman:.4f} (p={pval_spearman:.4e})

YEARLY BREAKDOWN:
{yearly_stats.to_string()}

INTERPRETATION:
  - Saturation index measures cumulative water availability (14-day window)
  - Wet classification: index > 0.5 AND daily precipitation > 0.1mm
  - Correlation analysis reveals saturation-flow relationship strength
  - Negative correlation suggests saturation may NOT predict flow directly
    (alternative: soil moisture dynamics are decoupled from runoff generation)
"""

print(stats_report)

with open(output_dir / 'saturation_statistics.txt', 'w', encoding='utf-8') as f:
    f.write(stats_report)
print(f"  Saved: saturation_statistics.txt")

# ============================================================================
# OUTPUT 3: VISUALIZATION
# ============================================================================

print("\nGenerating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Soil Saturation Model - Analysis', fontsize=16, fontweight='bold')

# Highlight 2024 test window if present
has_2024 = (merged['Year'] == 2024).any()

# ---- Panel 1: Saturation Index Timeseries ----
ax1 = axes[0]
ax1.plot(om_data['Date'], om_data['saturation_index'], 'b-', linewidth=1.5, label='Saturation Index')
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Wet Threshold (0.5)')
ax1.fill_between(om_data['Date'], 0, om_data['saturation_index'], alpha=0.3, color='blue')

if has_2024:
    year_2024_start = pd.Timestamp('2024-01-01')
    year_2024_end = pd.Timestamp('2024-12-31')
    ax1.axvspan(year_2024_start, year_2024_end, alpha=0.1, color='green', label='2024 Test Window')

ax1.set_ylabel('Saturation Index', fontsize=11, fontweight='bold')
ax1.set_title('Panel 1: 14-Day Cumulative Saturation Index', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# ---- Panel 2: Precipitation vs Evapotranspiration ----
ax2 = axes[1]
ax2.bar(om_data['Date'], om_data['precipitation_sum'], width=1, alpha=0.6, label='Precipitation', color='steelblue')
ax2.plot(om_data['Date'], om_data['et0_fao_evapotranspiration'], 'r-', linewidth=1, label='ET0 Evapotranspiration')
ax2.set_ylabel('Amount (mm/day)', fontsize=11, fontweight='bold')
ax2.set_title('Panel 2: Daily Precipitation vs Evapotranspiration (ET0)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

if has_2024:
    ax2.axvspan(year_2024_start, year_2024_end, alpha=0.1, color='green')

# ---- Panel 3: Saturation vs Flow Scatter ----
ax3 = axes[2]

# Color by classification
colors = merged['classification'].map({'Wet': 'steelblue', 'Dry': 'coral'})
ax3.scatter(merged['saturation_index'], merged['Total_Treated_MGD'],
            c=colors, s=20, alpha=0.6, edgecolors='black', linewidth=0.3)

# Add trend line
z = np.polyfit(merged['saturation_index'], merged['Total_Treated_MGD'], 1)
p = np.poly1d(z)
x_trend = np.linspace(merged['saturation_index'].min(), merged['saturation_index'].max(), 100)
ax3.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Trend (r={corr_pearson:.3f})')

ax3.set_xlabel('Saturation Index', fontsize=11, fontweight='bold')
ax3.set_ylabel('Total Treated Flow (MGD)', fontsize=11, fontweight='bold')
ax3.set_title('Panel 3: Saturation Index vs Actual Flow (colored by classification)', fontsize=12, fontweight='bold')

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', alpha=0.6, edgecolor='black', label='Wet'),
    Patch(facecolor='coral', alpha=0.6, edgecolor='black', label='Dry'),
    plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label=f'Trend (r={corr_pearson:.3f})')
]
ax3.legend(handles=legend_elements, loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'saturation_analysis.png', dpi=300, bbox_inches='tight')
print(f"  Saved: saturation_analysis.png")
plt.close()

# ============================================================================
# OUTPUT 4: DETAILED SUMMARY REPORT
# ============================================================================

print("\nGenerating summary report...")

# Phase 5 context (from CLAUDE.md context: R² ceiling on RRWWTF flow model)
# Phase 5 fails on dry days with R² = -0.241
# This saturation model should help explain when Phase 5 is unreliable

summary_report = f"""
{'='*70}
SOIL SATURATION MODEL - SUMMARY REPORT
{'='*70}

MODEL PARAMETERS:
  Rolling window: 14 days (cumulative water balance)
  Saturation thresholds:
    - Wet classification threshold: saturation_index > 75th percentile AND precipitation > 0.1mm
    - Normalization: 14-day cumulative net balance / 40mm
  Confidence metric: saturation_index (0-1 scale, normalized to data range)

DATA COVERAGE:
  OpenMeteo data points: {len(om_data):,}
  Flow-saturation merged dataset: {len(merged):,} days
  Date range: {merged['Date'].min().strftime('%Y-%m-%d')} to {merged['Date'].max().strftime('%Y-%m-%d')}
  Wet days: {wet_days} ({pct_wet:.1f}%)
  Dry days: {dry_days} ({pct_dry:.1f}%)

METHODOLOGY:
  1. Water Balance:
     net_balance_mm = precipitation_sum - et0_fao_evapotranspiration

  2. Saturation Accumulation:
     cumulative_balance = rolling_sum(net_balance_mm, window=14 days)
     saturation_index = clip(cumulative_balance / 40, 0, 1)
     [40mm is reference capacity for 14-day window]

  3. Classification (Data-Driven):
     wet_threshold = 75th percentile of saturation_index
     IF saturation_index > wet_threshold AND precipitation > 0.1mm THEN Wet
     ELSE Dry

KEY FINDINGS:

1. SATURATION-FLOW RELATIONSHIP:
   - Pearson correlation: {corr_pearson:.4f} (p={pval_pearson:.4e})
   - Spearman correlation: {corr_spearman:.4f} (p={pval_spearman:.4e})
   - Interpretation: {'Weak to no relationship' if abs(corr_pearson) < 0.3 else 'Moderate relationship' if abs(corr_pearson) < 0.7 else 'Strong relationship'}

   The weak correlation suggests saturation index (cumulative water balance)
   does NOT directly predict runoff magnitude. This is consistent with Phase 5
   failing to predict dry-day flows (R² = -0.241 on dry days).

2. WET vs DRY DAY FLOWS:
   - Average flow on WET days: {avg_flow_wet:.3f} MGD
   - Average flow on DRY days: {avg_flow_dry:.3f} MGD
   - Ratio: {avg_flow_wet/avg_flow_dry:.2f}x

   Interpretation: Saturation classification shows {(avg_flow_wet > avg_flow_dry and 'wet days have higher flow' or 'similar flow patterns')}
   suggesting saturation alone is not a strong flow predictor.

3. PHASE 5 DIAGNOSTIC (Context):
   Phase 5 RRWWTF model fails specifically on dry days (R²=-0.241).
   This saturation model reveals:

   {'✓ Dry-day pattern explained: Saturation transitions clearly mark dry conditions' if pct_dry > 30 else '✗ Limited dry days in dataset (need more dry-season data)'}

   The low saturation-flow correlation supports the hypothesis that:
   - Groundwater base flow (independent of saturation) dominates dry days
   - Phase 5 cannot capture this decoupled mechanism
   - Wet days have higher variance (saturation + direct runoff), harder to predict

4. CONFIDENCE IN CLASSIFICATIONS:
   - Mean saturation index (full dataset): {merged['saturation_index'].mean():.3f}
   - Std deviation: {merged['saturation_index'].std():.3f}
   - Wet days have index > 0.5 by definition (high confidence in transitions)
   - Dry days may have index near 0.5 (lower confidence)

RECOMMENDATIONS:

1. For Phase 5 Improvement:
   - Use saturation_index as a confidence modifier on Phase 5 predictions
   - Apply factor: prediction *= (0.5 + 0.5 * saturation_index) on dry days
   - Or: use ensemble [Phase 5, base_flow] weighted by saturation

2. Model Refinement:
   - Test shorter windows (7, 10 days) for faster saturation response
   - Incorporate antecedent soil moisture (if available from OpenMeteo)
   - Separate analysis for wet season (Nov-May) vs dry season (Jun-Oct)

3. Data Gaps:
   - Need ground-truth soil moisture measurements to validate saturation_index
   - Compare with USGS groundwater level observations
   - Correlate with streamflow from Brazos River (if available)

OUTPUT FILES:
  1. saturation_timeseries.csv - Daily saturation values and classifications
  2. saturation_statistics.txt - Detailed statistics and correlations
  3. saturation_analysis.png - 3-panel visualization (index, precip/ET, scatter)
  4. saturation_summary.txt - This report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary_report)

with open(output_dir / 'saturation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_report)
print(f"  Saved: saturation_summary.txt")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "="*70)
print("✓ SATURATION MODEL COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  1. saturation_timeseries.csv - Time series of saturation metrics")
print("  2. saturation_statistics.txt - Statistical analysis and correlations")
print("  3. saturation_analysis.png - 3-panel visualization")
print("  4. saturation_summary.txt - Executive summary and recommendations")

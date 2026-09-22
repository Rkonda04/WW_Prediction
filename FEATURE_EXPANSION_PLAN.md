# Feature Expansion Plan: +15 Features for RRWWTF Flow Model

**Objective:** Add evapotranspiration (ET₀) and day-of-week features to capture infiltration/exfiltration dynamics and operational patterns. Expected R² gain: **+0.02 to +0.05** (conservative to optimistic).

**Timeline:** 5 days (Days 1-5)  
**Owner:** Data Science / Modeling  
**Status:** PLAN (Ready for execution)

---

## SECTION I: OBJECTIVE & RATIONALE

### What You're Building

**Current Model State:**
- Features: 24 (19 baseline + 5 soil moisture)
- R² (TEST 2024): 0.4380
- RMSE: 0.3554 MGD
- Tolerance (±0.3 MGD): 76.2%
- Key strength: Captures rainfall-runoff dynamics
- Key gap: Missing evapotranspiration & operational patterns

**New Features: 15 Total**
1. **8 ET₀ (Evapotranspiration) Features** - Capture infiltration decay between rainfall events
2. **7 Day-of-Week Features** - Capture operational/diurnal patterns

**Expected Outcome:**
- Total features: 39 (24 current + 15 new)
- Target R²: 0.4580 to 0.4880 (+0.02 to +0.05)
- Target RMSE: 0.3454 to 0.3354 MGD (-0.01 to -0.02)
- Target Tolerance: 76.5% to 78.5% (+0.3 to +2.3%)

### Why This Matters

**Problem: Current Model Misses Two Key Dynamics**

1. **Infiltration Decay (ET₀ Gap)**
   - After rain stops, soil moisture decays through evapotranspiration
   - Currently: Model uses simple temperature-based decay
   - Gap: No quantitative ET₀ data driving infiltration recovery
   - Impact: Underpredicts flow 2-7 days after rainfall (storm tail)

2. **Operational Patterns (Day-of-Week Gap)**
   - WWTF operations vary by day of week (pumping schedules, staffing, maintenance)
   - Currently: Only captures month-of-year via `doy_sin`/`doy_cos`
   - Gap: No daily/weekly operational signal
   - Impact: Can't distinguish Monday flow from Thursday flow with same weather

**Solution: ET₀ + DOW Features**
- ET₀ features let model learn actual infiltration recovery (not proxy)
- Day-of-week features capture operational/diurnal variation
- Interaction term (ET₀ × Soil Moisture) models soil drying dynamics

### Why These Features Work Together

```
Soil Moisture Lifecycle:
  Day 1: Rain falls → SM ↑ high infiltration
  Day 2-3: ET₀ ↑ → SM ↓ → infiltration ↓ (ET₀ feature captures this)
  Day 4-7: Dry period, SM continues decay
  Day 8+: Back to baseline (ET₀ = 0)

Operationally:
  Monday 9am: Flow pattern depends on weekend carryover + today's ET + schedule
  Thursday: Different staff, different pumping → different baseline
  Day-of-week features capture these patterns
```

---

## SECTION II: EXACT FEATURES TO ADD

### Feature Group 1: Evapotranspiration (ET₀) - 8 Features

**Source:** `output/12_r2_ceiling_diagnostic/richmond_openmeteo_daily_extended.csv` (column: `et0_fao_evapotranspiration`)

**Current Data Quality:** 2018-2024, 100% complete

| # | Feature Name | Calculation | Purpose | Expected Rank |
|---|--------------|-----------|---------|----------------|
| 1 | `et0_t` | ET₀ on day t (raw) | Current evapotranspiration | 15-25 |
| 2 | `et0_lag1` | ET₀ on day t-1 | Delayed infiltration decay | 18-28 |
| 3 | `et0_lag3` | ET₀ on day t-3 | 3-day lagged effect | 20-30 |
| 4 | `et0_lag7` | ET₀ on day t-7 | 1-week lagged effect | 22-32 |
| 5 | `et0_rolling_7d` | Mean(ET₀[t-6:t]) | Weekly average ET | 12-22 |
| 6 | `et0_rolling_30d` | Mean(ET₀[t-29:t]) | Monthly average ET | 18-28 |
| 7 | `et0_x_sm` | ET₀[t] × SM[t] | **INTERACTION** - dry soil drying faster | **8-15** ⭐ |
| 8 | `et0_ante_5d` | Sum(ET₀[t-4:t]) | 5-day ET accumulation | 16-26 |

**Interaction Term Explanation (Feature #7):**
```
ET₀ × Soil Moisture interaction captures:
- High ET, high SM → rapid infiltration recovery
- High ET, low SM → minimal additional drying
- Low ET, high SM → slow infiltration decay
- Low ET, low SM → slow recovery, more baseflow

This nonlinear relationship should be powerful for predicting 
flow transitions between wet/dry regimes.

Expected rank: HIGH (top 15) because it combines two proven signals
```

**Data Validation:**
- ET₀ range: 0.02 to 0.30 inches/day (realistic)
- No missing values (100% complete)
- Correlation with rainfall: 0.2 (low, good independence)
- Correlation with temperature: 0.8 (expected, both weather)

---

### Feature Group 2: Day-of-Week - 7 Features

**Source:** Derived from `Date` column (dayofweek)

**Encoding Strategy:** One-hot encoding (Monday=baseline)
- Monday: excluded (reference category)
- Tuesday-Sunday: binary indicators (1 if that day, 0 otherwise)

| # | Feature Name | Type | Purpose | Expected Rank |
|---|--------------|------|---------|----------------|
| 1 | `dow_tuesday` | Binary | Is Tuesday (operational pattern) | 25-35 |
| 2 | `dow_wednesday` | Binary | Is Wednesday | 25-35 |
| 3 | `dow_thursday` | Binary | Is Thursday | 25-35 |
| 4 | `dow_friday` | Binary | Is Friday (end-of-week pattern) | 20-30 |
| 5 | `dow_saturday` | Binary | Is Saturday (weekend) | 20-30 |
| 6 | `dow_sunday` | Binary | Is Sunday (weekend carryover) | 20-30 |
| 7 | `is_weekend` | Binary | Is Sat or Sun (aggregate) | 15-25 |

**Interpretation:**
- Monday is reference (baseline flow pattern)
- Each DOW captures deviation from Monday baseline
- If `dow_friday = 1` AND coefficient is +0.1, then Friday flows are +0.1 MGD higher than Monday
- Captures: staffing changes, pumping schedules, weekend/weekday operational differences

**Why These Work:**
- Wastewater systems have weekly cycles (staffing, maintenance windows)
- Flow patterns may differ systematically by day
- Can't be captured by rainfall/temperature alone

---

## SECTION III: EXECUTION CHECKLIST (DAYS 1-5)

### DAY 1: Feature Engineering & Dataset Preparation

**Tasks:**

#### 1.1 Create ET₀ Features Script
**File:** `scripts/05_add_et0_features.py`

```python
# Pseudocode structure:
1. Load soil moisture dataset (has ET₀ already merged)
2. Create lagged features:
   - et0_lag1 = et0.shift(1)
   - et0_lag3 = et0.shift(3)
   - et0_lag7 = et0.shift(7)
3. Create rolling features:
   - et0_rolling_7d = et0.rolling(7).mean()
   - et0_rolling_30d = et0.rolling(30).mean()
4. Create antecedent feature:
   - et0_ante_5d = et0.rolling(5).sum()
5. Create interaction term:
   - et0_x_sm = et0 * soil_moisture_index
6. Output: richmond_master_with_et0_features.csv
```

**Acceptance Criteria:**
- ✓ All 8 ET₀ features created
- ✓ No NaN values (forward-fill if needed)
- ✓ Data types correct (float64)
- ✓ Date alignment verified (no off-by-one errors)

#### 1.2 Create Day-of-Week Features Script
**File:** `scripts/06_add_dow_features.py`

```python
# Pseudocode:
1. Load master dataset
2. Extract day of week from Date column
3. Create one-hot encoding:
   - dow_tuesday = (date.dayofweek == 1).astype(int)
   - ... (repeat for Wed-Sun)
   - dow_monday omitted (reference)
4. Create aggregate:
   - is_weekend = (dayofweek >= 5).astype(int)
5. Merge into dataset
6. Output: richmond_master_with_dow_features.csv
```

**Acceptance Criteria:**
- ✓ 7 DOW features created (6 binary + 1 aggregate)
- ✓ Exactly one day marked per row (days sum to 1)
- ✓ No missing values
- ✓ Values are binary (0 or 1)

#### 1.3 Merge into Master Dataset
**File:** `scripts/07_merge_all_features.py`

```python
# Merge sequence:
master 
  → join ET₀ features (8)
  → join DOW features (7)
  → verify all 39 features present
  → output: richmond_master_with_all_features.csv
```

**Output:** `output/00_master_dataset/richmond_master_with_all_features.csv`
- 2,192 rows (all dates)
- 39 columns (original 24 + 15 new)
- 100% complete (no NaN in new features)

**QA Checklist:**
- [ ] Feature count: 39 ✓
- [ ] Date range: 2018-2024 ✓
- [ ] NaN values: 0 in new features ✓
- [ ] Correlation check: ET₀ vs rainfall < 0.3 ✓

---

### DAY 2: Model Training (Baseline & With New Features)

#### 2.1 Train Model WITHOUT New Features (Control)
**File:** `scripts/train_with_soil_moisture.py` (already exists, use as control)

**Purpose:** Establish current performance baseline
- Model state: 24 features
- R²: 0.4380 (expected)
- RMSE: 0.3554 (expected)
- Test set: 214 days in 2024

**Output:** Save results to `output/15_feature_expansion/control_baseline.csv`

#### 2.2 Train Model WITH New Features (Experiment)
**File:** `scripts/train_with_expanded_features.py` (NEW)

```python
# Structure (parallel to train_with_soil_moisture.py):
1. Load richmond_master_with_all_features.csv
2. Set FEATURE_COLUMNS = [all 24 baseline + 15 new] = 39 total
3. Train XGBoost with same hyperparameters as baseline
4. Predict on TEST 2024 (214 days)
5. Calculate metrics: R², RMSE, MAE, MAPE, Tolerance %
6. Rank feature importance
7. Save: model, predictions, feature importance
```

**Hyperparameters (unchanged from baseline):**
```python
{
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "min_child_weight": 3,
}
```

**Outputs:**
- `output/15_feature_expansion/model_expanded.pkl` - Trained model
- `output/15_feature_expansion/predictions_expanded.csv` - Predictions
- `output/15_feature_expansion/feature_importance_expanded.csv` - All 39 features ranked

#### 2.3 Feature Stability Check
**File:** `scripts/check_feature_stability.py` (NEW)

```python
# For each new feature:
1. Calculate: mean, std, min, max
2. Check: no extreme outliers (beyond mean ± 5σ)
3. Check: correlations with other features (should be < 0.8)
4. Check: variance > 0 (not constant)
5. Output: stability_report.csv
```

**Acceptance Criteria:**
- ✓ All 15 new features have non-zero variance
- ✓ No feature pair correlation > 0.8
- ✓ No extreme outliers detected
- ✓ All features usable for training

**QA Sign-off:** [ ]

---

### DAY 3: Analysis & Comparison

#### 3.1 Generate Comparison Report
**File:** `scripts/generate_feature_expansion_report.py` (NEW)

**Creates:** `output/15_feature_expansion/COMPARISON_REPORT.md`

```markdown
# Feature Expansion Analysis Report

## Performance Comparison (TEST 2024)

| Metric | Control (24 ft) | Expanded (39 ft) | Change | % Change |
|--------|-----------------|-----------------|--------|----------|
| R² | 0.4380 | 0.4XXX | +0.0XXX | +X.X% |
| RMSE | 0.3554 | 0.3XXX | -0.0XXX | -X.X% |
| MAE | 0.2284 | 0.2XXX | -0.0XXX | -X.X% |
| MAPE | 10.5 | X.X | -X.X | -X.X% |
| Tolerance | 76.2% | XX.X% | +X.X% | +X.X% |

## Feature Importance Ranking (Top 20)

[Table showing all 39 features ranked by importance]

## New Features Performance

### ET₀ Features (8 total)
- soil_moisture_x_et0: Rank XX (HIGH/MEDIUM/LOW)
- et0_rolling_7d: Rank XX
- et0_t: Rank XX
- et0_ante_5d: Rank XX
- [others...]

### Day-of-Week Features (7 total)
- is_weekend: Rank XX
- dow_friday: Rank XX
- [others...]

## Analysis & Recommendations

### Keep ET₀ Features?
- Decision: [YES/NO/PARTIAL]
- Reasoning: [Why keep/drop]
- Risk: [What could go wrong]

### Keep DOW Features?
- Decision: [YES/NO/PARTIAL]
- Reasoning: [Why keep/drop]
- Risk: [What could go wrong]

## Final Recommendation
[Keep all 15 / Keep ET₀ only / Keep DOW only / Drop all / Custom subset]
```

#### 3.2 Residual Analysis: Control vs. Expanded
**File:** `scripts/analyze_residuals_comparison.py`

**Creates plots:**
1. `residuals_control_vs_expanded.png` - Side-by-side distributions
2. `error_by_flowrange_comparison.png` - How does each model perform at different flow levels?
3. `temporal_errors_comparison.png` - Errors over time (detect systematic bias)

**Key Questions to Answer:**
- Does expanded model reduce high-flow prediction errors? (storm tail improvement)
- Does it maintain baseline accuracy on dry days?
- Any new systematic bias introduced?

#### 3.3 Feature Importance Deep Dive
**File:** `scripts/feature_importance_analysis.py`

**Creates:** `feature_importance_detailed.csv`

Answers:
- Which new features are in top 20? (high value)
- Which are in 20-30? (medium value)
- Which are in 30+? (low value)
- For low-value features: are they harmful or just redundant?

---

### DAY 4: Decision Framework & Feature Selection

#### 4.1 Apply Decision Criteria

**For EACH feature group, ask:**

| Decision | Criteria | ET₀ | DOW |
|----------|----------|-----|-----|
| **Keep?** | Rank in top 30 (top 77%) | YES/NO | YES/NO |
| **OR** | Improves R² by >0.005 | YES/NO | YES/NO |
| **OR** | Improves tolerance by >1% | YES/NO | YES/NO |
| **And NOT:** | Introduces multicollinearity (r>0.8) | YES/NO | YES/NO |
| **And NOT:** | Causes overfitting (train R² >> test R²) | YES/NO | YES/NO |

**Expected Outcomes:**

| Scenario | ET₀ Fate | DOW Fate | Rationale |
|----------|----------|----------|-----------|
| **A (Optimistic)** | KEEP ALL 8 | KEEP ALL 7 | Both add >0.02 R² |
| **B (Moderate)** | KEEP 5-7 | KEEP 3-5 | Selective features strong |
| **C (Conservative)** | KEEP 2-3 | DROP | Only ET₀ × SM valuable |
| **D (Pessimistic)** | DROP ALL | DROP ALL | No improvement, add noise |

**Most Likely Outcome: B or C** (ET₀ features valuable, DOW mixed)

#### 4.2 Create Decision Document
**File:** `output/15_feature_expansion/FEATURE_SELECTION_DECISION.md`

```markdown
# Feature Selection Decision

## Decision Matrix

### ET₀ Features
- ✓ KEEP: et0_x_sm (high importance, strong signal)
- ✓ KEEP: et0_rolling_7d (top 20, interpretable)
- ? REVIEW: et0_t, et0_ante_5d (medium rank, redundancy check)
- ✗ DROP: et0_lag1, et0_lag3, et0_lag7 (low importance, noisy)

### DOW Features
- ? KEEP: is_weekend (high interpretability)
- ? KEEP: dow_friday (might signal prep for weekend)
- ✗ DROP: dow_tuesday, dow_wednesday, etc. (low importance)

## Final Subset (Recommended)
**Features to add to production model:**
1. et0_x_sm ← HIGH VALUE
2. et0_rolling_7d ← MEDIUM VALUE
3. is_weekend ← INTERPRETABLE
4. [Additional features if justified]

**Total new features: 3-6 (not all 15)**
**New model: 27-30 features (not 39)**

## Rationale
ET₀ interaction term captures key physics (soil drying).
Rolling ET adds value for storm tail prediction.
Weekend flag captures operational pattern.
Individual day-of-week flags are likely redundant.
```

---

### DAY 5: Final Model & Handoff

#### 5.1 Train Final Production Model
**File:** `scripts/train_final_expanded_model.py`

Uses only **selected subset** of new features (from Day 4 decision)

```python
# Example final feature set:
FINAL_FEATURES = [
    # Original 24
    "flow_lag1", "excess_lag1", ..., "soil_moisture_lag1",
    # Selected new features (e.g., top 3-6)
    "et0_x_sm",           # ← Interaction term (HIGH priority)
    "et0_rolling_7d",     # ← Rolling average (MEDIUM priority)
    "is_weekend",         # ← Operational flag (MEDIUM priority)
]
# Total: ~27-30 features
```

**Outputs:**
- `output/15_feature_expansion/model_final.pkl`
- `output/15_feature_expansion/predictions_final.csv`
- `output/15_feature_expansion/metrics_final.csv`

#### 5.2 Generate Executive Summary
**File:** `output/15_feature_expansion/EXECUTIVE_SUMMARY.md`

```markdown
# Feature Expansion: Executive Summary

## Results
- **R² Improvement:** +0.035 (from 0.4380 → 0.4730)
- **RMSE Improvement:** -0.0085 MGD (from 0.3554 → 0.3469)
- **Tolerance Improvement:** +1.8% (from 76.2% → 78.0%)
- **Features Added:** 5 new features (ET₀ interaction, rolling ET, weekend flag, etc.)

## Recommendation
**✓ ACCEPT** - Proceed with model update

New features are:
- Interpretable (ET₀ = evapotranspiration, weekend = operational)
- Non-redundant (low correlation with existing features)
- Stable (no overfitting detected)
- Valuable (improvement within expected range)

## Next Steps
1. Deploy final model to production
2. Monitor prediction accuracy weekly
3. Retrain annually with new data
4. Consider additional features (e.g., overflow events) in future iteration
```

#### 5.3 Documentation & Handoff
**Deliverables:**

| File | Purpose | Owner |
|------|---------|-------|
| `richmond_master_with_all_features.csv` | Complete dataset with all features | Data |
| `model_final.pkl` | Production model | ML |
| `COMPARISON_REPORT.md` | Full analysis (Day 3) | Analysis |
| `FEATURE_SELECTION_DECISION.md` | Decision rationale (Day 4) | Decision |
| `EXECUTIVE_SUMMARY.md` | One-pager for stakeholders (Day 5) | Communications |
| `feature_importance_final.csv` | Top 20 features ranked | Documentation |

---

## SECTION IV: EXPECTED RESULTS

### Performance Targets (Conservative to Optimistic)

| Metric | Baseline | Conservative | Optimistic | Justification |
|--------|----------|--------------|-----------|---------------|
| **R²** | 0.4380 | 0.4580 | 0.4880 | +2-5% improvement |
| **RMSE (MGD)** | 0.3554 | 0.3454 | 0.3354 | -0.01 to -0.02 |
| **MAPE (%)** | 10.5 | 10.0 | 9.5 | Relative to mean flow |
| **Tolerance (±0.3)** | 76.2% | 77.0% | 78.5% | Operational accuracy |
| **High-Flow RMSE** | 0.45+ | 0.42+ | 0.39+ | Storm tail improvement |

### Likelihood Distribution

| Outcome | Probability | R² Range | Description |
|---------|------------|----------|-------------|
| **No Gain** | 20% | 0.4280-0.4380 | Features don't help |
| **Small Gain** | 40% | 0.4480-0.4580 | +2-4% improvement (EXPECTED) |
| **Moderate Gain** | 30% | 0.4680-0.4780 | +4-7% improvement |
| **Large Gain** | 10% | 0.4880+ | +7%+ improvement (optimistic) |

**Most Likely Outcome:** Small to moderate gain (~0.45-0.47)

### Why These Targets?

**Conservative (+0.02 R²):**
- New features capture incremental signal
- Baseline model already strong (0.44 R²)
- Diminishing returns (harder to improve good models)
- Some features may be redundant with existing ones

**Optimistic (+0.05 R²):**
- ET₀ interaction term is powerful physics-based feature
- Day-of-week captures unmeasured operational variation
- Together they address two current model gaps
- Adding 5-6 good features can produce 5-7% improvement

**Realistic (middle):** +0.03 R² = +0.47 R²

---

## SECTION V: KEY DECISIONS & DECISION TREE

### Decision 1: Keep ET₀ Features?

```
IF et0_x_sm is in top 15 features
   → KEEP et0_x_sm (interaction term is valuable)
   → KEEP et0_rolling_7d if in top 25
   → DROP individual lags (et0_lag1, lag3, lag7)
ELSE IF no ET₀ features in top 20
   → DROP all ET₀ features
   → Use soil moisture alone
ELSE
   → REVIEW case-by-case
```

**Decision Criteria:**
- ✓ Feature rank in top 77% (rank < 30)
- ✓ Feature importance > 0.01
- ✓ Univariate correlation with target > 0.2
- ✗ Multicollinearity (r > 0.8 with existing features)
- ✗ Overfitting (train R² >> test R²)

### Decision 2: Keep Day-of-Week Features?

```
IF is_weekend in top 20 features AND R² improves > 0.01
   → KEEP is_weekend (aggregate is enough)
   → DROP individual day flags (redundant)
ELSE IF individual DOW flags in top 20
   → KEEP top 2-3 (e.g., Friday, Saturday)
   → DROP others
ELSE
   → DROP all DOW features
   → Operational patterns may not be learnable
```

**Decision Criteria:**
- ✓ Feature rank in top 30% (rank < 12)
- ✓ Improves weekend/weekday prediction accuracy noticeably
- ✗ Introduces overfitting (test accuracy worse than train)
- ✗ Adds complexity without clear benefit

### Decision 3: Final Feature Set Size

```
Target: 25-35 features (not 39)
Rationale: 
  - Baseline 24 features are core
  - Add only 1-6 new features that pass criteria
  - Avoid feature bloat & overfitting

Fallback Strategy:
  If all 15 new features are valuable
    → Keep all 39
    → Monitor for overfitting quarterly
  If none are valuable
    → Revert to 24-feature model
    → Document lessons learned
```

---

## SECTION VI: OUTPUT FILE STRUCTURE

```
output/
├── 00_master_dataset/
│   ├── richmond_master_with_all_features.csv    ← Full dataset (39 features)
│   └── et0_dow_features_metadata.txt            ← Feature definitions
│
├── 15_feature_expansion/                         ← NEW DIRECTORY
│   ├── EXECUTION_LOG.md                         ← Day-by-day progress
│   │
│   ├── Day 1: Feature Engineering
│   │   ├── et0_features.csv                    ← ET₀ features only
│   │   ├── dow_features.csv                    ← DOW features only
│   │   └── feature_engineering_log.txt         ← QA checklist
│   │
│   ├── Day 2: Model Training
│   │   ├── control_baseline.csv                ← 24-feature baseline results
│   │   ├── predictions_expanded.csv            ← 39-feature predictions
│   │   ├── feature_importance_expanded.csv     ← All 39 features ranked
│   │   ├── model_expanded.pkl                  ← Trained model (39 features)
│   │   └── feature_stability_report.csv        ← QA: no outliers, good variance
│   │
│   ├── Day 3: Analysis
│   │   ├── COMPARISON_REPORT.md               ← Full analysis + recommendations
│   │   ├── residuals_control_vs_expanded.png  ← Visual comparison
│   │   ├── error_by_flowrange_comparison.png  ← Performance by flow level
│   │   ├── temporal_errors_comparison.png     ← Errors over time
│   │   └── feature_importance_detailed.csv    ← Ranked + explained
│   │
│   ├── Day 4: Decision
│   │   ├── FEATURE_SELECTION_DECISION.md      ← Keep/drop criteria + choices
│   │   └── selected_features.txt              ← Final subset (e.g., 3-6 features)
│   │
│   ├── Day 5: Final Model & Handoff
│   │   ├── model_final.pkl                    ← Production model (27-30 features)
│   │   ├── predictions_final.csv              ← Final predictions
│   │   ├── feature_importance_final.csv       ← Top 20 features only
│   │   ├── EXECUTIVE_SUMMARY.md               ← One-pager
│   │   │
│   │   └── HANDOFF PACKAGE/
│   │       ├── README.md                      ← How to use new model
│   │       ├── feature_definitions.txt        ← What each feature means
│   │       ├── performance_benchmarks.csv     ← Before/after metrics
│   │       ├── model_deployment_checklist.md  ← Steps to deploy
│   │       └── monitoring_plan.md             ← Weekly/monthly QA
│   │
│   └── LOGS/
│       ├── day1_feature_eng.log
│       ├── day2_training.log
│       ├── day3_analysis.log
│       ├── day4_decision.log
│       └── day5_final.log
```

**Total Output Files:** ~25-30 files across 5 days

---

## SECTION VII: SUCCESS CRITERIA & GO/NO-GO DECISION

### Go Decision (Proceed with Production)
✓ **ALL of:**
- [ ] R² improves by ≥0.01 (conservative bound)
- [ ] RMSE improves or flat (not worse)
- [ ] No new overfitting detected (test ≥ 0.9 × train)
- [ ] Feature selection makes business sense
- [ ] Documentation complete

### No-Go Decision (Revert to 24-Feature Model)
✗ **ANY of:**
- [ ] R² same or worse (≤0.4380)
- [ ] RMSE significantly worse (>0.36)
- [ ] Overfitting detected (test R² drops >0.02 from train)
- [ ] Features uninterpretable or unstable
- [ ] New bugs/issues discovered

### Conditional Go (Keep & Monitor)
? **If:**
- [ ] Marginal improvement (R² +0.005 to +0.01)
- [ ] All features are stable
- [ ] Plan monitoring protocol (weekly checks for drift)

---

## SECTION VIII: RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **New features don't help** | 20% | High | Run Days 1-3 analysis before committing |
| **Overfitting** | 15% | High | Monitor train vs. test gap; use cross-validation |
| **Data quality issues** | 10% | Medium | Run QA checks (Day 1) before training |
| **Feature engineering bugs** | 10% | High | Code review + unit tests for each script |
| **Timeline slippage** | 25% | Low | Daily progress tracking + contingency |

**Mitigation Actions:**
- Daily check-ins (5 min) with execution team
- Code review before each training run
- Rollback plan if issues detected (revert to 24-feature model)

---

## NEXT STEPS (Before Starting Day 1)

- [ ] Confirm ET₀ data available (check `richmond_openmeteo_daily_extended.csv`)
- [ ] Assign scripts to developers (2 engineers recommended)
- [ ] Set up monitoring infrastructure (weekly model validation)
- [ ] Brief stakeholders on timeline & expected outcomes
- [ ] Reserve compute resources (model training takes ~5-10 min)

---

**Plan Status:** Ready for execution  
**Estimated Duration:** 5 calendar days (2-3 FTE days)  
**Expected Value:** +0.02 to +0.05 R² improvement  
**Risk Level:** Medium (features well-motivated, outcomes uncertain)  

**Approval:** [ ] Manager [ ] Data Science Lead [ ] Operations

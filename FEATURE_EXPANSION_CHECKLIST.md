# Feature Expansion: 5-Day Execution Checklist

**Project:** Add 15 new features (8 ET₀ + 7 day-of-week) to RRWWTF flow model  
**Timeline:** Days 1-5 (this week)  
**Owner:** Data Science Team  
**Expected Outcome:** +0.02 to +0.05 R² improvement  

---

## DAY 1: FEATURE ENGINEERING ✓

### Phase 1.1: ET₀ Features (3 hours)

- [ ] Load ET₀ data from `output/12_r2_ceiling_diagnostic/richmond_openmeteo_daily_extended.csv`
- [ ] Create 8 ET₀ features:
  - [ ] `et0_t` (raw daily value)
  - [ ] `et0_lag1`, `et0_lag3`, `et0_lag7` (shifted versions)
  - [ ] `et0_rolling_7d`, `et0_rolling_30d` (averages)
  - [ ] `et0_ante_5d` (5-day sum)
  - [ ] `et0_x_sm` (INTERACTION: ET₀ × soil_moisture_index) ⭐ **HIGH PRIORITY**
- [ ] Script: `scripts/05_add_et0_features.py`
- [ ] Output: `temp/et0_features.csv`
- [ ] QA Check:
  - [ ] All 8 features present, no NaN
  - [ ] Data types: float64
  - [ ] Date alignment correct
  - [ ] ET₀ × SM interaction makes sense (no extreme values)

### Phase 1.2: Day-of-Week Features (2 hours)

- [ ] Extract day-of-week from Date column
- [ ] Create 7 DOW features (one-hot encoding, Monday=baseline):
  - [ ] `dow_tuesday`, `dow_wednesday`, `dow_thursday` (mid-week)
  - [ ] `dow_friday` (end-of-week)
  - [ ] `dow_saturday`, `dow_sunday` (weekend)
  - [ ] `is_weekend` (aggregate: Sat or Sun)
- [ ] Script: `scripts/06_add_dow_features.py`
- [ ] Output: `temp/dow_features.csv`
- [ ] QA Check:
  - [ ] All 7 features present
  - [ ] Each row has exactly one day=1, rest=0
  - [ ] Binary values only (0 or 1)
  - [ ] No missing values

### Phase 1.3: Merge & Master Dataset (2 hours)

- [ ] Script: `scripts/07_merge_all_features.py`
- [ ] Merge ET₀ (8) + DOW (7) + existing (24) = 39 total
- [ ] Output: `output/00_master_dataset/richmond_master_with_all_features.csv`
- [ ] QA Checklist:
  - [ ] Rows: 2,192 ✓
  - [ ] Columns: 39 ✓
  - [ ] No NaN in new features
  - [ ] Date range: 2018-01-01 to 2024-08-31 ✓
  - [ ] Feature independence (no corr > 0.8)

**End of Day 1:** Master dataset ready with all 39 features

---

## DAY 2: MODEL TRAINING ✓

### Phase 2.1: Baseline Control Model (2 hours)

- [ ] Use: `scripts/train_with_soil_moisture.py` (existing 24-feature model)
- [ ] Purpose: Lock in current performance
- [ ] Record:
  - [ ] R² = 0.4380
  - [ ] RMSE = 0.3554 MGD
  - [ ] MAPE = 10.5%
  - [ ] Tolerance = 76.2%
- [ ] Output: `output/15_feature_expansion/control_baseline.csv`

### Phase 2.2: Expanded Feature Model (3 hours)

- [ ] Create: `scripts/train_with_expanded_features.py`
- [ ] Load: `richmond_master_with_all_features.csv` (39 features)
- [ ] Train:
  - [ ] Set `FEATURE_COLUMNS = [all 39 features]`
  - [ ] Use same hyperparameters as baseline
  - [ ] Train on 2018-2022 (1,690 days)
  - [ ] Test on 2024 (214 days)
- [ ] Outputs:
  - [ ] `model_expanded.pkl` (trained model)
  - [ ] `predictions_expanded.csv` (test predictions)
  - [ ] `feature_importance_expanded.csv` (all 39 ranked)

### Phase 2.3: Feature Stability QA (1 hour)

- [ ] Script: `scripts/check_feature_stability.py`
- [ ] For each new feature (8+7=15):
  - [ ] Mean, std, min, max computed ✓
  - [ ] No outliers beyond mean ± 5σ ✓
  - [ ] Variance > 0 (not constant) ✓
- [ ] Correlation check:
  - [ ] No feature pairs with r > 0.8 ✓
- [ ] Output: `stability_report.csv`

**End of Day 2:** Model trained, performance measured, 39 features ranked by importance

---

## DAY 3: ANALYSIS & INTERPRETATION ✓

### Phase 3.1: Performance Comparison (2 hours)

- [ ] Create: `scripts/generate_feature_expansion_report.py`
- [ ] Compare (24-feature baseline vs. 39-feature expanded):
  - [ ] R²: 0.4380 → 0.???? (target: 0.4580 to 0.4880)
  - [ ] RMSE: 0.3554 → 0.???? (target: 0.3354 to 0.3454)
  - [ ] MAPE: 10.5 → ???? (target: 9.5 to 10.0)
  - [ ] Tolerance: 76.2% → ????% (target: 77% to 78.5%)
- [ ] Output: `COMPARISON_REPORT.md`

### Phase 3.2: Feature Ranking Analysis (1 hour)

- [ ] Extract top 20 features from expanded model
- [ ] Identify new features in top 20:
  - [ ] `et0_x_sm` rank = __ (⭐ **SHOULD BE TOP 15**)
  - [ ] `et0_rolling_7d` rank = __
  - [ ] `is_weekend` rank = __
  - [ ] Other new features ranked = __
- [ ] Count: How many of 15 new features in top 30?
  - [ ] Expected: 3-8 features
- [ ] Output: `feature_importance_detailed.csv`

### Phase 3.3: Error Analysis (1 hour)

- [ ] Create plots (visual comparison):
  - [ ] `residuals_control_vs_expanded.png` (error distributions)
  - [ ] `error_by_flowrange_comparison.png` (low/medium/high flow accuracy)
  - [ ] `temporal_errors_comparison.png` (errors over time)
- [ ] Key questions:
  - [ ] Does expanded model reduce storm-tail errors (flow 2-7 days after rain)?
  - [ ] Does it maintain accuracy on dry days?
  - [ ] Any new systematic bias?

**End of Day 3:** Full performance comparison documented; understand which new features work

---

## DAY 4: DECISION & FEATURE SELECTION ✓

### Phase 4.1: ET₀ Features Decision (1 hour)

**Question:** Keep ET₀ features?

- [ ] Check: Is `et0_x_sm` in top 15? 
  - [ ] YES → Keep it (high-value interaction term)
  - [ ] NO → Evaluate for drop
- [ ] Check: Is `et0_rolling_7d` in top 25?
  - [ ] YES → Keep it (rolling average valuable)
  - [ ] NO → Evaluate for drop
- [ ] Check: Individual lags (`et0_lag1`, `lag3`, `lag7`) in top 25?
  - [ ] YES → Keep (surprising, review for redundancy)
  - [ ] NO → Drop (redundant with rolling features)

**Decision Logic:**
```
IF et0_x_sm in top 15 AND et0_rolling_7d in top 25:
   → KEEP BOTH (ET₀ features valuable)
   → DROP individual lags (redundant)
ELSE IF only one of above:
   → KEEP the valuable one
   → DROP others
ELSE:
   → DROP all ET₀ features (not predictive)
```

**Decision: [ ] KEEP ET₀  [ ] DROP ET₀  [ ] KEEP (partial: ____ only)**

### Phase 4.2: Day-of-Week Features Decision (1 hour)

**Question:** Keep DOW features?

- [ ] Check: Is `is_weekend` in top 20?
  - [ ] YES → Keep it (operational signal exists)
  - [ ] NO → Drop all (no daily pattern)
- [ ] Check: Individual day flags in top 20?
  - [ ] YES → Keep 1-2 (e.g., Friday) if interpretable
  - [ ] NO → Drop all (redundant)
- [ ] Verify: No overfitting (test ≈ train accuracy)?
  - [ ] YES → Safe to keep
  - [ ] NO → Drop (overfitting risk)

**Decision Logic:**
```
IF is_weekend in top 20 AND no overfitting:
   → KEEP is_weekend only (drop individual days)
   → Individual days are redundant
ELSE IF individual days in top 20:
   → KEEP top 2 (e.g., Friday, Saturday)
   → Rest redundant
ELSE:
   → DROP all DOW features
```

**Decision: [ ] KEEP DOW  [ ] DROP DOW  [ ] KEEP (partial: ____ only)**

### Phase 4.3: Final Feature Set Selection (1 hour)

- [ ] From Day 3 importance ranking, select features that passed criteria above
- [ ] Example selections (you'll choose based on your data):
  - [ ] **Must Keep (High Value):**
    - [ ] `et0_x_sm` (if in top 15) ⭐
  - [ ] **Should Keep (Medium Value):**
    - [ ] `et0_rolling_7d` (if in top 25)
    - [ ] `is_weekend` (if in top 20)
  - [ ] **Optional (Low Value but Interpretable):**
    - [ ] `dow_friday` (if has business meaning)
  - [ ] **Drop (Low Value + Low Interpretability):**
    - [ ] `et0_lag1`, `et0_lag3`, `et0_lag7` (if not top 20)
    - [ ] Individual day flags except Friday

**Final Count:** X new features selected (target: 3-6)
- Example: Keep 3 (`et0_x_sm`, `et0_rolling_7d`, `is_weekend`)
- Result: 24 + 3 = **27 features** for production model

- [ ] Document: `FEATURE_SELECTION_DECISION.md`
- [ ] List selected features: `selected_features.txt`

**End of Day 4:** Decided which 3-6 new features to use; justified each choice

---

## DAY 5: FINAL MODEL & HANDOFF ✓

### Phase 5.1: Train Final Production Model (2 hours)

- [ ] Create: `scripts/train_final_expanded_model.py`
- [ ] Use ONLY selected features from Day 4 (e.g., 27 features, not 39)
- [ ] Train on 2018-2022, test on 2024
- [ ] Record performance:
  - [ ] R² = ____ (target: 0.45-0.49)
  - [ ] RMSE = ____ (target: 0.33-0.35)
  - [ ] MAPE = ____ 
  - [ ] Tolerance = ____% (target: 76-78%)
- [ ] Outputs:
  - [ ] `model_final.pkl` (production model, ~27 features)
  - [ ] `predictions_final.csv` (test predictions)
  - [ ] `feature_importance_final.csv` (top 20 features only)

### Phase 5.2: Executive Summary (1 hour)

- [ ] Create: `EXECUTIVE_SUMMARY.md`
- [ ] Fill in:
  - [ ] R² improvement: +____ (calculated from baseline)
  - [ ] RMSE improvement: -____ MGD
  - [ ] Tolerance improvement: +____% 
  - [ ] Final recommendation: **KEEP / REJECT / MONITOR**

**Template:**
```
# Feature Expansion Results

## Performance
- **R² improved from 0.4380 to ____** (+____%)
- **RMSE improved from 0.3554 to ____** (-____ MGD)
- **Tolerance improved from 76.2% to ____%** (+___%)

## Recommendation
✓ **ACCEPT** - Model improved within targets. 
Selected 3-6 new features are interpretable and stable.

[OR]

✗ **REJECT** - No improvement. Return to 24-feature baseline.

[OR]

? **MONITOR** - Marginal improvement. Deploy with weekly checks.
```

### Phase 5.3: Handoff Package (1 hour)

Create documentation for deployment team:

- [ ] `README.md` - How to use new model
- [ ] `feature_definitions.txt` - What each feature means
  - Example:
    ```
    et0_x_sm = ET₀ × Soil Moisture interaction
    is_weekend = 1 if Saturday or Sunday, 0 otherwise
    et0_rolling_7d = 7-day rolling average of ET₀
    ```
- [ ] `performance_benchmarks.csv` - Before/after metrics
- [ ] `model_deployment_checklist.md` - Steps to deploy to production
- [ ] `monitoring_plan.md` - Weekly/monthly QA checks

### Phase 5.4: Final Sign-Off (30 min)

**Checklist before closing:**

- [ ] R² improved or maintained (no worse than 0.4380)
- [ ] RMSE stable or improved (no worse than 0.3554)
- [ ] Feature selection documented & justified
- [ ] No bugs or data quality issues
- [ ] Performance stable (no train/test divergence)
- [ ] All files saved to `output/15_feature_expansion/`

**Sign-Off By:**
- [ ] Data Science Lead: _______________
- [ ] Analytics Manager: _______________
- [ ] Operations: _______________

**End of Day 5:** Production model ready; handoff complete

---

## SUCCESS CRITERIA

### ✓ GO (Proceed to Production)
- R² ≥ 0.4480 (±0.01 improvement minimum)
- RMSE ≤ 0.3554 (flat or better)
- No overfitting detected
- 3-6 new features selected & justified

### ✗ NO-GO (Revert to Baseline)
- R² ≤ 0.4380 (no improvement)
- RMSE > 0.3654 (significantly worse)
- Overfitting detected (train >> test)
- Features unstable or uninterpretable

### ? CONDITIONAL (Keep & Monitor)
- R² gain: +0.005 to +0.01 (marginal)
- All features stable
- Deploy with weekly performance checks

---

## DAILY STANDUP TEMPLATE

**Use this at end of each day:**

```
DAY X COMPLETE

Features Created:    [___/15 done]
QA Passed:          [___/__ checks]
Key Finding:        [Summary of progress]
Blockers:           [None / ____]
Tomorrow's Plan:    [Next phase]

Next Handoff:       [Who gets the work]
```

---

## QUICK LINKS TO KEY FILES

**Created Today:**
- Feature engineering scripts: `scripts/05_*, 06_*, 07_*`
- Master dataset: `output/00_master_dataset/richmond_master_with_all_features.csv`
- Training scripts: `scripts/train_with_expanded_features.py`
- Analysis: `output/15_feature_expansion/COMPARISON_REPORT.md`
- Decision: `output/15_feature_expansion/FEATURE_SELECTION_DECISION.md`
- Final model: `output/15_feature_expansion/model_final.pkl`
- Handoff: `output/15_feature_expansion/EXECUTIVE_SUMMARY.md`

**Full Plan:** `FEATURE_EXPANSION_PLAN.md` (this directory)

---

## ESTIMATED TIME BREAKDOWN

| Day | Phase | Duration | Status |
|-----|-------|----------|--------|
| 1 | Feature engineering | 7 hours | [ ] TO DO |
| 2 | Model training + QA | 6 hours | [ ] TO DO |
| 3 | Analysis & interpretation | 4 hours | [ ] TO DO |
| 4 | Decision making | 3 hours | [ ] TO DO |
| 5 | Final model + handoff | 3.5 hours | [ ] TO DO |
| **Total** | | **23.5 hours** | |

**Assumes:** 2 FTE engineers working in parallel = **3-4 calendar days**

---

## START DATE & TIMELINE

- **Day 1 Start:** _______________
- **Target Completion:** _______________
- **Deployment Date:** _______________
- **Monitoring Start:** _______________

---

**Print this checklist. Check off as you go. Good luck! 🚀**

"""
RRWWTF Correlation Matrix and Multicollinearity Analysis (Companion to the
Lag-Structure Analysis)

Diagnostic analysis only -- NO modeling/prediction happens here. This is a
companion to scripts/correlation_analysis.py (the lag-structure/CCF/PACF
stage): where that script looks at flow and rainfall two at a time across
many lags, this script assembles the actual candidate feature set in one
frame -- flow lags, excess-over-baseline flow, rainfall lags, antecedent
wetness, calendar/seasonal terms, day-of-week -- and asks two different
questions of it: (1) which features correlate with the target strongly
enough to be worth keeping, both linearly (Pearson) and by rank
(Spearman, since rainfall is zero-inflated and its relationship to flow is
not linear), and (2) which candidate features are themselves redundant
with each other (pairwise |r| > 0.8, and VIF), since redundant predictors
add multicollinearity without adding information.

    1. Full Pearson correlation matrix, all engineered features + target,
       diverging colormap centered at zero.
    2. Full Spearman correlation matrix, same feature set -- flagged
       wherever it disagrees with Pearson by more than 0.15.
    3. Wet-days-only Pearson matrix (rainfall > 0.1 in, plus the 2 days
       following) -- the full-series matrix is dominated by dry days,
       which flattens every rainfall relationship toward zero.
    4. Sorted bar chart of each feature's Pearson correlation with the
       target.
    5. Multicollinearity flags: every feature PAIR (target excluded) with
       |r| > 0.8.
    6. VIF for every numeric predictor (target excluded), via manual OLS
       (statsmodels is not available in this environment) -- VIF_i =
       1 / (1 - R_i^2), R_i^2 from regressing feature i on every other
       feature.
    7. output/07_correlation_matrix_analysis/correlation_findings.md --
       narrative summary citing the specific numbers behind each claim.

GAP HANDLING (record is 2018-2024 with real gaps -- 2023 is Jun-Sep only,
2024 is ~242 days): every lag/rolling feature below is built on the
continuous calendar-day index from daily_excess_flow.csv (one row per
calendar day, gaps present as NaN rather than missing rows), so a "lag of
k days" is always a true k-calendar-day offset, never a k-row offset
across a gap. The full row-wise feature frame is then built with a single
.dropna() (no interpolation, no forward-fill) and the exact drop count is
reported, split out into (a) calendar days with no plant record at all and
(b) calendar days that DO have a plant record but were dropped anyway
because a lag/rolling window on that row reaches into a gap or into the
first 14 days of the record (the longest rolling window used).

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_dry_weather_baseline.csv
        Date, Total_Treated_MGD, Rainfall_in, Year, Month, Source_File,
        Source_Year, Dry_5Day, Expected_Baseline_MGD, Baseline_Source,
        Baseline_Dry_Day_Count
    output/05_rdii_events/daily_excess_flow.csv
        Date, Total_Treated_MGD, Expected_Baseline_MGD, Excess_Flow_MGD,
        RDII_MGD, Row_Exists, Dry_5Day

NOTE ON SCOPE DEVIATIONS FROM THE ORIGINAL REQUEST:
  - No temperature column exists anywhere in this project's data (the
    master dataset only carries plant-recorded Rainfall_in; see
    rainfall_analysis.py, which states no external weather data is
    blended in). The temperature feature is therefore omitted, not
    fabricated -- this is called out again in the console output and in
    correlation_findings.md.
  - The master dataset has a single flow column, Total_Treated_MGD
    ("total treated flow"), with no separate influent/effluent split.
    That single column is used as the target ("flow") throughout.
  - output/05_correlation_analysis/ was requested as the output directory,
    but "05" is already used twice over in this project's numbered
    sequence: output/05_rdii_events/ (existing pipeline stage) and
    output/06_correlation_analysis/ (the lag-structure companion script,
    added earlier in this same body of work). This script therefore
    writes to output/07_correlation_matrix_analysis/ to keep the existing
    sequence intact and keep the two correlation-themed stages
    distinguishable by name.

Usage:
    python scripts/correlation_matrix.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

MASTER_BASELINE_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_dry_weather_baseline.csv"
DAILY_EXCESS_CSV = OUTPUT_DIR / "05_rdii_events" / "daily_excess_flow.csv"

ANALYSIS_DIR = OUTPUT_DIR / "07_correlation_matrix_analysis"

PEARSON_CSV = ANALYSIS_DIR / "correlation_matrix.csv"
SPEARMAN_CSV = ANALYSIS_DIR / "correlation_matrix_spearman.csv"
WET_DAYS_PEARSON_CSV = ANALYSIS_DIR / "correlation_matrix_wet_days_only.csv"
TARGET_CORR_CSV = ANALYSIS_DIR / "target_correlation_ranked.csv"
MULTICOLLINEARITY_CSV = ANALYSIS_DIR / "multicollinearity_flags.csv"
VIF_CSV = ANALYSIS_DIR / "vif.csv"
FINDINGS_MD = ANALYSIS_DIR / "correlation_findings.md"

FIG1_PEARSON = ANALYSIS_DIR / "01_pearson_correlation_matrix.png"
FIG2_SPEARMAN = ANALYSIS_DIR / "02_spearman_correlation_matrix.png"
FIG3_WET_DAYS = ANALYSIS_DIR / "03_wet_days_only_pearson_matrix.png"
FIG4_TARGET_BAR = ANALYSIS_DIR / "04_target_correlation_ranked.png"

TARGET_COL = "flow"
WET_DAY_RAIN_THRESHOLD_IN = 0.1
WET_DAY_TAIL_DAYS = 2
AW_WINDOWS = [3, 5, 7, 14]
MULTICOLLINEARITY_THRESHOLD = 0.8
SPEARMAN_PEARSON_DISAGREEMENT_THRESHOLD = 0.15
VIF_HIGH_THRESHOLD = 10.0
VIF_MODERATE_THRESHOLD = 5.0

# Reference day-of-week dropped from the dummy set (Sunday) to avoid the
# dummy-variable trap -- six dummies fully represent seven days once an
# intercept/other correlations are present.
DAY_NAMES_ALL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_REFERENCE = "Sunday"

FLOW_COLOR = "#2b6cb0"
POS_COLOR = "#c0392b"
NEG_COLOR = "#2b6cb0"
INCOMPLETE_COLOR = "#9ca3af"

DPI = 300

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Load + merge onto a single continuous calendar-day series
# ---------------------------------------------------------------------------

def load_ts() -> pd.DataFrame:
    master = pd.read_csv(MASTER_BASELINE_CSV, parse_dates=["Date"])
    excess = pd.read_csv(DAILY_EXCESS_CSV, parse_dates=["Date"])

    merged = excess.merge(master[["Date", "Rainfall_in"]], on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    ts = merged.set_index("Date")
    gaps = ts.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=1)], (
        "Expected one row per calendar day (gaps present as NaN, not missing rows) -- "
        "found irregular row spacing, which would break every lag/rolling feature below."
    )
    return ts


# ---------------------------------------------------------------------------
# Feature frame assembly
# ---------------------------------------------------------------------------

def build_feature_frame(ts: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=ts.index)

    feat[TARGET_COL] = ts["Total_Treated_MGD"]
    feat["flow_lag1"] = ts["Total_Treated_MGD"].shift(1)
    feat["flow_lag2"] = ts["Total_Treated_MGD"].shift(2)
    feat["flow_lag3"] = ts["Total_Treated_MGD"].shift(3)

    feat["excess_over_baseline"] = ts["Excess_Flow_MGD"]
    feat["excess_lag1"] = ts["Excess_Flow_MGD"].shift(1)

    feat["rainfall"] = ts["Rainfall_in"]
    feat["rain_lag1"] = ts["Rainfall_in"].shift(1)
    feat["rain_lag2"] = ts["Rainfall_in"].shift(2)
    feat["rain_lag3"] = ts["Rainfall_in"].shift(3)

    for w in AW_WINDOWS:
        feat[f"antecedent_wetness_{w}d"] = ts["Rainfall_in"].rolling(w, min_periods=w).sum()

    # No temperature column exists anywhere in this project's data (see module
    # docstring) -- intentionally omitted rather than fabricated.

    feat["month"] = ts.index.month
    doy = ts.index.dayofyear
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    dow = ts.index.dayofweek  # 0 = Monday .. 6 = Sunday
    feat["is_weekend"] = (dow >= 5).astype(int)
    for i, name in enumerate(DAY_NAMES_ALL):
        if name == DAY_REFERENCE:
            continue
        feat[f"dow_{name.lower()}"] = (dow == i).astype(int)

    return feat


def drop_incomplete_rows(ts: pd.DataFrame, feat: pd.DataFrame):
    dropped_mask = feat.isna().any(axis=1)
    complete = feat.loc[~dropped_mask].copy()

    n_total = len(feat)
    n_dropped_no_own_record = int((dropped_mask & ~ts["Row_Exists"]).sum())
    n_dropped_with_own_record = int((dropped_mask & ts["Row_Exists"]).sum())
    n_complete = len(complete)

    report = {
        "n_total_calendar_days": n_total,
        "n_complete_rows": n_complete,
        "n_dropped_total": int(dropped_mask.sum()),
        "n_dropped_no_own_plant_record": n_dropped_no_own_record,
        "n_dropped_own_record_present_but_lag_or_rolling_incomplete": n_dropped_with_own_record,
    }
    return complete, report


# ---------------------------------------------------------------------------
# Wet-day conditioning (Task 3) -- computed on the continuous calendar
# index BEFORE the row-wise dropna, then intersected with the complete-case
# rows, so the +/-2-day tail is a true calendar offset even across gaps.
# ---------------------------------------------------------------------------

def build_wet_day_mask(ts: pd.DataFrame, tail_days: int) -> pd.Series:
    wet = ts["Rainfall_in"] > WET_DAY_RAIN_THRESHOLD_IN
    wet = wet.fillna(False)
    mask = wet.copy()
    for k in range(1, tail_days + 1):
        mask = mask | wet.shift(k).fillna(False)
    return mask


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------

def compute_pearson(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def compute_spearman(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="spearman")


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, path: Path):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(15, 0.85 * n), max(12, 0.75 * n)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Correlation")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_target_correlation_bar(target_corr: pd.Series, path: Path):
    ordered = target_corr.drop(TARGET_COL).sort_values()
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in ordered.values]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.32 * len(ordered))))
    ax.barh(ordered.index, ordered.values, color=colors, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    for y, (name, val) in enumerate(ordered.items()):
        ax.annotate(f"{val:.3f}", xy=(val, y), xytext=(4 if val >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8)

    ax.set_xlabel("Pearson Correlation with Target (flow)")
    ax.set_title(f"Feature Correlation with Target – Richmond RRWWTF (n={len(ordered)} features)")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multicollinearity flags
# ---------------------------------------------------------------------------

def find_multicollinear_pairs(pearson: pd.DataFrame, threshold: float) -> pd.DataFrame:
    predictors = [c for c in pearson.columns if c != TARGET_COL]
    sub = pearson.loc[predictors, predictors]
    rows = []
    for i, a in enumerate(predictors):
        for b in predictors[i + 1:]:
            r = sub.loc[a, b]
            if pd.notna(r) and abs(r) > threshold:
                rows.append({"Feature_A": a, "Feature_B": b, "Correlation": r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Abs_Correlation"] = out["Correlation"].abs()
        out = out.sort_values("Abs_Correlation", ascending=False).drop(columns="Abs_Correlation")
        out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# VIF (manual OLS -- statsmodels not available in this environment)
# ---------------------------------------------------------------------------

def compute_vif(predictors: pd.DataFrame) -> pd.DataFrame:
    cols = predictors.columns.tolist()
    n = len(predictors)
    rows = []
    for i, col in enumerate(cols):
        y = predictors[col].values.astype(float)
        other_cols = [c for c in cols if c != col]
        X_others = np.column_stack([np.ones(n)] + [predictors[c].values.astype(float) for c in other_cols])
        beta, _, _, _ = np.linalg.lstsq(X_others, y, rcond=None)
        y_hat = X_others @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = np.inf if r2 >= 1 - 1e-10 else 1 / (1 - r2)
        rows.append({"Feature": col, "R_Squared_vs_Other_Features": r2, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pearson vs. Spearman disagreement
# ---------------------------------------------------------------------------

def find_pearson_spearman_disagreements(pearson: pd.DataFrame, spearman: pd.DataFrame,
                                         threshold: float) -> pd.DataFrame:
    cols = pearson.columns.tolist()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if pd.isna(p) or pd.isna(s):
                continue
            diff = s - p
            if abs(diff) > threshold:
                rows.append({"Feature_A": a, "Feature_B": b, "Pearson": p, "Spearman": s,
                             "Spearman_Minus_Pearson": diff})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Abs_Diff"] = out["Spearman_Minus_Pearson"].abs()
        out = out.sort_values("Abs_Diff", ascending=False).drop(columns="Abs_Diff").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------

def write_findings_md(complete_report, wet_n, target_corr_full, target_corr_wet,
                       disagreements, multicollinear_pairs, vif_df):
    ordered = target_corr_full.drop(TARGET_COL).sort_values(key=lambda s: s.abs(), ascending=False)
    top5 = ordered.head(5)

    lines = []
    lines.append("# Correlation Matrix Findings – Richmond RRWWTF Flow Prediction")
    lines.append("")
    lines.append("Diagnostic output of `scripts/correlation_matrix.py`, a companion to the "
                  "lag-structure analysis (`scripts/correlation_analysis.py`). Every number below "
                  "comes directly from the accompanying CSVs.")
    lines.append("")

    lines.append("## Rows each matrix was computed on")
    lines.append("")
    lines.append(
        f"- Full-series matrices (Pearson, Spearman, VIF): **{complete_report['n_complete_rows']:,} "
        f"complete rows** out of {complete_report['n_total_calendar_days']:,} total calendar days "
        f"({complete_report['n_dropped_total']:,} dropped, no interpolation or fill)."
    )
    lines.append(
        f"  - {complete_report['n_dropped_no_own_plant_record']:,} dropped because the day itself has "
        f"no plant record at all (2023 Jan-May/Oct-Dec, 2024 tail, and other isolated gaps)."
    )
    lines.append(
        f"  - {complete_report['n_dropped_own_record_present_but_lag_or_rolling_incomplete']:,} dropped "
        f"despite having their own plant record, because a lag or rolling-window feature on that row "
        f"reaches into a gap or into the first 14 days of the record (the longest window used, "
        f"antecedent_wetness_14d)."
    )
    lines.append(f"- Wet-days-only matrix (rainfall > {WET_DAY_RAIN_THRESHOLD_IN} in, plus "
                  f"{WET_DAY_TAIL_DAYS} days following): **{wet_n:,} rows**, a subset of the complete rows above.")
    lines.append("")

    lines.append("## Features most strongly related to the target (flow)")
    lines.append("")
    for name, val in top5.items():
        lines.append(f"- `{name}`: r = {val:+.3f}")
    lines.append("")
    lines.append(
        f"`{top5.index[0]}` is the single strongest correlate of flow (r={top5.iloc[0]:+.3f}); note that "
        f"`excess_over_baseline` and `excess_lag1` are correlated with flow largely because they are "
        f"*derived* from flow (flow minus the dry-weather baseline), not because they carry new "
        f"predictive information -- see the multicollinearity flags below."
    )
    lines.append("")

    lines.append("## Pearson vs. Spearman disagreement (|Spearman - Pearson| > "
                  f"{SPEARMAN_PEARSON_DISAGREEMENT_THRESHOLD})")
    lines.append("")
    if disagreements.empty:
        lines.append(f"- No feature pair disagreed by more than {SPEARMAN_PEARSON_DISAGREEMENT_THRESHOLD} "
                      f"between the two methods -- the linear (Pearson) correlations reported above are not "
                      f"masking a materially different rank relationship.")
    else:
        for _, row in disagreements.iterrows():
            lines.append(f"- `{row['Feature_A']}` vs `{row['Feature_B']}`: Pearson={row['Pearson']:+.3f}, "
                          f"Spearman={row['Spearman']:+.3f} (diff {row['Spearman_Minus_Pearson']:+.3f})")
    lines.append("")

    lines.append("## How the wet-day subset changes the rainfall picture")
    lines.append("")
    rain_full = target_corr_full.get("rainfall", np.nan)
    rain_wet = target_corr_wet.get("rainfall", np.nan)
    if pd.notna(rain_full) and pd.notna(rain_wet):
        lines.append(
            f"- `rainfall` vs `flow`: r={rain_full:+.3f} on the full series vs r={rain_wet:+.3f} on the "
            f"wet-days-only subset ({wet_n:,} rows) -- "
            + (f"the wet-day-conditioned relationship is "
               f"{'substantially stronger' if abs(rain_wet) > abs(rain_full) * 1.2 else 'similar in strength'}, "
               f"confirming the full-series correlation is diluted by the many dry days with rainfall = 0."
               if abs(rain_wet) >= abs(rain_full) else
               "the wet-day-conditioned relationship is actually weaker/similar, suggesting rainfall's "
               "full-series correlation with flow is not simply being suppressed by dry-day dilution.")
        )
    for w in AW_WINDOWS:
        col = f"antecedent_wetness_{w}d"
        full_v = target_corr_full.get(col, np.nan)
        wet_v = target_corr_wet.get(col, np.nan)
        if pd.notna(full_v) and pd.notna(wet_v):
            lines.append(f"- `{col}` vs `flow`: r={full_v:+.3f} full-series vs r={wet_v:+.3f} wet-days-only.")
    lines.append("")

    lines.append(f"## Multicollinearity flags (predictor pairs with |r| > {MULTICOLLINEARITY_THRESHOLD})")
    lines.append("")
    if multicollinear_pairs.empty:
        lines.append(f"- No predictor pair exceeded |r| > {MULTICOLLINEARITY_THRESHOLD}.")
    else:
        for _, row in multicollinear_pairs.iterrows():
            lines.append(f"- `{row['Feature_A']}` & `{row['Feature_B']}`: r={row['Correlation']:+.3f} -- "
                          f"redundant; keep at most one of the pair as a model feature.")
    lines.append("")

    lines.append(f"## VIF (predictors only, target excluded)")
    lines.append("")
    infinite_vif = vif_df[np.isinf(vif_df["VIF"])]
    finite_high_vif = vif_df[(vif_df["VIF"] > VIF_HIGH_THRESHOLD) & np.isfinite(vif_df["VIF"])]
    moderate_vif = vif_df[(vif_df["VIF"] > VIF_MODERATE_THRESHOLD) & (vif_df["VIF"] <= VIF_HIGH_THRESHOLD)]

    if not infinite_vif.empty:
        inf_features = set(infinite_vif["Feature"])
        lines.append(
            f"- **{len(infinite_vif)} feature(s) have infinite VIF** -- an *exact* linear dependency, not "
            f"just a strong one, and notably **none of the pairs involved cross the |r| > "
            f"{MULTICOLLINEARITY_THRESHOLD} pairwise-correlation threshold above** (max pairwise r within "
            f"either cluster is well under 0.8) -- this is exactly why a multivariate VIF check is run in "
            f"addition to pairwise correlation: pairwise |r| cannot see a redundancy that only appears "
            f"across three or more columns at once."
        )
        if {"rainfall", "rain_lag1", "rain_lag2", "antecedent_wetness_3d"} <= inf_features:
            lines.append(
                "  - `rainfall`, `rain_lag1`, `rain_lag2`, `antecedent_wetness_3d`: infinite VIF because "
                "`antecedent_wetness_3d` is *defined* as the 3-day rolling sum of rainfall -- i.e. "
                "`antecedent_wetness_3d = rainfall + rain_lag1 + rain_lag2` exactly, by construction, not "
                "by coincidence. These 4 columns span only a 3-dimensional space. Keep the 3 raw lags OR "
                "the 3-day antecedent-wetness window, never both."
            )
        dow_cluster = {"is_weekend", "dow_monday", "dow_tuesday", "dow_wednesday", "dow_thursday", "dow_friday"}
        if dow_cluster <= inf_features:
            lines.append(
                "  - `is_weekend`, `dow_monday`, `dow_tuesday`, `dow_wednesday`, `dow_thursday`, "
                "`dow_friday`: infinite VIF because `is_weekend` is an exact linear function of the 5 "
                f"weekday dummies once `dow_{DAY_REFERENCE.lower()}` is dropped as the reference "
                "(`dow_saturday` cancels out of that identity algebraically, which is why it alone keeps a "
                "finite VIF). `is_weekend` and the day-of-week dummy set encode the same underlying "
                "day-of-week categorical at two different resolutions -- keep the 6 `dow_*` dummies "
                "(finer-grained) and drop `is_weekend`, or the reverse, never both."
            )
    lines.append(f"- {len(finite_high_vif)} additional feature(s) with finite VIF > {VIF_HIGH_THRESHOLD:g}: "
                 + (", ".join(f"`{r.Feature}` (VIF={r.VIF:.1f})" for r in finite_high_vif.itertuples())
                    if not finite_high_vif.empty else "none."))
    lines.append(f"- {len(moderate_vif)} feature(s) with {VIF_MODERATE_THRESHOLD:g} < VIF <= "
                 f"{VIF_HIGH_THRESHOLD:g} (moderate): "
                 + (", ".join(f"`{r.Feature}` (VIF={r.VIF:.1f})" for r in moderate_vif.itertuples())
                    if not moderate_vif.empty else "none."))
    lines.append("- Full VIF table: see `vif.csv`.")
    lines.append("")

    lines.append("## Omitted from this analysis")
    lines.append("")
    lines.append("- **Temperature**: no temperature column exists anywhere in this project's data -- "
                  "only plant-recorded `Rainfall_in` is tracked, and `rainfall_analysis.py` explicitly "
                  "documents that no external weather data is blended into the RRWWTF dataset. Omitted "
                  "rather than fabricated.")
    lines.append("- **Separate weekday/weekend columns**: only `is_weekend` is included (weekday would be "
                  "its perfect inverse and add nothing).")
    lines.append(f"- **`dow_{DAY_REFERENCE.lower()}`**: dropped as the day-of-week reference category to "
                  f"avoid the dummy-variable trap; the six included `dow_*` columns fully represent the week.")
    lines.append("")

    FINDINGS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ts = load_ts()
    feat = build_feature_frame(ts)
    complete, drop_report = drop_incomplete_rows(ts, feat)

    print("CORRELATION MATRIX AND MULTICOLLINEARITY ANALYSIS")
    print()
    print(f"Feature frame: {len(feat.columns)} columns "
          f"({len(feat.columns) - 1} predictors + target '{TARGET_COL}')")
    print(f"Total calendar days: {drop_report['n_total_calendar_days']:,}")
    print(f"Complete rows (no NaN in any feature): {drop_report['n_complete_rows']:,}")
    print(f"Dropped: {drop_report['n_dropped_total']:,} "
          f"({drop_report['n_dropped_no_own_plant_record']:,} no plant record that day, "
          f"{drop_report['n_dropped_own_record_present_but_lag_or_rolling_incomplete']:,} "
          f"own record present but lag/rolling context incomplete)")
    print()

    # --- Task 1: full Pearson ---
    pearson = compute_pearson(complete)
    pearson.to_csv(PEARSON_CSV)
    plot_correlation_heatmap(pearson, "Full Pearson Correlation Matrix – Richmond RRWWTF", FIG1_PEARSON)

    target_corr_full = pearson[TARGET_COL]
    target_corr_full.sort_values(key=lambda s: s.abs(), ascending=False).to_csv(TARGET_CORR_CSV)

    print("TASK 1 -- Top 8 |Pearson correlation| with target (flow):")
    ranked = target_corr_full.drop(TARGET_COL).reindex(
        target_corr_full.drop(TARGET_COL).abs().sort_values(ascending=False).index)
    for name, val in ranked.head(8).items():
        print(f"    {name:<28s} {val:+.3f}")
    print()

    # --- Task 2: full Spearman ---
    spearman = compute_spearman(complete)
    spearman.to_csv(SPEARMAN_CSV)
    plot_correlation_heatmap(spearman, "Full Spearman Correlation Matrix – Richmond RRWWTF", FIG2_SPEARMAN)

    disagreements = find_pearson_spearman_disagreements(pearson, spearman,
                                                          SPEARMAN_PEARSON_DISAGREEMENT_THRESHOLD)
    print(f"TASK 2 -- Pearson vs. Spearman: {len(disagreements)} pair(s) disagree by more than "
          f"{SPEARMAN_PEARSON_DISAGREEMENT_THRESHOLD}")
    if not disagreements.empty:
        print(disagreements.to_string(index=False))
    print()

    # --- Task 3: wet-days-only Pearson ---
    wet_mask_full_calendar = build_wet_day_mask(ts, WET_DAY_TAIL_DAYS)
    wet_complete = complete.loc[complete.index.intersection(wet_mask_full_calendar[wet_mask_full_calendar].index)]
    wet_pearson = compute_pearson(wet_complete)
    wet_pearson.to_csv(WET_DAYS_PEARSON_CSV)
    plot_correlation_heatmap(
        wet_pearson,
        f"Wet-Days-Only Pearson Correlation Matrix – Richmond RRWWTF\n"
        f"(rainfall > {WET_DAY_RAIN_THRESHOLD_IN} in + {WET_DAY_TAIL_DAYS}-day tail, n={len(wet_complete):,})",
        FIG3_WET_DAYS)

    target_corr_wet = wet_pearson[TARGET_COL]
    print(f"TASK 3 -- Wet-days-only matrix computed on {len(wet_complete):,} rows "
          f"(of {len(complete):,} complete rows)")
    print(f"    rainfall vs flow: full={target_corr_full['rainfall']:+.3f}  "
          f"wet-only={target_corr_wet.get('rainfall', float('nan')):+.3f}")
    print()

    # --- Task 4: sorted bar chart ---
    plot_target_correlation_bar(target_corr_full, FIG4_TARGET_BAR)

    # --- Multicollinearity flags ---
    multicollinear_pairs = find_multicollinear_pairs(pearson, MULTICOLLINEARITY_THRESHOLD)
    multicollinear_pairs.to_csv(MULTICOLLINEARITY_CSV, index=False)
    print(f"MULTICOLLINEARITY FLAGS -- {len(multicollinear_pairs)} predictor pair(s) with "
          f"|r| > {MULTICOLLINEARITY_THRESHOLD}")
    if not multicollinear_pairs.empty:
        print(multicollinear_pairs.to_string(index=False))
    print()

    # --- VIF ---
    predictors = complete.drop(columns=[TARGET_COL])
    vif_df = compute_vif(predictors)
    vif_df.to_csv(VIF_CSV, index=False)
    print("VIF (predictors only, target excluded), highest first:")
    print(vif_df.to_string(index=False))
    n_inf = int(np.isinf(vif_df["VIF"]).sum())
    if n_inf:
        print(f"  NOTE: {n_inf} feature(s) show infinite VIF -- an EXACT linear dependency (not just a "
              f"strong one), none of which cross the |r| > {MULTICOLLINEARITY_THRESHOLD} pairwise flag "
              f"above. This is the multivariate redundancy pairwise correlation can't see: "
              f"antecedent_wetness_3d = rainfall + rain_lag1 + rain_lag2 by construction, and is_weekend "
              f"is an exact linear function of the 5 weekday dummies given dow_{DAY_REFERENCE.lower()} is "
              f"the dropped reference. See correlation_findings.md for the full explanation.")
    print()

    # --- Findings markdown ---
    write_findings_md(drop_report, len(wet_complete), target_corr_full, target_corr_wet,
                       disagreements, multicollinear_pairs, vif_df)

    print("CORRELATION MATRIX ANALYSIS COMPLETE")
    print()
    print("Generated files:")
    for p in [PEARSON_CSV, SPEARMAN_CSV, WET_DAYS_PEARSON_CSV, TARGET_CORR_CSV, MULTICOLLINEARITY_CSV,
              VIF_CSV, FINDINGS_MD, FIG1_PEARSON, FIG2_SPEARMAN, FIG3_WET_DAYS, FIG4_TARGET_BAR]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

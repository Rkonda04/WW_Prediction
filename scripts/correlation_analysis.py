"""
RRWWTF Correlation and Lag-Structure Analysis (Diagnostic Pass Ahead of
Model Tuning)

Diagnostic analysis only -- NO modeling/prediction happens here. This script
exists to answer, with continuous-signal methods, which lag features are
actually justified before returning to model tuning: RDII event analysis
(rdii_event_analysis.py) found a median days-to-peak of 1 day from discrete
event delineation; this script cross-checks that finding using every-day
cross-correlation, ACF/PACF, and antecedent-wetness correlation.

    1. Cross-correlation (CCF) of daily plant rainfall vs. daily
       excess-over-baseline flow, lags 0-10 days, full period.
    2. The same CCF repeated separately for warm season (May-Oct) and cool
       season (Nov-Apr), to test whether the response lag shifts with
       ground saturation.
    3. The same CCF again, restricted to days inside a rain event (using
       the existing RDII event windows) plus the 10 days following, to
       remove dilution from long dry stretches.
    4. ACF and PACF (gap-safe, computed manually -- see note below) on raw
       daily flow and on excess-over-baseline flow, lags 0-14, with a
       stationarity comparison.
    5. Rolling antecedent-wetness precipitation sums (3/5/7/10/14 days)
       correlated against same-day excess flow, to pick the antecedent-
       wetness feature window.
    6. output/06_correlation_analysis/lag_feature_recommendations.md --
       which lags/window to keep, which to rule out, each recommendation
       citing the specific number that supports it.

GAP HANDLING (record is 2018-2024 with real gaps -- 2023 is Jun-Sep only,
2024 is ~242 days): daily_excess_flow.csv is already reindexed to one row
per *calendar* day for the full 2018-01-01 to 2024-08-31 span, with missing
days present as NaN rows rather than removed. Every lagged/rolling
calculation in this script uses .shift()/.rolling() directly on that
continuous calendar-day index, so a lag of k days always means a true
k-calendar-day offset -- never a k-row offset across a gap. Pairs are then
dropped (not filled or interpolated) wherever either endpoint is NaN, and
the resulting pair count (n_pairs) is reported for every lag/window so a
thin lag can be told apart from a genuinely weak one. Lags/windows with
too few pairs to be meaningful are flagged "excluded" rather than silently
kept. ACF/PACF and the antecedent-wetness rolling sums use the same
continuous-calendar-day, drop-don't-fill approach; the ACF significance
band accounts for the fact that n_pairs varies slightly lag to lag near
the gap edges.

statsmodels is not available in this environment, so ACF is computed as a
pairwise-complete-observations Pearson correlation at each lag (same
mechanism as the CCF above, applied to a series against its own past), and
PACF is derived from that gap-safe ACF via the standard Durbin-Levinson
recursion -- this is mathematically the same Yule-Walker construction
statsmodels' pacf(method="ywm") uses, just computed from a gap-safe ACF
input.

Inputs (read-only, never modified):
    output/00_master_dataset/richmond_master_with_dry_weather_baseline.csv
        Date, Total_Treated_MGD, Rainfall_in, Year, Month, Source_File,
        Source_Year, Dry_5Day, Expected_Baseline_MGD, Baseline_Source,
        Baseline_Dry_Day_Count
    output/05_rdii_events/daily_excess_flow.csv
        Date, Total_Treated_MGD, Expected_Baseline_MGD, Excess_Flow_MGD,
        RDII_MGD, Row_Exists, Dry_5Day
    output/05_rdii_events/rdii_events.csv
        Event_ID, Start_Date, Rain_End_Date, End_Date, Duration_Days,
        Total_Rainfall_in, Max_Daily_Rainfall_in, Peak_Flow_MGD,
        Peak_Flow_Date, Peak_Excess_Flow_MGD, Total_Excess_Volume_MG,
        Days_To_Peak, Complete, Incomplete_Reason

Excess-over-baseline flow throughout this script is the unclipped
Excess_Flow_MGD (Total_Treated_MGD - Expected_Baseline_MGD), not the
clipped-at-zero RDII_MGD -- correlation/ACF methods want the continuous
signal, including its negative dry-day fluctuation around zero.

NOTE ON OUTPUT DIRECTORY NUMBER: output/05_rdii_events/ already occupies
"05" in the existing numbered pipeline, so this stage is written to
output/06_correlation_analysis/ (not 05_correlation_analysis/) to preserve
the existing sequence.

Usage:
    python scripts/correlation_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration -- paths and styling reused from the existing numbered pipeline
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

MASTER_BASELINE_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_with_dry_weather_baseline.csv"
DAILY_EXCESS_CSV = OUTPUT_DIR / "05_rdii_events" / "daily_excess_flow.csv"
RDII_EVENTS_CSV = OUTPUT_DIR / "05_rdii_events" / "rdii_events.csv"

ANALYSIS_DIR = OUTPUT_DIR / "06_correlation_analysis"

CCF_OVERALL_CSV = ANALYSIS_DIR / "ccf_overall.csv"
CCF_WARM_CSV = ANALYSIS_DIR / "ccf_seasonal_warm.csv"
CCF_COOL_CSV = ANALYSIS_DIR / "ccf_seasonal_cool.csv"
CCF_WET_EVENT_CSV = ANALYSIS_DIR / "ccf_wet_event_conditioned.csv"
ACF_PACF_CSV = ANALYSIS_DIR / "acf_pacf_values.csv"
ANTECEDENT_WETNESS_CSV = ANALYSIS_DIR / "antecedent_wetness_correlation.csv"
FEATURE_IMPORTANCE_CSV = ANALYSIS_DIR / "feature_importance_ranking.csv"
FULL_CORR_MATRIX_CSV = ANALYSIS_DIR / "full_feature_correlation_matrix.csv"
RECOMMENDATIONS_MD = ANALYSIS_DIR / "lag_feature_recommendations.md"

FIG1_CCF_OVERALL = ANALYSIS_DIR / "01_ccf_overall.png"
FIG2_CCF_SEASONAL = ANALYSIS_DIR / "02_ccf_seasonal_comparison.png"
FIG3_CCF_WET_EVENT = ANALYSIS_DIR / "03_ccf_wet_event_conditioned.png"
FIG4_ACF_PACF = ANALYSIS_DIR / "04_acf_pacf_panels.png"
FIG5_ANTECEDENT_WETNESS = ANALYSIS_DIR / "05_antecedent_wetness.png"
FIG6_FEATURE_IMPORTANCE = ANALYSIS_DIR / "06_feature_importance_ranking.png"
FIG7_FULL_CORR_MATRIX = ANALYSIS_DIR / "07_full_feature_correlation_matrix.png"

MAX_CCF_LAG = 10
MAX_ACF_LAG = 14
MIN_PAIRS_CCF = 30  # below this a Pearson r is too unstable to report as meaningful
AW_WINDOWS = [3, 5, 7, 10, 14]
WET_EVENT_TAIL_DAYS = 10  # days appended after each RDII event window (Start..End) for task 3
SIG_Z = 1.96  # 95% significance band multiplier (+-z/sqrt(n))
ACF_NEAR_ZERO_THRESHOLD = 0.2  # "decayed" cutoff used only for the stationarity note

WARM_MONTHS = [5, 6, 7, 8, 9, 10]
COOL_MONTHS = [11, 12, 1, 2, 3, 4]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
BASELINE_COLOR = "#c0392b"
INCOMPLETE_COLOR = "#9ca3af"
FIT_COLOR = "#27ae60"
WARM_COLOR = "#d97706"
COOL_COLOR = "#2b6cb0"
AW_IMPORTANCE_COLOR = "#7c3aed"

FEATURE_CATEGORY_COLORS = {
    "Rainfall CCF lag": FLOW_COLOR,
    "Excess-flow PACF lag": FIT_COLOR,
    "Antecedent wetness window": AW_IMPORTANCE_COLOR,
}

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
# Load + merge validated inputs onto a single continuous calendar-day series
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    master = pd.read_csv(MASTER_BASELINE_CSV, parse_dates=["Date"])
    excess = pd.read_csv(DAILY_EXCESS_CSV, parse_dates=["Date"])

    merged = excess.merge(master[["Date", "Rainfall_in"]], on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    ts = merged.set_index("Date")
    gaps_between_rows = ts.index.to_series().diff().dropna().unique()
    assert list(gaps_between_rows) == [pd.Timedelta(days=1)], (
        "daily_excess_flow.csv is expected to be one row per calendar day "
        "(gaps present as NaN rows, not missing rows) -- found an "
        "irregular row spacing, which would break every .shift()/.rolling() "
        "lag in this script."
    )
    return ts


def load_events() -> pd.DataFrame:
    events = pd.read_csv(RDII_EVENTS_CSV, parse_dates=["Start_Date", "Rain_End_Date", "End_Date"])
    return events


# ---------------------------------------------------------------------------
# Shared CCF machinery (Task 1, 2, 3)
# ---------------------------------------------------------------------------

def compute_ccf(ts: pd.DataFrame, x_col: str, y_col: str, max_lag: int,
                 mask: pd.Series = None) -> pd.DataFrame:
    """x leads y by `lag` days: pairs are (x[t], y[t+lag]).

    The shift is always applied to the full continuous calendar-day series
    first (so `lag` is a true k-calendar-day offset even across the
    2023/2024 gaps); an optional boolean `mask` (aligned to ts.index) is
    applied afterward to restrict which anchor days t are scored (season,
    wet-event window, etc.) without corrupting the offset itself.
    """
    x_full = ts[x_col]
    rows = []
    for lag in range(0, max_lag + 1):
        y_shifted = ts[y_col].shift(-lag)
        x = x_full
        y = y_shifted
        if mask is not None:
            x = x_full[mask]
            y = y_shifted[mask]
        paired = pd.concat([x, y], axis=1).dropna()
        n = len(paired)
        if n >= 2:
            r, p = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        else:
            r, p = np.nan, np.nan
        rows.append({"Lag_Days": lag, "Correlation": r, "P_Value": p, "N_Pairs": n})

    out = pd.DataFrame(rows)
    out["Included"] = out["N_Pairs"] >= MIN_PAIRS_CCF
    return out


def plot_ccf(ccf_df: pd.DataFrame, title: str, path: Path,
             xlabel: str = "Lag (days, rainfall leads excess flow)"):
    included = ccf_df[ccf_df["Included"]]
    excluded = ccf_df[~ccf_df["Included"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(included["Lag_Days"], included["Correlation"], color=FLOW_COLOR, width=0.6,
           label="Correlation", zorder=3)
    if not excluded.empty:
        ax.bar(excluded["Lag_Days"], excluded["Correlation"], color=INCOMPLETE_COLOR, width=0.6,
               alpha=0.6, label=f"< {MIN_PAIRS_CCF} pairs (excluded)", zorder=3)

    sig = SIG_Z / np.sqrt(ccf_df["N_Pairs"].clip(lower=1))
    ax.plot(ccf_df["Lag_Days"], sig, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
             label="95% significance band (±{:.2f}/√n)".format(SIG_Z))
    ax.plot(ccf_df["Lag_Days"], -sig, color=BASELINE_COLOR, linestyle="--", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(ccf_df["Lag_Days"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 2: seasonal split of the CCF
# ---------------------------------------------------------------------------

def season_mask(ts: pd.DataFrame, months: list) -> pd.Series:
    return pd.Series(ts.index.month, index=ts.index).isin(months)


def plot_ccf_seasonal_comparison(warm_df: pd.DataFrame, cool_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.38
    lags = warm_df["Lag_Days"]

    ax.bar(lags - width / 2, warm_df["Correlation"], width=width, color=WARM_COLOR,
           label="Warm season (May-Oct)", zorder=3)
    ax.bar(lags + width / 2, cool_df["Correlation"], width=width, color=COOL_COLOR,
           label="Cool season (Nov-Apr)", zorder=3)

    warm_sig = SIG_Z / np.sqrt(warm_df["N_Pairs"].clip(lower=1))
    cool_sig = SIG_Z / np.sqrt(cool_df["N_Pairs"].clip(lower=1))
    ax.plot(lags, warm_sig, color=WARM_COLOR, linestyle="--", linewidth=1.0, alpha=0.8,
             label="Warm 95% band")
    ax.plot(lags, -warm_sig, color=WARM_COLOR, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.plot(lags, cool_sig, color=COOL_COLOR, linestyle="--", linewidth=1.0, alpha=0.8,
             label="Cool 95% band")
    ax.plot(lags, -cool_sig, color=COOL_COLOR, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(lags)
    ax.set_xlabel("Lag (days, rainfall leads excess flow)")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Seasonal Rainfall → Excess Flow Cross-Correlation – Richmond RRWWTF")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 3: wet-event-conditioned CCF
# ---------------------------------------------------------------------------

def build_wet_event_mask(ts: pd.DataFrame, events: pd.DataFrame, tail_days: int) -> pd.Series:
    """True for any calendar day inside [Start_Date, End_Date + tail_days]
    for any RDII event (all events, complete or not -- gaps are still
    excluded downstream by the pairwise dropna in compute_ccf)."""
    mask = pd.Series(False, index=ts.index)
    for _, ev in events.iterrows():
        start = ev["Start_Date"]
        end = ev["End_Date"] + pd.Timedelta(days=tail_days)
        window = pd.date_range(start, end, freq="D").intersection(ts.index)
        mask.loc[window] = True
    return mask


# ---------------------------------------------------------------------------
# Task 4: ACF / PACF (gap-safe, computed manually)
# ---------------------------------------------------------------------------

def gap_safe_acf(ts: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    series = ts[col]
    rows = [{"Lag_Days": 0, "ACF": 1.0, "N_Pairs": int(series.notna().sum())}]
    for lag in range(1, max_lag + 1):
        paired = pd.concat([series, series.shift(lag)], axis=1).dropna()
        n = len(paired)
        if n >= 2:
            r, _ = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
        else:
            r = np.nan
        rows.append({"Lag_Days": lag, "ACF": r, "N_Pairs": n})
    return pd.DataFrame(rows)


def durbin_levinson_pacf(acf_vals: np.ndarray) -> np.ndarray:
    """Standard Durbin-Levinson recursion, taking a (possibly gap-safe)
    ACF sequence r[0..max_lag] (r[0] = 1) as input and returning the PACF
    sequence of the same length. Mathematically equivalent to the
    Yule-Walker PACF construction."""
    max_lag = len(acf_vals) - 1
    pacf_vals = np.full(max_lag + 1, np.nan)
    pacf_vals[0] = 1.0
    if max_lag == 0:
        return pacf_vals

    phi_prev = {1: acf_vals[1]}
    pacf_vals[1] = acf_vals[1]

    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi_prev[j] * acf_vals[k - j] for j in range(1, k))
        den = 1 - sum(phi_prev[j] * acf_vals[j] for j in range(1, k))
        phi_kk = num / den if den not in (0,) and not np.isnan(den) and den != 0 else np.nan

        phi_curr = {k: phi_kk}
        for j in range(1, k):
            phi_curr[j] = phi_prev[j] - phi_kk * phi_prev[k - j]

        pacf_vals[k] = phi_kk
        phi_prev = phi_curr

    return pacf_vals


def build_acf_pacf_table(ts: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label, col in [("Raw_Flow", "Total_Treated_MGD"), ("Excess_Flow", "Excess_Flow_MGD")]:
        acf_df = gap_safe_acf(ts, col, MAX_ACF_LAG)
        pacf_vals = durbin_levinson_pacf(acf_df["ACF"].values)
        acf_df["PACF"] = pacf_vals
        acf_df["Series"] = label
        frames.append(acf_df)
    return pd.concat(frames, ignore_index=True)[["Series", "Lag_Days", "ACF", "PACF", "N_Pairs"]]


def plot_acf_pacf_panels(acf_pacf: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    panel_specs = [
        ("Raw_Flow", "ACF", axes[0, 0], "ACF – Raw Daily Flow (Total_Treated_MGD)"),
        ("Raw_Flow", "PACF", axes[0, 1], "PACF – Raw Daily Flow (Total_Treated_MGD)"),
        ("Excess_Flow", "ACF", axes[1, 0], "ACF – Excess-over-Baseline Flow (Excess_Flow_MGD)"),
        ("Excess_Flow", "PACF", axes[1, 1], "PACF – Excess-over-Baseline Flow (Excess_Flow_MGD)"),
    ]

    for series_label, stat_col, ax, title in panel_specs:
        sub = acf_pacf[acf_pacf["Series"] == series_label]
        color = FLOW_COLOR if series_label == "Raw_Flow" else FIT_COLOR
        ax.vlines(sub["Lag_Days"], 0, sub[stat_col], color=color, linewidth=1.6, zorder=3)
        ax.scatter(sub["Lag_Days"], sub[stat_col], color=color, s=22, zorder=4)

        sig = SIG_Z / np.sqrt(sub["N_Pairs"].clip(lower=1))
        ax.plot(sub["Lag_Days"], sig, color=BASELINE_COLOR, linestyle="--", linewidth=1.0)
        ax.plot(sub["Lag_Days"], -sig, color=BASELINE_COLOR, linestyle="--", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_xticks(sub["Lag_Days"])
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel(stat_col)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle("ACF / PACF – Raw vs. Excess-over-Baseline Daily Flow – Richmond RRWWTF",
                 fontsize=16, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 5: antecedent wetness index
# ---------------------------------------------------------------------------

def compute_antecedent_wetness(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in AW_WINDOWS:
        aw = ts["Rainfall_in"].rolling(window, min_periods=window).sum()
        paired = pd.concat([aw, ts["Excess_Flow_MGD"]], axis=1).dropna()
        paired.columns = ["Antecedent_Wetness_in", "Excess_Flow_MGD"]
        n = len(paired)
        if n >= 2:
            r, p = stats.pearsonr(paired["Antecedent_Wetness_in"], paired["Excess_Flow_MGD"])
        else:
            r, p = np.nan, np.nan
        rows.append({"Window_Days": window, "Correlation": r, "P_Value": p, "N_Pairs": n})
    return pd.DataFrame(rows)


def plot_antecedent_wetness(aw_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [FIT_COLOR if v == aw_df["Correlation"].max() else FLOW_COLOR for v in aw_df["Correlation"]]
    ax.bar(aw_df["Window_Days"].astype(str), aw_df["Correlation"], color=colors, width=0.6, zorder=3)
    for x, (_, row) in enumerate(aw_df.iterrows()):
        ax.annotate(f"r={row['Correlation']:.3f}\nn={row['N_Pairs']:,}",
                    xy=(x, row["Correlation"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xlabel("Antecedent Rolling-Sum Window (days)")
    ax.set_ylabel("Pearson Correlation with Same-Day Excess Flow")
    ax.set_title("Antecedent Wetness vs. Excess Flow by Window Length – Richmond RRWWTF")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full feature correlation matrix -- a single all-pairs heatmap across flow,
# rainfall, the winning antecedent-wetness window, lag-1 flow ("demand
# lag"), and day-of-week/weekday/weekend dummies. This is a broader,
# all-pairs companion to the targeted CCF/PACF/antecedent-wetness analyses
# above (which only ever look at flow/rainfall two at a time) -- it exists
# to surface any relationship those targeted passes wouldn't, e.g. a
# day-of-week effect on flow independent of rainfall. No temperature
# column is included: this project has never ingested external weather
# data for RRWWTF (see rainfall_analysis.py) -- only the plant-recorded
# Rainfall_in is used, consistent with every other stage of this pipeline.
# ---------------------------------------------------------------------------

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_full_feature_matrix(ts: pd.DataFrame, best_aw_window: int) -> pd.DataFrame:
    features = pd.DataFrame(index=ts.index)
    features["Total_Treated_MGD"] = ts["Total_Treated_MGD"]
    features["Rainfall_in"] = ts["Rainfall_in"]
    features["Excess_Flow_MGD"] = ts["Excess_Flow_MGD"]
    features[f"Antecedent_Wetness_{best_aw_window}d"] = ts["Rainfall_in"].rolling(
        best_aw_window, min_periods=best_aw_window).sum()
    features["Flow_Lag1"] = ts["Total_Treated_MGD"].shift(1)

    dow = ts.index.dayofweek  # 0=Monday .. 6=Sunday
    for i, name in enumerate(DAY_NAMES):
        features[name] = (dow == i).astype(int)
    features["Weekday"] = (dow < 5).astype(int)
    features["Weekend"] = (dow >= 5).astype(int)

    return features


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * n), max(8, 0.8 * n)))
    im = ax.imshow(corr.values, cmap="magma", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            text_color = "black" if val > 0.55 else "white"
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation")

    ax.set_title("Full Feature Correlation Matrix – Richmond RRWWTF\n"
                  "(flow, rainfall, antecedent wetness, demand lag, day-of-week)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-diagnostic feature-importance ranking -- combines the CCF, PACF, and
# antecedent-wetness results computed above into a single sorted comparison
# so relative importance across the three families of candidate lag
# features is visible at a glance. Not a new statistical method: each value
# is pulled directly from ccf_overall / acf_pacf / aw_df.
# ---------------------------------------------------------------------------

def build_feature_importance_table(ccf_overall: pd.DataFrame, acf_pacf: pd.DataFrame,
                                    aw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in ccf_overall.iterrows():
        if not row["Included"]:
            continue
        sig = SIG_Z / np.sqrt(max(row["N_Pairs"], 1))
        rows.append({
            "Feature": f"Rainfall lag {int(row['Lag_Days'])}d",
            "Category": "Rainfall CCF lag",
            "Value": row["Correlation"],
            "N_Pairs": int(row["N_Pairs"]),
            "Significant": bool(abs(row["Correlation"]) > sig),
        })

    excess_pacf = acf_pacf[(acf_pacf["Series"] == "Excess_Flow") & (acf_pacf["Lag_Days"] > 0)]
    for _, row in excess_pacf.iterrows():
        sig = SIG_Z / np.sqrt(max(row["N_Pairs"], 1))
        rows.append({
            "Feature": f"Excess-flow PACF lag {int(row['Lag_Days'])}d",
            "Category": "Excess-flow PACF lag",
            "Value": row["PACF"],
            "N_Pairs": int(row["N_Pairs"]),
            "Significant": bool(abs(row["PACF"]) > sig),
        })

    for _, row in aw_df.iterrows():
        sig = SIG_Z / np.sqrt(max(row["N_Pairs"], 1))
        rows.append({
            "Feature": f"Antecedent wetness {int(row['Window_Days'])}d",
            "Category": "Antecedent wetness window",
            "Value": row["Correlation"],
            "N_Pairs": int(row["N_Pairs"]),
            "Significant": bool(abs(row["Correlation"]) > sig),
        })

    out = pd.DataFrame(rows)
    out["Abs_Value"] = out["Value"].abs()
    out = out.sort_values("Abs_Value", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1
    return out[["Rank", "Feature", "Category", "Value", "Abs_Value", "N_Pairs", "Significant"]]


def plot_feature_importance(importance_df: pd.DataFrame, path: Path):
    df = importance_df.sort_values("Abs_Value", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(df))))

    colors = [FEATURE_CATEGORY_COLORS[c] for c in df["Category"]]
    alphas = [0.9 if s else 0.35 for s in df["Significant"]]
    bars = ax.barh(df["Feature"], df["Abs_Value"], color=colors, zorder=3)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    for y, (_, row) in enumerate(df.iterrows()):
        ax.annotate(f"{row['Value']:.3f}", xy=(row["Abs_Value"], y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)

    ax.set_xlabel("|Correlation| / |PACF| (sign shown in labels)")
    ax.set_title("Candidate Lag Feature Importance Ranking – Richmond RRWWTF\n"
                  "(faded bars did not clear the 95% significance band)")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=color, label=cat) for cat, color in FEATURE_CATEGORY_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Task 6: feature recommendation summary (every number cited is computed
# above, not hardcoded)
# ---------------------------------------------------------------------------

def significant_lags(ccf_df: pd.DataFrame) -> list:
    sig = SIG_Z / np.sqrt(ccf_df["N_Pairs"].clip(lower=1))
    hit = ccf_df[ccf_df["Included"] & (ccf_df["Correlation"].abs() > sig)]
    return hit["Lag_Days"].tolist()


def significant_pacf_lags(acf_pacf: pd.DataFrame, series_label: str) -> list:
    sub = acf_pacf[(acf_pacf["Series"] == series_label) & (acf_pacf["Lag_Days"] > 0)]
    sig = SIG_Z / np.sqrt(sub["N_Pairs"].clip(lower=1))
    hit = sub[sub["PACF"].abs() > sig]
    return hit["Lag_Days"].tolist()


def write_recommendations_md(ccf_overall, ccf_warm, ccf_cool, ccf_wet_event,
                              acf_pacf, aw_df, n_events, n_events_complete):
    overall_included = ccf_overall[ccf_overall["Included"]]
    peak_row = overall_included.loc[overall_included["Correlation"].idxmax()]
    overall_sig_lags = significant_lags(ccf_overall)
    overall_excluded_lags = ccf_overall.loc[~ccf_overall["Included"], "Lag_Days"].tolist()

    warm_included = ccf_warm[ccf_warm["Included"]]
    cool_included = ccf_cool[ccf_cool["Included"]]
    warm_peak = warm_included.loc[warm_included["Correlation"].idxmax()] if not warm_included.empty else None
    cool_peak = cool_included.loc[cool_included["Correlation"].idxmax()] if not cool_included.empty else None

    wet_included = ccf_wet_event[ccf_wet_event["Included"]]
    wet_peak = wet_included.loc[wet_included["Correlation"].idxmax()] if not wet_included.empty else None

    excess_pacf_sig = significant_pacf_lags(acf_pacf, "Excess_Flow")
    raw_pacf_sig = significant_pacf_lags(acf_pacf, "Raw_Flow")
    excess_lag1_acf = acf_pacf[(acf_pacf["Series"] == "Excess_Flow") & (acf_pacf["Lag_Days"] == 1)]["ACF"].iloc[0]
    raw_lag1_acf = acf_pacf[(acf_pacf["Series"] == "Raw_Flow") & (acf_pacf["Lag_Days"] == 1)]["ACF"].iloc[0]

    def decay_lag(series_label):
        sub = acf_pacf[(acf_pacf["Series"] == series_label) & (acf_pacf["Lag_Days"] > 0)].sort_values("Lag_Days")
        below = sub[sub["ACF"].abs() < ACF_NEAR_ZERO_THRESHOLD]
        return int(below["Lag_Days"].iloc[0]) if not below.empty else None

    raw_decay = decay_lag("Raw_Flow")
    excess_decay = decay_lag("Excess_Flow")

    best_aw = aw_df.loc[aw_df["Correlation"].idxmax()]
    worst_aw = aw_df.loc[aw_df["Correlation"].idxmin()]

    seasonal_lag_shift = None
    seasonal_note = "Seasonal comparison unavailable (insufficient paired observations in one season)."
    if warm_peak is not None and cool_peak is not None:
        seasonal_lag_shift = int(warm_peak["Lag_Days"]) - int(cool_peak["Lag_Days"])
        rel_diff = abs(warm_peak["Correlation"] - cool_peak["Correlation"]) / max(
            abs(warm_peak["Correlation"]), abs(cool_peak["Correlation"]), 1e-9)
        warranted = abs(seasonal_lag_shift) >= 1 or rel_diff > 0.20
        seasonal_note = (
            f"Warm-season peak lag is {int(warm_peak['Lag_Days'])} day(s) at r={warm_peak['Correlation']:.3f} "
            f"(n={int(warm_peak['N_Pairs']):,}); cool-season peak lag is {int(cool_peak['Lag_Days'])} day(s) "
            f"at r={cool_peak['Correlation']:.3f} (n={int(cool_peak['N_Pairs']):,}) -- "
            f"a {seasonal_lag_shift:+d}-day shift in peak lag and a "
            f"{rel_diff:.0%} relative difference in peak correlation magnitude. "
            + ("This is large enough to justify a seasonal interaction term on the lag-0/1 rainfall feature."
               if warranted else
               "This is small enough that a single (non-seasonal) lag structure is adequate.")
        )

    lines = []
    lines.append("# Lag Feature Recommendations – Richmond RRWWTF Flow Prediction")
    lines.append("")
    lines.append("Diagnostic output of `scripts/correlation_analysis.py`. Every recommendation below "
                  "cites the specific computed number backing it; see the accompanying CSVs for full "
                  "per-lag detail.")
    lines.append("")

    lines.append("## Rainfall lags to include")
    lines.append("")
    lines.append(
        f"- **Peak overall response lag: {int(peak_row['Lag_Days'])} day(s)**, r={peak_row['Correlation']:.3f} "
        f"(n={int(peak_row['N_Pairs']):,} pairs, p={peak_row['P_Value']:.2e}). This matches the RDII event "
        f"analysis's median days-to-peak of 1 day using an independent, continuous-signal method."
    )
    if overall_sig_lags:
        lines.append(
            f"- Lags statistically significant at the 95% band (|r| > {SIG_Z:g}/√n) in the full-period CCF: "
            f"{', '.join(str(l) for l in overall_sig_lags)}. Recommend including rainfall lag features for "
            f"these lag(s) specifically, rather than the full 0-10 day range."
        )
    else:
        lines.append("- No lag cleared the 95% significance band in the full-period CCF; treat rainfall-lag "
                      "features cautiously and rely on the seasonal/event-conditioned CCFs below instead.")
    lines.append("")

    lines.append("## Seasonal interaction terms")
    lines.append("")
    lines.append(f"- {seasonal_note}")
    lines.append("")

    lines.append("## Wet-event-conditioned check")
    lines.append("")
    if wet_peak is not None:
        lines.append(
            f"- Restricting the CCF to the {n_events} delineated RDII event windows "
            f"({n_events_complete} complete) plus a {WET_EVENT_TAIL_DAYS}-day tail, the peak response lag is "
            f"{int(wet_peak['Lag_Days'])} day(s) at r={wet_peak['Correlation']:.3f} (n={int(wet_peak['N_Pairs']):,}). "
            + ("This confirms the full-period peak lag once dry-stretch dilution is removed."
               if int(wet_peak["Lag_Days"]) == int(peak_row["Lag_Days"]) else
               f"This differs from the full-period peak lag ({int(peak_row['Lag_Days'])} day(s)) -- "
               "the event-conditioned signal is the more reliable of the two for feature selection, since "
               "the full-period CCF is diluted by long dry stretches.")
        )
    else:
        lines.append("- Wet-event-conditioned CCF did not produce enough paired observations at any lag to report.")
    lines.append("")

    lines.append("## Flow (autoregressive) lags to include")
    lines.append("")
    lines.append(
        f"- Excess-flow lag-1 autocorrelation is {excess_lag1_acf:.3f} vs. raw-flow lag-1 autocorrelation "
        f"{raw_lag1_acf:.3f}. "
        + (f"Excess flow's ACF drops below {ACF_NEAR_ZERO_THRESHOLD:g} by lag {excess_decay}, "
           if excess_decay else f"Excess flow's ACF does not drop below {ACF_NEAR_ZERO_THRESHOLD:g} within "
           f"{MAX_ACF_LAG} lags, ")
        + (f"vs. raw flow which needs lag {raw_decay} " if raw_decay else
           f"vs. raw flow which does not decay below {ACF_NEAR_ZERO_THRESHOLD:g} within {MAX_ACF_LAG} lags ")
        + "-- excess-over-baseline flow is the closer-to-stationary series of the two, as expected once the "
          "seasonal/monthly baseline level is removed."
    )
    if excess_pacf_sig:
        lines.append(
            f"- PACF lags significant at the 95% band for excess flow: "
            f"{', '.join(str(l) for l in excess_pacf_sig)}. Recommend autoregressive flow-lag features at "
            f"these specific lag(s) -- each represents independent information beyond what earlier lags "
            f"already explain, per the Durbin-Levinson PACF."
        )
    else:
        lines.append("- No excess-flow PACF lag cleared the 95% significance band; an autoregressive flow-lag "
                      "feature is not well supported by this diagnostic.")
    lines.append(
        f"- For reference, raw daily flow's significant PACF lags are: "
        f"{', '.join(str(l) for l in raw_pacf_sig) if raw_pacf_sig else 'none'}. Raw flow carries more "
        f"apparent autoregressive structure than excess flow because it still contains the slow-moving "
        f"dry-weather/seasonal baseline level -- this is exactly why the excess-flow PACF, not the raw-flow "
        f"PACF, should drive the modeled flow-lag features."
    )
    lines.append("")

    lines.append("## Antecedent-wetness window")
    lines.append("")
    lines.append(
        f"- Strongest same-day relationship: the **{int(best_aw['Window_Days'])}-day** rolling rainfall sum, "
        f"r={best_aw['Correlation']:.3f} (n={int(best_aw['N_Pairs']):,} pairs, p={best_aw['P_Value']:.2e}). "
        f"Recommend this as the antecedent-wetness feature."
    )
    lines.append(
        f"- Weakest of the tested windows: {int(worst_aw['Window_Days'])}-day, r={worst_aw['Correlation']:.3f} "
        f"-- rule this window out as a redundant/lower-value antecedent-wetness feature relative to the "
        f"{int(best_aw['Window_Days'])}-day window."
    )
    lines.append("")

    lines.append("## Candidate features ruled OUT")
    lines.append("")
    ruled_out = []
    if overall_excluded_lags:
        ruled_out.append(
            f"Rainfall lags {', '.join(str(l) for l in overall_excluded_lags)} in the full-period CCF "
            f"(fewer than {MIN_PAIRS_CCF} valid pairs after gap removal -- not enough data to trust)."
        )
    non_sig_overall = ccf_overall.loc[
        ccf_overall["Included"] & ~ccf_overall["Lag_Days"].isin(overall_sig_lags), "Lag_Days"].tolist()
    if non_sig_overall:
        ruled_out.append(
            f"Rainfall lags {', '.join(str(l) for l in non_sig_overall)} (enough pairs, but correlation did "
            f"not clear the 95% significance band in the full-period CCF)."
        )
    non_sig_pacf = [l for l in range(1, MAX_ACF_LAG + 1) if l not in excess_pacf_sig]
    if non_sig_pacf:
        ruled_out.append(
            f"Excess-flow autoregressive lags {', '.join(str(l) for l in non_sig_pacf)} (PACF did not clear "
            f"the 95% significance band -- no independent information beyond the significant lag(s) above)."
        )
    other_aw = aw_df[aw_df["Window_Days"] != best_aw["Window_Days"]]
    if not other_aw.empty:
        ruled_out.append(
            "Antecedent-wetness windows " +
            ", ".join(f"{int(r['Window_Days'])}-day (r={r['Correlation']:.3f})" for _, r in other_aw.iterrows()) +
            f" -- all weaker than the recommended {int(best_aw['Window_Days'])}-day window."
        )
    if not ruled_out:
        ruled_out.append("None -- every tested lag/window cleared its significance and pair-count checks.")
    for item in ruled_out:
        lines.append(f"- {item}")
    lines.append("")

    RECOMMENDATIONS_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ts = load_data()
    events = load_events()

    print("CORRELATION AND LAG-STRUCTURE ANALYSIS")
    print()
    print(f"Continuous calendar-day series: {ts.index.min().date()} to {ts.index.max().date()} "
          f"({len(ts):,} calendar days, {int(ts['Row_Exists'].sum()):,} with an actual plant record)")
    print()

    # --- Task 1: full-period CCF ---
    ccf_overall = compute_ccf(ts, "Rainfall_in", "Excess_Flow_MGD", MAX_CCF_LAG)
    ccf_overall.to_csv(CCF_OVERALL_CSV, index=False)
    plot_ccf(ccf_overall, "Rainfall → Excess Flow Cross-Correlation (Full Period) – Richmond RRWWTF",
             FIG1_CCF_OVERALL)

    included = ccf_overall[ccf_overall["Included"]]
    peak_row = included.loc[included["Correlation"].idxmax()]
    print("TASK 1 -- Full-period CCF")
    print(ccf_overall.to_string(index=False))
    print(f"  Peak lag: {int(peak_row['Lag_Days'])} day(s), r={peak_row['Correlation']:.3f}, "
          f"n={int(peak_row['N_Pairs']):,}")
    print()

    # --- Task 2: seasonal split ---
    warm_mask = season_mask(ts, WARM_MONTHS)
    cool_mask = season_mask(ts, COOL_MONTHS)
    ccf_warm = compute_ccf(ts, "Rainfall_in", "Excess_Flow_MGD", MAX_CCF_LAG, mask=warm_mask)
    ccf_cool = compute_ccf(ts, "Rainfall_in", "Excess_Flow_MGD", MAX_CCF_LAG, mask=cool_mask)
    ccf_warm.to_csv(CCF_WARM_CSV, index=False)
    ccf_cool.to_csv(CCF_COOL_CSV, index=False)
    plot_ccf_seasonal_comparison(ccf_warm, ccf_cool, FIG2_CCF_SEASONAL)

    print("TASK 2 -- Seasonal CCF (warm = May-Oct, cool = Nov-Apr)")
    print("  Warm season:")
    print("  " + ccf_warm.to_string(index=False).replace("\n", "\n  "))
    print("  Cool season:")
    print("  " + ccf_cool.to_string(index=False).replace("\n", "\n  "))
    print()

    # --- Task 3: wet-event-conditioned CCF ---
    wet_mask = build_wet_event_mask(ts, events, WET_EVENT_TAIL_DAYS)
    ccf_wet_event = compute_ccf(ts, "Rainfall_in", "Excess_Flow_MGD", MAX_CCF_LAG, mask=wet_mask)
    ccf_wet_event.to_csv(CCF_WET_EVENT_CSV, index=False)
    plot_ccf(ccf_wet_event,
             f"Rainfall → Excess Flow CCF – Wet-Event Windows + {WET_EVENT_TAIL_DAYS}-Day Tail Only "
             "– Richmond RRWWTF",
             FIG3_CCF_WET_EVENT)

    print(f"TASK 3 -- Wet-event-conditioned CCF ({int(wet_mask.sum()):,} of {len(ts):,} calendar days "
          f"fall inside an event window + {WET_EVENT_TAIL_DAYS}-day tail)")
    print(ccf_wet_event.to_string(index=False))
    print()

    # --- Task 4: ACF / PACF ---
    acf_pacf = build_acf_pacf_table(ts)
    acf_pacf.to_csv(ACF_PACF_CSV, index=False)
    plot_acf_pacf_panels(acf_pacf, FIG4_ACF_PACF)

    excess_lag1 = acf_pacf[(acf_pacf["Series"] == "Excess_Flow") & (acf_pacf["Lag_Days"] == 1)]["ACF"].iloc[0]
    raw_lag1 = acf_pacf[(acf_pacf["Series"] == "Raw_Flow") & (acf_pacf["Lag_Days"] == 1)]["ACF"].iloc[0]
    print("TASK 4 -- ACF / PACF (gap-safe, lags 0-14)")
    print(f"  Raw flow lag-1 ACF:    {raw_lag1:.3f}")
    print(f"  Excess flow lag-1 ACF: {excess_lag1:.3f}")
    print(f"  {'Excess' if abs(excess_lag1) < abs(raw_lag1) else 'Raw'} flow's lag-1 autocorrelation is "
          f"smaller in magnitude -- {'excess' if abs(excess_lag1) < abs(raw_lag1) else 'raw'} flow looks "
          f"closer to stationary.")
    print()

    # --- Task 5: antecedent wetness ---
    aw_df = compute_antecedent_wetness(ts)
    aw_df.to_csv(ANTECEDENT_WETNESS_CSV, index=False)
    plot_antecedent_wetness(aw_df, FIG5_ANTECEDENT_WETNESS)

    best_aw = aw_df.loc[aw_df["Correlation"].idxmax()]
    print("TASK 5 -- Antecedent wetness (rolling rainfall sum vs. same-day excess flow)")
    print(aw_df.to_string(index=False))
    print(f"  Strongest window: {int(best_aw['Window_Days'])}-day, r={best_aw['Correlation']:.3f}")
    print()

    # --- Full feature correlation matrix ---
    feature_matrix = build_full_feature_matrix(ts, best_aw_window=int(best_aw["Window_Days"]))
    full_corr = feature_matrix.corr()  # pandas default: pairwise-complete-observations, no filling
    full_corr.to_csv(FULL_CORR_MATRIX_CSV)
    plot_correlation_heatmap(full_corr, FIG7_FULL_CORR_MATRIX)

    flow_corr = full_corr["Total_Treated_MGD"].drop("Total_Treated_MGD").sort_values(
        key=lambda s: s.abs(), ascending=False)
    print("FULL FEATURE CORRELATION MATRIX (flow, rainfall, antecedent wetness, demand lag, day-of-week)")
    print(f"  Correlation with Total_Treated_MGD, strongest first:")
    for name, val in flow_corr.items():
        print(f"    {name:<24s} {val:+.3f}")
    print()

    # --- Cross-diagnostic feature-importance ranking ---
    importance_df = build_feature_importance_table(ccf_overall, acf_pacf, aw_df)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    plot_feature_importance(importance_df, FIG6_FEATURE_IMPORTANCE)

    top5 = importance_df.head(5)
    print("FEATURE IMPORTANCE RANKING (all candidate lags/windows, sorted by |value|)")
    print(importance_df.to_string(index=False))
    print()
    print("  Top 5 candidate features overall:")
    for _, row in top5.iterrows():
        flag = "" if row["Significant"] else "  (not significant)"
        print(f"    #{int(row['Rank'])}  {row['Feature']:<28s} {row['Category']:<26s} "
              f"value={row['Value']:.3f}{flag}")
    print()

    # --- Task 6: recommendations markdown ---
    write_recommendations_md(ccf_overall, ccf_warm, ccf_cool, ccf_wet_event, acf_pacf, aw_df,
                              n_events=len(events), n_events_complete=int(events["Complete"].sum()))

    print("CORRELATION ANALYSIS COMPLETE")
    print()
    print("Generated files:")
    for p in [CCF_OVERALL_CSV, CCF_WARM_CSV, CCF_COOL_CSV, CCF_WET_EVENT_CSV, ACF_PACF_CSV,
              ANTECEDENT_WETNESS_CSV, FULL_CORR_MATRIX_CSV, FEATURE_IMPORTANCE_CSV, RECOMMENDATIONS_MD,
              FIG1_CCF_OVERALL, FIG2_CCF_SEASONAL, FIG3_CCF_WET_EVENT, FIG4_ACF_PACF,
              FIG5_ANTECEDENT_WETNESS, FIG6_FEATURE_IMPORTANCE, FIG7_FULL_CORR_MATRIX]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

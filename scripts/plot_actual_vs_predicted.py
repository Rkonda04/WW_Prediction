"""
RRWWTF Observed vs. Predicted Daily Flow -- Time-Series Figures

Reads the Phase-5 out-of-sample predictions written by
`scripts/train_screened_model.py` and renders the two report figures:

    04_actual_vs_predicted_all_years.png
        Small multiples, one panel per out-of-sample year (2020, 2021, 2022,
        2024). Each year is predicted by a model fitted only on years before
        it, so every point plotted is out-of-sample. Panels share a y-axis and
        a Jan-Dec x-axis so the years are directly comparable.

    05_actual_vs_predicted_2024_detail.png
        The held-out 2024 test year at full width, with the screened no-rain
        spike days marked and the largest residuals called out.

Years are NOT plotted on one continuous axis: 2018/2019 are training-only,
2023 is a four-month fragment excluded from the study, and Sep-Dec 2024 does
not exist yet. A single axis would draw long straight segments across those
gaps that look like data. Faceting by year is the honest layout.

Inputs (read-only):
    output/13_screened_model/out_of_sample_predictions_all_years.csv
    output/13_screened_model/screened_days.csv

Usage:
    python scripts/plot_actual_vs_predicted.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "output" / "13_screened_model"

OOS_CSV = MODEL_DIR / "out_of_sample_predictions_all_years.csv"
SCREENED_CSV = MODEL_DIR / "screened_days.csv"

FIG_ALL = MODEL_DIR / "04_actual_vs_predicted_all_years.png"
FIG_2024 = MODEL_DIR / "05_actual_vs_predicted_2024_detail.png"

# Categorical slots 1 and 2 from the validated reference palette.
# Checked with dataviz/scripts/validate_palette.js --mode light: all six checks
# PASS, worst adjacent CVD dE 24.7 (protan), normal-vision dE 33.6.
OBSERVED = "#2a78d6"
PREDICTED = "#eb6834"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e4e1"
SURFACE = "#fcfcfb"
FLAG = "#e34948"

PANEL_ORDER = ["Fold 1 val 2020", "Fold 2 val 2021", "Fold 3 val 2022", "TEST 2024"]
PANEL_TITLE = {
    "Fold 1 val 2020": "2020",
    "Fold 2 val 2021": "2021",
    "Fold 3 val 2022": "2022",
    "TEST 2024": "2024",
}
PANEL_NOTE = {
    "Fold 1 val 2020": "CV validation year - model trained on 2018-2019",
    "Fold 2 val 2021": "CV validation year - model trained on 2018-2020",
    "Fold 3 val 2022": "CV validation year - model trained on 2018-2021",
    "TEST 2024": "HELD-OUT TEST YEAR - scored once (record ends 31 Aug)",
}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def load():
    df = pd.read_csv(OOS_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    screened = pd.read_csv(SCREENED_CSV, parse_dates=["Date"])
    return df, screened


def scores(sub):
    err = sub["p"] - sub["y"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((sub["y"] - sub["y"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float((err.abs() / sub["y"]).mean() * 100)
    return r2, rmse, 100 - mape


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def to_common_year(dates):
    """Map every year onto 2000 (a leap year) so panels share one Jan-Dec axis."""
    return pd.to_datetime({
        "year": 2000,
        "month": dates.dt.month,
        "day": dates.dt.day,
    })


def month_axis(ax, labels=True):
    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    if labels:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment("left")
    else:
        ax.xaxis.set_major_formatter(plt.NullFormatter())


# ---------------------------------------------------------------------------
# Figure 1 -- small multiples, one panel per out-of-sample year
# ---------------------------------------------------------------------------

def figure_all_years(df):
    ymax = float(max(df["y"].max(), df["p"].max())) * 1.08
    fig, axes = plt.subplots(
        len(PANEL_ORDER), 1, figsize=(9.6, 8.4), sharey=True,
        # top leaves room for title + subtitle + legend above the first panel,
        # whose own heading sits at 1.20 in axes coords
        gridspec_kw={"hspace": 0.42, "top": 0.845, "bottom": 0.085,
                     "left": 0.075, "right": 0.985},
    )

    for ax, period in zip(axes, PANEL_ORDER):
        sub = df[df["Period"] == period].copy()
        sub["x"] = to_common_year(sub["Date"])
        # break the line wherever the record skips days, so gaps stay gaps
        gap = sub["Date"].diff().dt.days.fillna(1) > 1
        seg = gap.cumsum()

        for _, chunk in sub.groupby(seg):
            ax.plot(chunk["x"], chunk["y"], color=OBSERVED, linewidth=1.5,
                    solid_capstyle="round", zorder=3)
            ax.plot(chunk["x"], chunk["p"], color=PREDICTED, linewidth=1.5,
                    solid_capstyle="round", zorder=4)

        r2, rmse, acc = scores(sub)
        ax.set_ylim(0, ymax)
        month_axis(ax, labels=period == PANEL_ORDER[-1])
        tidy(ax)

        ax.text(0.0, 1.20, PANEL_TITLE[period], transform=ax.transAxes,
                fontsize=13, fontweight="bold", color=INK, va="top")
        ax.text(0.068, 1.185, PANEL_NOTE[period], transform=ax.transAxes,
                fontsize=8, color=INK_MUTED, va="top")
        ax.text(1.0, 1.20,
                f"n = {len(sub)}    R$^2$ = {r2:.2f}    RMSE = {rmse:.3f} MGD"
                f"    mean accuracy {acc:.0f}%",
                transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY,
                va="top", ha="right")

    axes[len(PANEL_ORDER) // 2].set_ylabel("Daily treated flow (MGD)",
                                           fontsize=9.5, color=INK_SECONDARY)
    axes[len(PANEL_ORDER) // 2].yaxis.set_label_coords(-0.055, 1.05)

    fig.text(0.075, 0.975, "Observed vs. predicted daily flow, out-of-sample years",
             fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.075, 0.945,
             "Richmond Regional WWTF - each year predicted by a model fitted only on the years before it",
             fontsize=9.5, color=INK_SECONDARY, va="top")

    handles = [
        Line2D([], [], color=OBSERVED, linewidth=2, label="Observed (plant log, 24-h normalized)"),
        Line2D([], [], color=PREDICTED, linewidth=2, label="Predicted (XGBoost, screened decomposition model)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.073, 0.912),
               frameon=False, ncol=2, fontsize=9, handlelength=1.8,
               columnspacing=1.6, labelcolor=INK_SECONDARY)

    fig.text(0.075, 0.022,
             "Pooled across all 1,246 days shown: R$^2$ = 0.42, RMSE = 0.358 MGD, mean accuracy 86.5%. "
             "Line breaks are days with no plant record or an excluded reading - never interpolated.",
             fontsize=8, color=INK_MUTED, va="bottom")

    fig.savefig(FIG_ALL, dpi=200)
    plt.close(fig)
    return FIG_ALL


# ---------------------------------------------------------------------------
# Figure 2 -- the held-out test year in detail
# ---------------------------------------------------------------------------

def figure_2024(df, screened):
    sub = df[df["Period"] == "TEST 2024"].copy().reset_index(drop=True)
    flags = screened[screened["Date"].dt.year == 2024]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.135, left=0.075, right=0.985)

    # screened no-rain spike days: the model was blind to these by design
    for _, row in flags.iterrows():
        ax.axvspan(row["Date"] - pd.Timedelta(hours=14),
                   row["Date"] + pd.Timedelta(hours=14),
                   color=FLAG, alpha=0.11, linewidth=0, zorder=1)

    gap = sub["Date"].diff().dt.days.fillna(1) > 1
    for _, chunk in sub.groupby(gap.cumsum()):
        ax.plot(chunk["Date"], chunk["y"], color=OBSERVED, linewidth=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot(chunk["Date"], chunk["p"], color=PREDICTED, linewidth=1.6,
                solid_capstyle="round", zorder=4)

    # axis limits must be final before the callouts, which position against them
    ax.set_ylim(0, float(max(sub["y"].max(), sub["p"].max())) * 1.20)
    ax.set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-01"))

    # Call out the largest residuals -- the storm peaks the model misses.
    # Picked greedily with a minimum separation so the labels spread across the
    # year instead of stacking on one cluster (the top 3 by size are all in
    # January and would overplot each other).
    sub["resid"] = sub["y"] - sub["p"]
    ranked = sub.reindex(sub["resid"].abs().sort_values(ascending=False).index)
    picked = []
    for _, row in ranked.iterrows():
        if all(abs((row["Date"] - p["Date"]).days) >= 30 for p in picked):
            picked.append(row)
        if len(picked) == 3:
            break

    xmin, xmax = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    # points of vertical space per MGD, so a callout can clear nearby peaks
    axes_h_pt = ax.get_window_extent().height * 72 / fig.dpi
    pt_per_mgd = axes_h_pt / (yhi - ylo)

    for row in picked:
        frac = (mdates.date2num(row["Date"]) - xmin) / (xmax - xmin)
        # keep the text inside the axes at both ends
        ha, dx = "center", 0
        if frac < 0.12:
            ha, dx = "left", -6
        elif frac > 0.88:
            ha, dx = "right", 6
        # lift the label above the tallest observed point nearby, not just above
        # its own point -- otherwise it lands on a taller neighbouring peak
        window = sub[(sub["Date"] - row["Date"]).abs() <= pd.Timedelta(days=7)]
        clearance = max(0.0, float(window["y"].max()) - float(row["y"]))
        ax.annotate(
            f"{row['Date']:%d %b}   obs {row['y']:.2f} / pred {row['p']:.2f}",
            xy=(row["Date"], row["y"]),
            xytext=(dx, 16 + clearance * pt_per_mgd), textcoords="offset points",
            fontsize=7.5, color=INK_SECONDARY, ha=ha, va="bottom",
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.7,
                            shrinkA=0, shrinkB=3),
        )

    r2, rmse, acc = scores(sub)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("left")
    ax.set_ylabel("Daily treated flow (MGD)", fontsize=9.5, color=INK_SECONDARY)
    tidy(ax)

    fig.text(0.075, 0.975, "Held-out test year: observed vs. predicted daily flow, 2024",
             fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.075, 0.928,
             f"Richmond Regional WWTF - scored once, never used for model selection.    "
             f"n = {len(sub)}    R$^2$ = {r2:.2f}    RMSE = {rmse:.3f} MGD    mean accuracy {acc:.0f}%",
             fontsize=9.5, color=INK_SECONDARY, va="top")

    handles = [
        Line2D([], [], color=OBSERVED, linewidth=2, label="Observed"),
        Line2D([], [], color=PREDICTED, linewidth=2, label="Predicted"),
        Line2D([], [], color=FLAG, alpha=0.3, linewidth=7,
               label="Screened no-rain spike day (excluded from scoring)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.073, 0.90),
               frameon=False, ncol=3, fontsize=9, handlelength=1.8,
               columnspacing=1.6, labelcolor=INK_SECONDARY)

    fig.text(0.075, 0.018,
             "Plant record ends 31 Aug 2024. Storm peaks are systematically underpredicted "
             "(high-flow days: mean error -0.440 MGD) - the model's main remaining failure mode.",
             fontsize=8, color=INK_MUTED, va="bottom")

    fig.savefig(FIG_2024, dpi=200)
    plt.close(fig)
    return FIG_2024


def main():
    style()
    df, screened = load()
    for path in (figure_all_years(df), figure_2024(df, screened)):
        print(f"wrote {path.relative_to(BASE_DIR)}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

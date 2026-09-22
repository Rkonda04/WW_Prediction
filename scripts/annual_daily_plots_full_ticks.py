"""
RRWWTF Step 2 — Annual Daily-Resolution Flow & Rainfall Figures (full daily ticks)

Variant of annual_daily_plots.py where EVERY individual calendar day gets
its own labeled X-axis tick (no weekly tick reduction). Intended as a
large-format engineering review figure (huge canvas, 300 dpi PNG + vector
PDF) meant to be zoomed or printed, not viewed at normal screen size.

This script performs NO data extraction, recomputation, aggregation,
smoothing, or interpolation — it only reads the existing merged dataset
and plots it.

Input (read-only, not modified):
    output/richmond_daily_flow_rainfall.csv

Output:
    output/figures/annual_daily_full_ticks/flow_rainfall_daily_<year>.png
    output/figures/annual_daily_full_ticks/flow_rainfall_daily_<year>.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MERGED_CSV = OUTPUT_DIR / "richmond_daily_flow_rainfall.csv"
FULL_TICKS_DIR = OUTPUT_DIR / "figures" / "annual_daily_full_ticks"

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"


def build_year_index(df: pd.DataFrame, year: int, dataset_max: pd.Timestamp) -> pd.DataFrame:
    """Reindex the merged data onto a continuous daily calendar index for the
    year, so every calendar day gets its own x-axis position and missing
    flow days appear as real gaps rather than being skipped."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    if year == dataset_max.year:
        end = min(end, dataset_max)
    full_index = pd.date_range(start=start, end=end, freq="D")
    year_df = df[(df["Date"] >= start) & (df["Date"] <= end)].set_index("Date")
    return year_df.reindex(full_index)


def plot_year(year_df: pd.DataFrame, year: int, out_png: Path, out_pdf: Path):
    fig, ax1 = plt.subplots(figsize=(48, 12))

    # --- Left axis: daily flow, line with small markers, real gaps at NaN ---
    ax1.plot(year_df.index, year_df["Total_Treated_MGD"],
              color=FLOW_COLOR, linewidth=1.1, marker="o", markersize=3,
              markerfacecolor=FLOW_COLOR, markeredgewidth=0, label="Total Treated Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, axis="y", linewidth=0.3, alpha=0.3)
    ax1.grid(True, axis="x", linewidth=0.2, alpha=0.15)

    # --- Right axis: daily rainfall bars, exact same daily x-positions as flow ---
    ax2 = ax1.twinx()
    rain_valid = year_df.dropna(subset=["Rainfall_in"])
    bars = ax2.bar(rain_valid.index, rain_valid["Rainfall_in"], width=0.8,
                    color=RAIN_COLOR, alpha=0.6, label="Daily Rainfall")
    ax2.set_ylabel("Daily Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    year_start = year_df.index.min()
    year_end = year_df.index.max()
    ax1.set_xlim(year_start, year_end)

    # --- Subtle month-boundary separators + month abbreviation labels ---
    month_starts = pd.date_range(start=pd.Timestamp(year=year, month=1, day=1),
                                  end=pd.Timestamp(year=year, month=12, day=1), freq="MS")
    for ms in month_starts:
        if year_start <= ms <= year_end:
            ax1.axvline(ms, color="gray", linewidth=0.8, alpha=0.4, zorder=0)

    month_ends = month_starts + pd.offsets.MonthEnd(0)
    for ms, me in zip(month_starts, month_ends):
        span_end = min(me, year_end)
        span_start = max(ms, year_start)
        if span_start > year_end:
            continue
        mid = span_start + (span_end - span_start) / 2
        ax1.text(mid, 1.02, ms.strftime("%b").upper(), transform=ax1.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=13, color="dimgray", fontweight="bold")

    # --- Daily ticks: EVERY calendar day gets its own labeled tick ---
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", fontsize=6)

    ax1.set_title(f"RRWWTF Daily Total Treated Flow and Rainfall — {year}", pad=28, fontsize=18)

    lines, line_labels = ax1.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bar_handles, line_labels + bar_labels, loc="upper right", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "n_tick_positions": len(year_df.index),
        "n_flow_obs": int(year_df["Total_Treated_MGD"].notna().sum()),
        "n_rain_obs": int(year_df["Rainfall_in"].notna().sum()),
        "first_date": year_start,
        "last_date": year_end,
    }


def main():
    FULL_TICKS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    dataset_max = df["Date"].max()

    generated = []
    validation_lines = []

    for year in YEARS:
        if df[df["Date"].dt.year == year].empty:
            continue

        year_df = build_year_index(df, year, dataset_max)
        out_png = FULL_TICKS_DIR / f"flow_rainfall_daily_{year}.png"
        out_pdf = FULL_TICKS_DIR / f"flow_rainfall_daily_{year}.pdf"
        info = plot_year(year_df, year, out_png, out_pdf)
        generated.append(year)

        validation_lines.append(
            f"{year}: {info['n_tick_positions']} calendar-day positions, "
            f"daily tick interval = 1 day | flow obs plotted = {info['n_flow_obs']} | "
            f"rainfall obs plotted = {info['n_rain_obs']} | "
            f"first date = {info['first_date'].date()} | last date = {info['last_date'].date()}"
        )

    print("RRWWTF ANNUAL DAILY (FULL TICKS) FLOW-RAINFALL FIGURES COMPLETE")
    print(f"Annual figures generated:         {len(generated)}")
    print(f"Years generated:                  {', '.join(str(y) for y in generated)}")
    print(f"Output directory:                 {FULL_TICKS_DIR}")
    print()
    print("Validation (per-year tick/observation counts):")
    for line in validation_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()

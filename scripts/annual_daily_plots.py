"""
RRWWTF Step 2 — Annual Daily-Resolution Flow & Rainfall Figures

Replaces the month-by-month visualization with one large daily-resolution
figure per year. This script performs NO data extraction, re-extraction,
recomputation, aggregation, smoothing, or interpolation — it only reads the
existing merged dataset and plots it.

Input (read-only, not modified):
    output/richmond_daily_flow_rainfall.csv

Output:
    output/figures/annual_daily/flow_rainfall_daily_<year>.png
    output/figures/annual_daily/flow_rainfall_daily_<year>.pdf
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
ANNUAL_DIR = OUTPUT_DIR / "figures" / "annual_daily"

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"


def build_year_index(df: pd.DataFrame, year: int, dataset_max: pd.Timestamp) -> pd.DataFrame:
    """Reindex the merged data onto a continuous daily calendar index for the
    year, so missing days appear as real NaN gaps rather than being skipped
    (which would otherwise let the line connect across the gap)."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    if year == dataset_max.year:
        end = min(end, dataset_max)
    full_index = pd.date_range(start=start, end=end, freq="D")
    year_df = df[(df["Date"] >= start) & (df["Date"] <= end)].set_index("Date")
    return year_df.reindex(full_index)


def plot_year(year_df: pd.DataFrame, year: int, out_png: Path, out_pdf: Path):
    fig, ax1 = plt.subplots(figsize=(22, 11))

    # --- Left axis: daily flow, line with small markers, real gaps at NaN ---
    ax1.plot(year_df.index, year_df["Total_Treated_MGD"],
              color=FLOW_COLOR, linewidth=1.1, marker="o", markersize=3,
              markerfacecolor=FLOW_COLOR, markeredgewidth=0, label="Total Treated Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.3, alpha=0.3)

    # --- Right axis: daily rainfall bars, missing values simply not drawn ---
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
            ax1.axvline(ms, color="gray", linewidth=0.6, alpha=0.35, zorder=0)

    month_ends = month_starts + pd.offsets.MonthEnd(0)
    for ms, me in zip(month_starts, month_ends):
        span_end = min(me, year_end)
        span_start = max(ms, year_start)
        if span_start > year_end:
            continue
        mid = span_start + (span_end - span_start) / 2
        ax1.text(mid, 1.015, ms.strftime("%b").upper(), transform=ax1.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9, color="gray")

    # --- Date ticks: every ~7 days, concise format, rotated ---
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax1.set_title(f"RRWWTF Daily Total Treated Flow and Rainfall — {year}", pad=20)

    lines, line_labels = ax1.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bar_handles, line_labels + bar_labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ANNUAL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    dataset_max = df["Date"].max()

    coverage_notes = []
    generated = []

    for year in YEARS:
        if df[(df["Date"].dt.year == year)].empty:
            coverage_notes.append(f"{year}: no data in merged dataset - skipped")
            continue

        year_df = build_year_index(df, year, dataset_max)
        total_days = len(year_df)
        valid_flow_days = int(year_df["Total_Treated_MGD"].notna().sum())
        missing_flow_days = total_days - valid_flow_days
        missing_pct = 100 * missing_flow_days / total_days if total_days else 0

        out_png = ANNUAL_DIR / f"flow_rainfall_daily_{year}.png"
        out_pdf = ANNUAL_DIR / f"flow_rainfall_daily_{year}.pdf"
        plot_year(year_df, year, out_png, out_pdf)
        generated.append(year)

        if year == dataset_max.year and year_df.index.max() < pd.Timestamp(year=year, month=12, day=31):
            coverage_notes.append(
                f"{year}: partial year, plotted through {year_df.index.max().date()} "
                f"(dataset end) - {missing_flow_days}/{total_days} days without flow "
                f"({missing_pct:.0f}%)"
            )
        elif missing_pct >= 5:
            coverage_notes.append(
                f"{year}: incomplete flow coverage - {missing_flow_days}/{total_days} days "
                f"without flow ({missing_pct:.0f}%), gaps shown as breaks in the line"
            )
        elif missing_flow_days > 0:
            coverage_notes.append(
                f"{year}: {missing_flow_days}/{total_days} days without flow ({missing_pct:.0f}%)"
            )

    print("RRWWTF ANNUAL DAILY FLOW-RAINFALL FIGURES COMPLETE")
    print(f"Annual figures generated:         {len(generated)}")
    print(f"Years generated:                  {', '.join(str(y) for y in generated)}")
    print(f"Output directory:                 {ANNUAL_DIR}")
    print("Coverage / plotting notes:")
    if coverage_notes:
        for note in coverage_notes:
            print(f"  - {note}")
    else:
        print("  - None; all years have complete daily flow coverage")


if __name__ == "__main__":
    main()

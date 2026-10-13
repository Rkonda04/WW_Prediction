"""
RRWWTF — Annual Daily-Resolution Flow vs. Open-Meteo Rainfall Figures

Same visual style as scripts/annual_daily_plots.py (one continuous
January-to-December daily timeline per year, month separators/labels,
upward rainfall bars, wide landscape figure) but with rainfall sourced
ONLY from the independent Open-Meteo download instead of the `Rain`
column in the original RRWWTF workbooks, which is never read here.

This script performs NO data extraction, recomputation, aggregation,
smoothing, or interpolation beyond building the daily calendar skeleton
needed to show true gaps - it reuses the existing merged dataset built by
monthly_flow_openmeteo_plots.py (regenerating it via the same functions
only if that file is missing).

Input (read-only, not modified):
    output/richmond_flow_openmeteo_daily.csv
    (falls back to output/richmond_total_treated_mgd.csv +
     output/richmond_openmeteo_daily_rainfall.csv if the merged file
     does not exist yet)

Output:
    output/figures/annual_daily_openmeteo/flow_openmeteo_rainfall_daily_<year>.png
    output/Richmond_Yearly_Daily_Flow_OpenMeteo_Rainfall.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MERGED_CSV = OUTPUT_DIR / "richmond_flow_openmeteo_daily.csv"
FLOW_CSV = OUTPUT_DIR / "richmond_total_treated_mgd.csv"  # flow only, no rain column

ANNUAL_DIR = OUTPUT_DIR / "figures" / "annual_daily_openmeteo"
PDF_PATH = OUTPUT_DIR / "Richmond_Yearly_Daily_Flow_OpenMeteo_Rainfall.pdf"

LATITUDE = 29.5818
LONGITUDE = -95.7608

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"


# ---------------------------------------------------------------------------
# Data: reuse the existing merged daily dataset, or build it if missing
# ---------------------------------------------------------------------------

def load_merged_daily() -> pd.DataFrame:
    if MERGED_CSV.exists():
        print(f"Reusing existing merged dataset: {MERGED_CSV}")
        return pd.read_csv(MERGED_CSV, parse_dates=["Date"])

    print(f"{MERGED_CSV} not found - building it from the flow dataset and Open-Meteo rainfall.")
    sys.path.insert(0, str(SCRIPTS_DIR))
    from monthly_flow_openmeteo_plots import get_openmeteo_rainfall, build_merged_daily  # noqa: E402

    flow_df = pd.read_csv(FLOW_CSV, parse_dates=["Date"])
    rain_df = get_openmeteo_rainfall(flow_df["Date"].min(), flow_df["Date"].max())
    merged = build_merged_daily(flow_df, rain_df)
    merged.to_csv(MERGED_CSV, index=False)
    return merged


def build_year_index(df: pd.DataFrame, year: int, dataset_max: pd.Timestamp) -> pd.DataFrame:
    """Reindex onto a continuous daily calendar index for the year so
    missing flow days appear as real NaN gaps rather than being skipped."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    if year == dataset_max.year:
        end = min(end, dataset_max)
    full_index = pd.date_range(start=start, end=end, freq="D")
    year_df = df[(df["Date"] >= start) & (df["Date"] <= end)].set_index("Date")
    return year_df.reindex(full_index)


# ---------------------------------------------------------------------------
# Plot: one continuous Jan-Dec daily timeline per year
# ---------------------------------------------------------------------------

def plot_year(year_df: pd.DataFrame, year: int) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(40, 12))

    # --- Right axis first (rainfall bars), so it can be sent behind ax1 ---
    ax2 = ax1.twinx()
    rain_valid = year_df.dropna(subset=["OpenMeteo_Rainfall_in"])
    bars = ax2.bar(rain_valid.index, rain_valid["OpenMeteo_Rainfall_in"], width=0.8,
                    color=RAIN_COLOR, alpha=0.6, label="Open-Meteo Rainfall")
    ax2.set_ylabel("Daily Rainfall (inches)", color=RAIN_LABEL_COLOR)
    ax2.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    # --- Left axis (flow line) drawn on top: dominant series, bars behind ---
    # Every daily observation gets its own visible marker, per engineering
    # review requirement - thin connecting line, small marker at each point.
    ax1.plot(year_df.index, year_df["Flow_MGD"], color=FLOW_COLOR, linewidth=0.8,
              marker="o", markersize=2.5, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
              label="Total Treated Flow", zorder=5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax1.tick_params(axis="y", labelcolor=FLOW_COLOR)
    ax1.grid(True, linewidth=0.3, alpha=0.3)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    year_start = year_df.index.min()
    year_end = year_df.index.max()
    ax1.set_xlim(year_start, year_end)

    # --- Subtle month-boundary separators + month abbreviation labels ---
    month_starts = pd.date_range(start=pd.Timestamp(year=year, month=1, day=1),
                                  end=pd.Timestamp(year=year, month=12, day=1), freq="MS")
    for ms in month_starts:
        if year_start <= ms <= year_end:
            ax1.axvline(ms, color="gray", linewidth=0.7, alpha=0.4, zorder=0)

    month_ends = month_starts + pd.offsets.MonthEnd(0)
    for ms, me in zip(month_starts, month_ends):
        span_end = min(me, year_end)
        span_start = max(ms, year_start)
        if span_start > year_end:
            continue
        mid = span_start + (span_end - span_start) / 2
        ax1.text(mid, 1.02, ms.strftime("%b").upper(), transform=ax1.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=11, color="dimgray", fontweight="bold")

    # --- Every calendar date gets its own labeled tick (no weekly reduction) ---
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha="center", fontsize=5)

    fig.suptitle(f"RRWWTF Daily Total Treated Flow and Open-Meteo Rainfall — {year}",
                 fontsize=16, y=0.995)
    fig.text(0.5, 0.955, f"Rainfall Source: Open-Meteo | Richmond, TX ({LATITUDE}, {LONGITUDE})",
             ha="center", va="top", fontsize=10, color="gray")

    lines, line_labels = ax1.get_legend_handles_labels()
    bar_handles, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bar_handles, line_labels + bar_labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ANNUAL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged_daily()
    dataset_max = df["Date"].max()

    coverage_notes = []
    generated = []

    with PdfPages(PDF_PATH) as pdf:
        for year in YEARS:
            if df[df["Date"].dt.year == year].empty:
                coverage_notes.append(f"{year}: no data in merged dataset - skipped")
                continue

            year_df = build_year_index(df, year, dataset_max)
            total_days = len(year_df)
            valid_flow_days = int(year_df["Flow_MGD"].notna().sum())
            missing_flow_days = total_days - valid_flow_days
            missing_pct = 100 * missing_flow_days / total_days if total_days else 0

            fig = plot_year(year_df, year)
            out_png = ANNUAL_DIR / f"flow_openmeteo_rainfall_daily_{year}.png"
            fig.savefig(out_png, dpi=300)
            pdf.savefig(fig, orientation="landscape")
            plt.close(fig)
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

    print()
    print("RRWWTF ANNUAL DAILY FLOW vs. OPEN-METEO RAINFALL FIGURES COMPLETE")
    print(f"Annual figures generated:         {len(generated)}")
    print(f"Years generated:                  {', '.join(str(y) for y in generated)}")
    print(f"Output directory (PNGs):          {ANNUAL_DIR}")
    print(f"Combined PDF:                     {PDF_PATH}")
    print("Coverage / plotting notes:")
    if coverage_notes:
        for note in coverage_notes:
            print(f"  - {note}")
    else:
        print("  - None; all years have complete daily flow coverage")


if __name__ == "__main__":
    main()

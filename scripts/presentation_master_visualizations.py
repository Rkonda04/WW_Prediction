"""
RRWWTF Presentation-Ready Master Dataset Figures

Reads ONLY the complete master dataset
(output/richmond_master_flow_rainfall_dataset.csv) -- not the dry-weather,
wet-weather, or antecedent-dry subsets from the separate I&I baseline
analysis -- and produces a set of presentation/report-ready figures plus a
descriptive-statistics CSV. All available daily observations are plotted;
nothing is removed, smoothed, interpolated, or aggregated. Reference lines
(mean/median/5th/95th percentile) are descriptive visual aids only, not an
outlier classification.

Usage:
    python scripts/presentation_master_visualizations.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
MASTER_CSV = OUTPUT_DIR / "00_master_dataset" / "richmond_master_flow_rainfall_dataset.csv"

PRES_DIR = OUTPUT_DIR / "03_presentation"
STATS_CSV = PRES_DIR / "master_flow_descriptive_statistics.csv"

FIG1_PATH = PRES_DIR / "01_master_daily_flow.png"
FIG2_PATH = PRES_DIR / "02_master_flow_with_statistics.png"
FIG3_PATH = PRES_DIR / "03_master_flow_with_rainfall.png"
FIG4_PATH = PRES_DIR / "04_master_flow_distribution.png"
FIG5_PATH = PRES_DIR / "05_master_flow_statistics_table.png"
FIG6_PATH = PRES_DIR / "06_monthly_flow_distribution.png"
FIG7_PATH = PRES_DIR / "07_annual_flow_trend.png"

HIGH_FLOW_THRESHOLD = 3.0

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FLOW_COLOR = "#2b6cb0"
RAIN_COLOR = "#63b3ed"
RAIN_LABEL_COLOR = "#1a5276"
MEAN_COLOR = "#c0392b"
MEDIAN_COLOR = "#27ae60"
P5_COLOR = "#8e44ad"
P95_COLOR = "#d97706"

DPI = 300
FIGSIZE_TS = (20, 8)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def fmt3(v):
    return f"{v:.3f}"


def style_ts_axes(ax, fig):
    ax.set_xlabel("Date")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)


def main():
    PRES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    earliest = df["Date"].min()
    latest = df["Date"].max()

    # =======================================================================
    # STEP 1: descriptive statistics (complete master flow, no removal)
    # =======================================================================
    valid_flow = df["Total_Treated_MGD"].dropna()
    n_valid = int(len(valid_flow))
    n_missing = int(df["Total_Treated_MGD"].isna().sum())

    mean_v = valid_flow.mean()
    median_v = valid_flow.median()
    std_v = valid_flow.std()
    min_v = valid_flow.min()
    p5 = valid_flow.quantile(0.05)
    p25 = valid_flow.quantile(0.25)
    p75 = valid_flow.quantile(0.75)
    p95 = valid_flow.quantile(0.95)
    max_v = valid_flow.max()
    iqr = p75 - p25

    n_above3 = int((valid_flow > HIGH_FLOW_THRESHOLD).sum())
    pct_above3 = (n_above3 / n_valid * 100) if n_valid else 0.0

    n_valid_rain = int(df["Rainfall_in"].notna().sum())
    n_missing_rain = int(df["Rainfall_in"].isna().sum())

    stats_rows = [
        ("Count", str(n_valid)),
        ("Missing", str(n_missing)),
        ("Mean", fmt3(mean_v)),
        ("Median", fmt3(median_v)),
        ("Standard Deviation", fmt3(std_v)),
        ("Minimum", fmt3(min_v)),
        ("5th Percentile", fmt3(p5)),
        ("25th Percentile", fmt3(p25)),
        ("75th Percentile", fmt3(p75)),
        ("95th Percentile", fmt3(p95)),
        ("Maximum", fmt3(max_v)),
        ("IQR", fmt3(iqr)),
        ("Days Above 3 MGD", str(n_above3)),
        ("Percent Above 3 MGD", f"{pct_above3:.2f}%"),
    ]
    stats_df = pd.DataFrame(stats_rows, columns=["Statistic", "Value"])
    stats_df.to_csv(STATS_CSV, index=False)

    # Full daily calendar reindex for honest gap handling in line plots
    full_range = pd.date_range(earliest, latest, freq="D")
    flow_series = df.drop_duplicates(subset="Date").set_index("Date")["Total_Treated_MGD"].reindex(full_range)

    y_pad = (max_v - min_v) * 0.05
    y_min, y_max = min_v - y_pad, max_v + y_pad

    # =======================================================================
    # GRAPH 1: complete master flow time series (no rainfall, no ref lines)
    # =======================================================================
    fig1, ax1 = plt.subplots(figsize=FIGSIZE_TS)
    ax1.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=0.7,
              marker="o", markersize=2.2, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
              label="Daily Total Treated Flow")
    ax1.set_title("Historical Daily Wastewater Flow – Richmond RRWWTF")
    ax1.set_ylabel("Total Treated Flow (MGD)")
    ax1.set_ylim(y_min, y_max)
    style_ts_axes(ax1, fig1)
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(FIG1_PATH, dpi=DPI)
    plt.close(fig1)

    # =======================================================================
    # GRAPH 2: master flow + mean/median/5th/95th reference lines
    # =======================================================================
    fig2, ax2 = plt.subplots(figsize=FIGSIZE_TS)
    ax2.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=0.7,
              marker="o", markersize=2.2, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
              label="Daily Total Treated Flow", zorder=2)
    ax2.axhline(mean_v, color=MEAN_COLOR, linestyle="--", linewidth=1.6,
                label=f"Mean = {mean_v:.3f} MGD", zorder=3)
    ax2.axhline(median_v, color=MEDIAN_COLOR, linestyle="-.", linewidth=1.6,
                label=f"Median = {median_v:.3f} MGD", zorder=3)
    ax2.axhline(p5, color=P5_COLOR, linestyle=":", linewidth=1.8,
                label=f"5th Percentile = {p5:.3f} MGD", zorder=3)
    ax2.axhline(p95, color=P95_COLOR, linestyle=":", linewidth=1.8,
                label=f"95th Percentile = {p95:.3f} MGD", zorder=3)
    ax2.set_title("Historical Wastewater Flow with Descriptive Statistics – Richmond RRWWTF")
    ax2.set_ylabel("Total Treated Flow (MGD)")
    ax2.set_ylim(y_min, y_max)
    style_ts_axes(ax2, fig2)
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(FIG2_PATH, dpi=DPI)
    plt.close(fig2)

    # =======================================================================
    # GRAPH 3: master flow + plant rainfall (dual axis)
    # =======================================================================
    rain_valid = df.dropna(subset=["Rainfall_in"])

    fig3, ax3a = plt.subplots(figsize=FIGSIZE_TS)
    ax3b = ax3a.twinx()
    # Draw rainfall (ax3b) visually behind the flow line (ax3a)
    ax3a.set_zorder(ax3b.get_zorder() + 1)
    ax3a.patch.set_visible(False)

    bars = ax3b.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=1.5,
                     color=RAIN_COLOR, alpha=0.55, label="Plant Rainfall")
    ax3b.set_ylabel("Plant Rainfall (in)", color=RAIN_LABEL_COLOR)
    ax3b.tick_params(axis="y", labelcolor=RAIN_LABEL_COLOR)

    line, = ax3a.plot(flow_series.index, flow_series.values, color=FLOW_COLOR, linewidth=0.7,
                       marker="o", markersize=2.2, markerfacecolor=FLOW_COLOR, markeredgewidth=0,
                       label="Daily Total Treated Flow")
    ax3a.set_ylabel("Total Treated Flow (MGD)", color=FLOW_COLOR)
    ax3a.tick_params(axis="y", labelcolor=FLOW_COLOR)

    ax3a.set_title("Historical Daily Wastewater Flow and Plant Rainfall – Richmond RRWWTF")
    ax3a.legend([line, bars], ["Daily Total Treated Flow", "Plant Rainfall"], loc="upper right")
    style_ts_axes(ax3a, fig3)
    fig3.tight_layout()
    fig3.savefig(FIG3_PATH, dpi=DPI)
    plt.close(fig3)

    # =======================================================================
    # GRAPH 4: flow distribution histogram with reference lines
    # =======================================================================
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    ax4.hist(valid_flow, bins=50, color=FLOW_COLOR, edgecolor="white", alpha=0.85)
    ax4.axvline(mean_v, color=MEAN_COLOR, linestyle="--", linewidth=1.6,
                label=f"Mean = {mean_v:.3f} MGD")
    ax4.axvline(median_v, color=MEDIAN_COLOR, linestyle="-.", linewidth=1.6,
                label=f"Median = {median_v:.3f} MGD")
    ax4.axvline(p5, color=P5_COLOR, linestyle=":", linewidth=1.8,
                label=f"5th Percentile = {p5:.3f} MGD")
    ax4.axvline(p95, color=P95_COLOR, linestyle=":", linewidth=1.8,
                label=f"95th Percentile = {p95:.3f} MGD")
    ax4.set_title("Distribution of Daily Wastewater Flow – Richmond RRWWTF")
    ax4.set_xlabel("Total Treated Flow (MGD)")
    ax4.set_ylabel("Number of Days")
    ax4.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(FIG4_PATH, dpi=DPI)
    plt.close(fig4)

    # =======================================================================
    # GRAPH 5: descriptive statistics summary table (PowerPoint-ready)
    # =======================================================================
    table_rows = [
        ("Count", f"{n_valid:,}"),
        ("Mean", f"{mean_v:.3f}"),
        ("Median", f"{median_v:.3f}"),
        ("Standard Deviation", f"{std_v:.3f}"),
        ("Minimum", f"{min_v:.3f}"),
        ("5th Percentile", f"{p5:.3f}"),
        ("25th Percentile", f"{p25:.3f}"),
        ("75th Percentile", f"{p75:.3f}"),
        ("95th Percentile", f"{p95:.3f}"),
        ("Maximum", f"{max_v:.3f}"),
        ("IQR", f"{iqr:.3f}"),
    ]

    fig5, ax5 = plt.subplots(figsize=(9, 7))
    ax5.axis("off")
    ax5.set_title("Master Wastewater Flow – Descriptive Statistics", fontsize=17, pad=20)

    tbl = ax5.table(
        cellText=[[label, val] for label, val in table_rows],
        colLabels=["Statistic", "Total Treated Flow (MGD)"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.0)
    tbl.auto_set_column_width([0, 1])

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(FLOW_COLOR)
        else:
            cell.set_facecolor("#f3f4f6" if r % 2 == 0 else "white")
            if c == 1:
                cell.set_text_props(ha="right")

    fig5.tight_layout()
    fig5.savefig(FIG5_PATH, dpi=DPI)
    plt.close(fig5)

    # =======================================================================
    # GRAPH 6: monthly flow distribution boxplot (all years, outliers shown)
    # =======================================================================
    box_data = [df.loc[df["Month"] == m, "Total_Treated_MGD"].dropna().values for m in range(1, 13)]

    fig6, ax6 = plt.subplots(figsize=(14, 7))
    ax6.boxplot(box_data, tick_labels=MONTH_ABBR, showfliers=True)
    ax6.set_title("Monthly Distribution of Wastewater Flow – All Years")
    ax6.set_xlabel("Month")
    ax6.set_ylabel("Total Treated Flow (MGD)")
    ax6.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig6.tight_layout()
    fig6.savefig(FIG6_PATH, dpi=DPI)
    plt.close(fig6)

    # =======================================================================
    # GRAPH 7: annual mean and median flow trend
    # =======================================================================
    annual = df.dropna(subset=["Total_Treated_MGD"]).groupby("Year")["Total_Treated_MGD"].agg(["mean", "median"])
    years = annual.index.tolist()

    fig7, ax7 = plt.subplots(figsize=(12, 7))
    ax7.plot(years, annual["mean"], color=MEAN_COLOR, linewidth=1.6, marker="o", markersize=7,
             label="Annual Mean")
    ax7.plot(years, annual["median"], color=MEDIAN_COLOR, linewidth=1.6, marker="s", markersize=7,
             label="Annual Median")
    ax7.set_xticks(years)
    ax7.set_title("Annual Wastewater Flow Trend – Richmond RRWWTF")
    ax7.set_xlabel("Year")
    ax7.set_ylabel("Total Treated Flow (MGD)")
    ax7.grid(True, linewidth=0.4, alpha=0.5)
    ax7.legend()
    fig7.tight_layout()
    fig7.savefig(FIG7_PATH, dpi=DPI)
    plt.close(fig7)

    # =======================================================================
    # Final summary
    # =======================================================================
    print("PRESENTATION MASTER-DATA ANALYSIS COMPLETE")
    print()
    print("Date range:")
    print(f"{earliest.date()} to {latest.date()}")
    print()
    print("Total master rows:")
    print(len(df))
    print()
    print("Valid flow observations:")
    print(n_valid)
    print()
    print("Missing flow observations:")
    print(n_missing)
    print()
    print("Valid rainfall observations:")
    print(n_valid_rain)
    print()
    print("Missing rainfall observations:")
    print(n_missing_rain)
    print()
    print("Flow statistics:")
    print(f"  Mean:               {mean_v:.3f} MGD")
    print(f"  Median:             {median_v:.3f} MGD")
    print(f"  Standard Deviation: {std_v:.3f} MGD")
    print(f"  Minimum:            {min_v:.3f} MGD")
    print(f"  5th Percentile:     {p5:.3f} MGD")
    print(f"  25th Percentile:    {p25:.3f} MGD")
    print(f"  75th Percentile:    {p75:.3f} MGD")
    print(f"  95th Percentile:    {p95:.3f} MGD")
    print(f"  Maximum:            {max_v:.3f} MGD")
    print(f"  IQR:                {iqr:.3f} MGD")
    print()
    print(f"Days > {HIGH_FLOW_THRESHOLD} MGD:")
    print(n_above3)
    print()
    print(f"Percentage of valid days > {HIGH_FLOW_THRESHOLD} MGD:")
    print(f"{pct_above3:.2f}%")
    print()
    print("Files created in output/presentation/:")
    for p in [FIG1_PATH, FIG2_PATH, FIG3_PATH, FIG4_PATH, FIG5_PATH, FIG6_PATH, FIG7_PATH, STATS_CSV]:
        print(f"  {p.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()

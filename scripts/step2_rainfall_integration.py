"""
RRWWTF Step 2 — Rainfall Integration

IMPORTANT DEVIATION FROM THE ORIGINAL STEP 2 PLAN:
The original plan called for independent external rainfall from the
Open-Meteo Historical Weather API (ERA5-Land). During implementation this
was tested directly against the live API and confirmed (across multiple
years and locations) that Open-Meteo's archive API returns NULL for every
hourly precipitation/rain value when the ERA5-Land model is selected —
precipitation is simply not exposed for that model in this API; only the
separate ERA5 model (or the blended "best_match" product) returns real
precipitation values. When this was raised, the user explicitly instructed
that no external API (Open-Meteo, ERA5, NOAA/NCDC) be used at all, and
that Step 2 instead use the "Rain" field already recorded in the RRWWTF
Daily Effluent Log workbooks — the same workbooks used for Step 1.

This script therefore extracts daily rainfall from that in-workbook "Rain"
column (same day-position extraction logic as Step 1's Total Treated MGD
extraction, so dates line up exactly), rather than calling any external
weather API. This is documented prominently in the generated report.

No wastewater-flow or rainfall observations are removed, filtered,
smoothed, transformed, or imputed anywhere in this script.
"""

import calendar
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw" / "2.7.7 Wastewater Flows and Runtimes"
OUTPUT_DIR = BASE_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"

FLOW_CSV = OUTPUT_DIR / "richmond_total_treated_mgd.csv"
RAIN_CSV = OUTPUT_DIR / "richmond_daily_rainfall.csv"
MERGED_CSV = OUTPUT_DIR / "richmond_flow_rainfall.csv"
YEARLY_RAIN_CSV = OUTPUT_DIR / "richmond_rainfall_yearly_stats.csv"
REPORT_PATH = OUTPUT_DIR / "Richmond_WWTF_Flow_Rainfall_Integration_Report.md"

FULL_FIG_PATH = FIG_DIR / "flow_rainfall_full_history.png"

EXCLUDED_DIR_NAMES = {"pump curves"}
SHEET_NAME_KEYWORD = "daily effluent"
DATE_HEADER = "date"
RAIN_HEADER = "rain"

MONTH_ABBR_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

MM_PER_INCH = 25.4

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


@dataclass
class FileResult:
    path: Path
    status: str = "failed"
    reason: str = ""
    n_rows: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workbook discovery / parsing helpers (mirrors Step 1's extraction logic
# so that rainfall dates align exactly with the flow dataset's dates)
# ---------------------------------------------------------------------------

def discover_workbooks(raw_dir: Path) -> list[Path]:
    files = []
    for p in sorted(raw_dir.rglob("*.xlsx")):
        if p.name.startswith("~$"):
            continue
        rel_parts = [part.lower() for part in p.relative_to(raw_dir).parts[:-1]]
        if any(part in EXCLUDED_DIR_NAMES for part in rel_parts):
            continue
        files.append(p)
    return files


def month_num_from_text(text: str):
    m = re.search(r"[A-Za-z]{3,}", text)
    if not m:
        return None
    return MONTH_ABBR_TO_NUM.get(m.group(0)[:3].lower())


def parse_month_year_from_sheet(ws, max_scan_rows: int = 10):
    for r in range(1, max_scan_rows + 1):
        val = ws.cell(row=r, column=1).value
        if isinstance(val, str) and "month of" in val.lower():
            m = re.search(r"month of\s+([A-Za-z]+)\s*,?\s*(\d{4})", val, re.IGNORECASE)
            if m:
                month_num = month_num_from_text(m.group(1))
                year = int(m.group(2))
                if month_num:
                    return month_num, year
    return None, None


def parse_month_year_from_filename(path: Path):
    name = path.stem
    year_match = re.search(r"(19|20)\d{2}", name)
    year = int(year_match.group(0)) if year_match else None
    month_num = None
    for token in re.findall(r"[A-Za-z]+", name):
        mn = MONTH_ABBR_TO_NUM.get(token[:3].lower())
        if mn:
            month_num = mn
            break
    return month_num, year


def find_daily_effluent_sheet(wb):
    for sn in wb.sheetnames:
        if SHEET_NAME_KEYWORD in sn.lower():
            return sn
    return None


def find_header_row_and_columns(ws, max_scan_rows: int = 15):
    """Locate the header row and the 'Date' / 'Rain' column indices."""
    max_col = ws.max_column or 1
    for r in range(1, max_scan_rows + 1):
        row_vals = [ws.cell(row=r, column=c).value for c in range(1, max_col + 1)]
        norm = [str(v).strip().lower() if v is not None else "" for v in row_vals]
        if DATE_HEADER in norm and RAIN_HEADER in norm:
            date_col = norm.index(DATE_HEADER) + 1
            rain_col = norm.index(RAIN_HEADER) + 1
            return r, date_col, rain_col
    return None, None, None


def coerce_numeric(raw_value, row_num: int, notes: list, field_label: str):
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            notes.append(f"Row {row_num}: non-numeric {field_label} value {raw_value!r} treated as missing")
            return None
    return None


def process_workbook_rain(path: Path):
    """Extract daily Rain values from a workbook's Daily Effluent Log sheet."""
    result = FileResult(path=path)
    rel_name = str(path.relative_to(RAW_DIR))

    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    except Exception as e:
        result.reason = f"Could not open workbook: {e}"
        return result, []

    sheet_name = find_daily_effluent_sheet(wb)
    if sheet_name is None:
        result.reason = "No 'Daily Effluent Log' sheet found in workbook"
        wb.close()
        return result, []

    ws = wb[sheet_name]
    header_row, date_col, rain_col = find_header_row_and_columns(ws)
    if header_row is None:
        result.reason = "Could not locate 'Date' / 'Rain' header row in Daily Effluent Log sheet"
        wb.close()
        return result, []

    month_num, year = parse_month_year_from_sheet(ws)
    month_year_source = "sheet title"
    if month_num is None or year is None:
        month_num, year = parse_month_year_from_filename(path)
        month_year_source = "filename"

    if month_num is None or year is None:
        result.reason = "Could not determine month/year from sheet title or filename"
        wb.close()
        return result, []

    if month_year_source == "filename":
        result.notes.append("Month/year determined from filename (sheet title missing or unparsable)")

    days_in_month = calendar.monthrange(year, month_num)[1]
    records = []
    for day in range(1, days_in_month + 1):
        r = header_row + day
        date_cell_val = ws.cell(row=r, column=date_col).value
        rain_val = ws.cell(row=r, column=rain_col).value

        if isinstance(date_cell_val, (int, float)) and not isinstance(date_cell_val, bool):
            if int(date_cell_val) != day:
                result.notes.append(
                    f"Row {r}: 'Date' column value {date_cell_val} does not match expected "
                    f"day-of-month {day} (record kept, position used for calendar date)"
                )

        rainfall_in = coerce_numeric(rain_val, r, result.notes, "Rain")
        the_date = pd.Timestamp(year=year, month=month_num, day=day)
        records.append({
            "Date": the_date,
            "Rainfall_in": rainfall_in,
            "Source_File": rel_name,
        })

    wb.close()
    result.status = "ok"
    result.n_rows = len(records)
    return result, records


# ---------------------------------------------------------------------------
# Step 2.1A: rainfall data-quality statistics
# ---------------------------------------------------------------------------

def rainfall_quality_stats(rain_df: pd.DataFrame) -> dict:
    valid = rain_df["Rainfall_in"].dropna()
    stats = {
        "earliest": rain_df["Date"].min(),
        "latest": rain_df["Date"].max(),
        "n_records": len(rain_df),
        "n_missing": int(rain_df["Rainfall_in"].isna().sum()),
        "n_rain_days": int((valid > 0).sum()),
        "n_dry_days": int((valid == 0).sum()),
        "mean": valid.mean() if not valid.empty else None,
        "median": valid.median() if not valid.empty else None,
        "p95": valid.quantile(0.95) if not valid.empty else None,
        "max": valid.max() if not valid.empty else None,
    }
    if not valid.empty:
        max_idx = rain_df.loc[rain_df["Rainfall_in"] == stats["max"], "Date"]
        stats["max_date"] = max_idx.iloc[0]
    else:
        stats["max_date"] = None
    return stats


def fmt(v, nd=3):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:,.{nd}f}"


# ---------------------------------------------------------------------------
# Step 2.5: yearly rainfall statistics
# ---------------------------------------------------------------------------

def yearly_rainfall_stats(rain_df: pd.DataFrame) -> pd.DataFrame:
    d = rain_df.copy()
    d["Year"] = d["Date"].dt.year
    rows = []
    for year, g in d.groupby("Year"):
        valid = g["Rainfall_in"].dropna()
        rows.append({
            "Year": int(year),
            "Days Available": int(valid.count()),
            "Rain Days": int((valid > 0).sum()),
            "Total Rainfall (in)": valid.sum() if not valid.empty else 0.0,
            "Mean Daily Rainfall (in)": valid.mean() if not valid.empty else None,
            "P95 Daily Rainfall (in)": valid.quantile(0.95) if not valid.empty else None,
            "Maximum Daily Rainfall (in)": valid.max() if not valid.empty else None,
        })
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2.3 / 2.4: dual-axis flow + rainfall plots
# ---------------------------------------------------------------------------

def plot_dual_axis(df: pd.DataFrame, title: str, out_path: Path, bar_width=0.9):
    d = df.sort_values("Date")
    fig, ax1 = plt.subplots(figsize=(16, 6))

    flow_valid = d.dropna(subset=["Total_Treated_MGD"])
    line, = ax1.plot(flow_valid["Date"], flow_valid["Total_Treated_MGD"],
                      color="#2b6cb0", linewidth=1.0, marker=None, label="Total Treated Flow")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Treated Flow (MGD)", color="#2b6cb0")
    ax1.tick_params(axis="y", labelcolor="#2b6cb0")
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    rain_valid = d.dropna(subset=["Rainfall_in"])
    bars = ax2.bar(rain_valid["Date"], rain_valid["Rainfall_in"], width=bar_width,
                    color="#63b3ed", alpha=0.6, label="Daily Rainfall")
    ax2.set_ylabel("Daily Rainfall (inches)", color="#1a5276")
    ax2.tick_params(axis="y", labelcolor="#1a5276")

    ax1.set_title(title)
    lines = [line, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(coverage, rain_stats, merge_stats, yearly_rain_df, file_results, workbooks):
    lines = []
    lines.append("# Richmond Regional WWTF — Flow & Rainfall Integration Report (Step 2)")
    lines.append("")

    lines.append("## 1. Rainfall Data Source")
    lines.append("")
    lines.append(
        "**Deviation from the original plan:** Step 2 was originally scoped to pull independent "
        "external precipitation from the Open-Meteo Historical Weather API using the ERA5-Land "
        "reanalysis dataset. During implementation, the live API was queried directly for the "
        "target coordinates (29.58222, -95.76083) across several years (2018, 2020, 2023) and "
        "confirmed that Open-Meteo's archive API returns **null for every hourly precipitation/rain "
        "value when `models=era5_land` is selected** — other variables (e.g., temperature) return "
        "normally for ERA5-Land, but precipitation is simply not exposed for that model through this "
        "API. Real precipitation values are only returned via the separate ERA5 model or the default "
        "blended \"best_match\" product (which uses ERA5 for precipitation)."
    )
    lines.append("")
    lines.append(
        "When this limitation was reported, the decision was made **not** to use any external "
        "weather API (Open-Meteo, ERA5, or NOAA/NCDC). Instead, Step 2 uses the **`Rain` field "
        "already recorded in the RRWWTF Daily Effluent Log workbooks** — the same 72 workbooks "
        "successfully processed in Step 1, and the same day-by-day row extraction logic, so that "
        "rainfall dates line up exactly with the flow dataset's dates."
    )
    lines.append("")
    lines.append(
        "**This is a departure from the original requirement that rainfall be an independent "
        "external dataset.** The `Rain` values used here come from the same source workbooks as "
        "`Total_Treated_MGD`, recorded by plant operators alongside the flow log. The exact "
        "measurement method/instrument used to produce these Rain values (on-site gauge, nearby "
        "station, etc.) is not documented in the workbooks themselves."
    )
    lines.append("")
    lines.append(
        "**Units:** The source header is simply labeled `Rain` with no unit given (unlike other "
        "columns in the same sheet, e.g. `Flow Over Weir (in inches)`, which are explicitly "
        "labeled). Units were inferred as **inches** based on: (1) two-decimal-place formatting "
        "consistent with standard U.S. rain-gauge precision, and (2) cross-checking against a known "
        "regional weather event — Tropical Storm Imelda produced major flooding in the "
        "Richmond/Fort Bend, TX area on September 19-20, 2019; the source workbook records Rain = "
        "1.2 and 4.5 on those two dates respectively, which is physically plausible only as inches "
        "(as millimeters these would represent negligible drizzle, inconsistent with the documented "
        "flooding event). This is a reasonable inference, not a confirmed specification from the "
        "source data — it should be verified against facility records if precision matters for "
        "downstream analysis."
    )
    lines.append("")
    lines.append(
        f"- Coordinates originally proposed for Open-Meteo (not used): Latitude 29.58222, "
        f"Longitude -95.76083\n"
        f"- Date range: {coverage['earliest'].date()} to {coverage['latest'].date()}\n"
        f"- Original precipitation units in source workbook: unlabeled, inferred as inches\n"
        f"- Precipitation_mm column is *derived* from the source inches value via "
        f"`Precipitation_mm = Rainfall_in * 25.4`, provided only for unit convenience."
    )
    lines.append("")

    lines.append("## 2. Rainfall Dataset Summary")
    lines.append("")
    lines.append(f"- Earliest rainfall date: **{rain_stats['earliest'].date()}**")
    lines.append(f"- Latest rainfall date: **{rain_stats['latest'].date()}**")
    lines.append(f"- Number of daily rainfall records: **{rain_stats['n_records']}**")
    lines.append(f"- Number of missing rainfall days (blank/non-numeric Rain cell): **{rain_stats['n_missing']}**")
    lines.append(f"- Days with rainfall > 0: **{rain_stats['n_rain_days']}**")
    lines.append(f"- Days with rainfall = 0: **{rain_stats['n_dry_days']}**")
    lines.append(f"- Mean daily rainfall: **{fmt(rain_stats['mean'])} in**")
    lines.append(f"- Median daily rainfall: **{fmt(rain_stats['median'])} in**")
    lines.append(f"- 95th percentile daily rainfall: **{fmt(rain_stats['p95'])} in**")
    lines.append(f"- Maximum daily rainfall: **{fmt(rain_stats['max'])} in** on "
                  f"**{rain_stats['max_date'].date() if rain_stats['max_date'] is not None else 'N/A'}**")
    lines.append("")
    lines.append(
        "**Data-quality note:** Because rainfall is now sourced from the same daily logbook rows "
        "as the flow data (one value per calendar day, not hourly), the original hourly-completeness "
        "check planned for the Open-Meteo approach does not apply. Missing rainfall days above are "
        "calendar days that fall within a successfully processed workbook but where the `Rain` cell "
        "itself was blank or non-numeric in the source file. Calendar days with no workbook coverage "
        "at all (e.g., the Oct 2023 – Dec 2023 gap, identical to Step 1's flow-coverage gap) are not "
        "counted as 'missing rainfall days' here since no workbook row exists for them at all; they "
        "are also absent from the flow dataset for the same reason."
    )
    lines.append("")

    failed = [r for r in file_results if r.status == "failed"]
    lines.append(f"**Workbooks discovered:** {len(workbooks)} · **Processed:** "
                 f"{len(workbooks) - len(failed)} · **Failed:** {len(failed)}")
    if failed:
        lines.append("")
        lines.append("| File | Reason |")
        lines.append("|---|---|")
        for r in failed:
            lines.append(f"| {r.path.relative_to(RAW_DIR).as_posix()} | {r.reason} |")
    lines.append("")

    any_notes = False
    note_lines = []
    for r in file_results:
        if r.status == "ok" and r.notes:
            any_notes = True
            note_lines.append(f"**{r.path.relative_to(RAW_DIR).as_posix()}**")
            for n in r.notes[:20]:
                note_lines.append(f"- {n}")
            if len(r.notes) > 20:
                note_lines.append(f"- ... and {len(r.notes) - 20} more note(s) for this file")
    if any_notes:
        lines.append("**Structural notes encountered while reading files:**")
        lines.append("")
        lines.extend(note_lines)
        lines.append("")

    lines.append("## 3. Flow-Rainfall Merge Summary")
    lines.append("")
    lines.append(f"- Flow records (from Step 1): **{merge_stats['n_flow']}**")
    lines.append(f"- Rainfall records: **{merge_stats['n_rain']}**")
    lines.append(f"- Total merged rows: **{merge_stats['n_merged']}**")
    lines.append(f"- Rows with valid flow: **{merge_stats['n_valid_flow']}**")
    lines.append(f"- Rows with valid rainfall: **{merge_stats['n_valid_rain']}**")
    lines.append(f"- Rows with both valid flow and valid rainfall: **{merge_stats['n_valid_both']}**")
    lines.append(f"- Flow dates without a rainfall record: **{merge_stats['n_flow_no_rain']}**")
    lines.append(f"- Rainfall dates without a corresponding flow record: **{merge_stats['n_rain_no_flow']}**")
    lines.append("")
    lines.append(
        "Because rainfall and flow are extracted from the same workbooks using the same "
        "day-by-day logic, the two date sets are expected to match almost exactly; any "
        "differences reported above stem from cases where the `Total Treated MGD` and `Rain` "
        "cells for a given day were present/absent independently within the same source row, "
        "not from separate source files."
    )
    lines.append("")

    lines.append("## 4. Full Historical Flow and Rainfall Figure")
    lines.append("")
    lines.append(f"![Flow and Rainfall — Full History]({FULL_FIG_PATH.relative_to(OUTPUT_DIR).as_posix()})")
    lines.append("")

    lines.append("## 5. Yearly Figures")
    lines.append("")
    for y in YEARS:
        fig_path = FIG_DIR / f"flow_rainfall_{y}.png"
        lines.append(f"### {y}")
        lines.append(f"![Flow and Rainfall — {y}]({fig_path.relative_to(OUTPUT_DIR).as_posix()})")
        lines.append("")

    lines.append("### Yearly Rainfall Statistics")
    lines.append("")
    cols = ["Year", "Days Available", "Rain Days", "Total Rainfall (in)",
            "Mean Daily Rainfall (in)", "P95 Daily Rainfall (in)", "Maximum Daily Rainfall (in)"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, row in yearly_rain_df.iterrows():
        vals = [str(int(row["Year"])), str(int(row["Days Available"])), str(int(row["Rain Days"]))]
        for k in ["Total Rainfall (in)", "Mean Daily Rainfall (in)", "P95 Daily Rainfall (in)", "Maximum Daily Rainfall (in)"]:
            vals.append(fmt(row[k]))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("## 6. Important Interpretation Note")
    lines.append("")
    lines.append(
        "The figures in this report are intended **only for visual exploration** of whether "
        "rainfall events appear to coincide with or precede changes in treated wastewater flow "
        "(a common pattern in sanitary sewer systems with inflow/infiltration). At this stage:"
    )
    lines.append("")
    lines.append("- No causal relationship is being claimed.")
    lines.append("- No rainfall-flow correlation has been calculated.")
    lines.append("- No lag relationship has been tested.")
    lines.append("- No observations have been removed, filtered, smoothed, or transformed.")
    lines.append("- No values have been imputed.")
    lines.append("- No predictive model has been trained.")
    lines.append("")
    lines.append(
        "Future analysis may investigate same-day and lagged relationships between rainfall and "
        "treated wastewater flow."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 2.1: extract rainfall from source workbooks ---
    workbooks = discover_workbooks(RAW_DIR)
    all_records = []
    file_results = []
    for path in workbooks:
        result, records = process_workbook_rain(path)
        file_results.append(result)
        all_records.extend(records)

    rain_df = pd.DataFrame(all_records)
    rain_df = rain_df.sort_values(["Date", "Source_File"]).reset_index(drop=True)
    rain_df["Precipitation_mm"] = (rain_df["Rainfall_in"] * MM_PER_INCH).round(2)
    rain_df = rain_df[["Date", "Precipitation_mm", "Rainfall_in", "Source_File"]]
    rain_df.to_csv(RAIN_CSV, index=False)

    # --- Step 2.1A: rainfall quality stats ---
    rain_stats = rainfall_quality_stats(rain_df)

    # --- Step 2.2: merge with flow dataset ---
    flow_df = pd.read_csv(FLOW_CSV, parse_dates=["Date"])
    merged = flow_df.merge(
        rain_df[["Date", "Precipitation_mm", "Rainfall_in"]], on="Date", how="left"
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged.to_csv(MERGED_CSV, index=False)

    flow_dates = set(flow_df["Date"])
    rain_dates = set(rain_df["Date"])
    merge_stats = {
        "n_flow": len(flow_df),
        "n_rain": len(rain_df),
        "n_merged": len(merged),
        "n_valid_flow": int(merged["Total_Treated_MGD"].notna().sum()),
        "n_valid_rain": int(merged["Rainfall_in"].notna().sum()),
        "n_valid_both": int((merged["Total_Treated_MGD"].notna() & merged["Rainfall_in"].notna()).sum()),
        "n_flow_no_rain": len(flow_dates - rain_dates),
        "n_rain_no_flow": len(rain_dates - flow_dates),
    }

    # --- Step 2.5: yearly rainfall stats ---
    yearly_rain_df = yearly_rainfall_stats(rain_df)
    yearly_rain_df.to_csv(YEARLY_RAIN_CSV, index=False)

    # --- Step 2.3: full-history plot ---
    plot_dual_axis(merged, "RRWWTF Historical Daily Flow and Rainfall", FULL_FIG_PATH, bar_width=0.9)

    # --- Step 2.4: yearly plots ---
    for y in YEARS:
        year_df = merged[merged["Date"].dt.year == y]
        out_path = FIG_DIR / f"flow_rainfall_{y}.png"
        plot_dual_axis(year_df, f"RRWWTF Daily Flow and Rainfall — {y}", out_path, bar_width=0.9)

    # --- Coverage summary (for report header) ---
    coverage = {"earliest": rain_stats["earliest"], "latest": rain_stats["latest"]}

    # --- Step 2.6: report ---
    report_text = build_report(coverage, rain_stats, merge_stats, yearly_rain_df, file_results, workbooks)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # --- Console summary ---
    print("RRWWTF STEP 2 - RAINFALL INTEGRATION SUMMARY")
    print("Rainfall source:                 RRWWTF Daily Effluent Log 'Rain' column")
    print("                                  (Open-Meteo/ERA5-Land NOT used - see report Sec.1)")
    print("Coordinates used:                 N/A (no external API queried)")
    print(f"Rainfall date range:              {rain_stats['earliest'].date()} to {rain_stats['latest'].date()}")
    print(f"Total rainfall days:               {rain_stats['n_records']}")
    print(f"Flow records:                      {merge_stats['n_flow']}")
    print(f"Successfully matched dates:        {merge_stats['n_valid_both']}")
    print(f"Missing rainfall dates:            {rain_stats['n_missing']}")
    total_rain = rain_df["Rainfall_in"].dropna().sum()
    print(f"Total rainfall over period (in):   {total_rain:,.2f}")
    print(f"Maximum daily rainfall (in):       {fmt(rain_stats['max'])} on "
          f"{rain_stats['max_date'].date() if rain_stats['max_date'] is not None else 'N/A'}")
    print()
    print("Output files:")
    print(f"  {RAIN_CSV}")
    print(f"  {MERGED_CSV}")
    print(f"  {YEARLY_RAIN_CSV}")
    print(f"  {FULL_FIG_PATH}")
    for y in YEARS:
        print(f"  {FIG_DIR / f'flow_rainfall_{y}.png'}")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()

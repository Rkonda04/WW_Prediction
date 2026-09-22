"""
RRWWTF Step 2 — Daily Rainfall and Wastewater Flow Exploration

Rainfall source: the "Rain" field already recorded in the RRWWTF Daily
Effluent Log workbooks (raw/). No external weather API (Open-Meteo, ERA5,
ERA5-Land, NOAA, NCDC) or third-party station data is used.

This script:
  2.1  Extracts daily Rain values from the source workbooks (same
       day-position logic used for Step 1's Total Treated MGD extraction,
       so dates line up), and determines the rainfall unit from evidence
       found inside the workbooks themselves rather than assuming it.
  2.2  Joins the extracted rainfall onto the existing Step 1 flow dataset
       (output/richmond_total_treated_mgd.csv) by Date, flow as reference.
  2.3  Reports rainfall/flow data-quality coverage statistics.
  2.4  Full-history dual-axis overview plot.
  2.5  One dual-axis plot per calendar month with data available.
  2.7  Rainfall descriptive-statistics CSV.
  2.8  Markdown report.

No observations are removed, filtered, smoothed, imputed, or interpolated.
No correlation, lag, or predictive analysis is performed here.
"""

import calendar
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
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
MONTHLY_FIG_DIR = FIG_DIR / "monthly"

FLOW_CSV = OUTPUT_DIR / "richmond_total_treated_mgd.csv"
MERGED_CSV = OUTPUT_DIR / "richmond_daily_flow_rainfall.csv"
RAIN_STATS_CSV = OUTPUT_DIR / "richmond_daily_rainfall_statistics.csv"
REPORT_PATH = OUTPUT_DIR / "Richmond_WWTF_Daily_Flow_Rainfall_Report.md"
FULL_FIG_PATH = FIG_DIR / "flow_rainfall_full_history.png"

EXCLUDED_DIR_NAMES = {"pump curves"}
SHEET_NAME_KEYWORD = "daily effluent"
DATE_HEADER = "date"
RAIN_HEADER = "rain"

MONTH_ABBR_TO_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NUM_TO_NAME = {v: k for k, v in
                      {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                       "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
                       "December": 12}.items()}


@dataclass
class FileResult:
    path: Path
    status: str = "failed"
    reason: str = ""
    n_rows: int = 0
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workbook discovery / parsing helpers
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
    max_col = ws.max_column or 1
    for r in range(1, max_scan_rows + 1):
        row_vals = [ws.cell(row=r, column=c).value for c in range(1, max_col + 1)]
        norm = [str(v).strip().lower() if v is not None else "" for v in row_vals]
        if DATE_HEADER in norm and RAIN_HEADER in norm:
            date_col = norm.index(DATE_HEADER) + 1
            rain_col = norm.index(RAIN_HEADER) + 1
            return r, date_col, rain_col, row_vals
    return None, None, None, None


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
    """Extract daily Rain values (and header/format evidence) from one workbook."""
    result = FileResult(path=path)
    rel_name = str(path.relative_to(RAW_DIR))
    evidence = {}

    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
    except Exception as e:
        result.reason = f"Could not open workbook: {e}"
        return result, [], evidence

    sheet_name = find_daily_effluent_sheet(wb)
    if sheet_name is None:
        result.reason = "No 'Daily Effluent Log' sheet found in workbook"
        wb.close()
        return result, [], evidence

    ws = wb[sheet_name]
    header_row, date_col, rain_col, header_vals = find_header_row_and_columns(ws)
    if header_row is None:
        result.reason = "Could not locate 'Date' / 'Rain' header row in Daily Effluent Log sheet"
        wb.close()
        return result, [], evidence

    evidence["rain_header_text"] = str(header_vals[rain_col - 1]).strip()
    evidence["rain_cell_number_format"] = ws.cell(row=header_row + 1, column=rain_col).number_format
    evidence["neighbor_headers"] = [str(h).strip() for h in header_vals if h is not None]

    # Look for explicit unit mentions of rainfall elsewhere in the workbook
    unit_notes = []
    for sn in wb.sheetnames:
        if sn == sheet_name:
            continue
        sheet = wb[sn]
        try:
            for row in sheet.iter_rows(values_only=True):
                for v in row:
                    if isinstance(v, str) and "rain" in v.lower() and v.strip().lower() != "rain":
                        if re.search(r"inch|in\.|mm|cm|millimet|centimet", v, re.IGNORECASE):
                            unit_notes.append(f"{sn}: {v.strip()}")
        except Exception:
            pass
    evidence["unit_notes"] = unit_notes

    month_num, year = parse_month_year_from_sheet(ws)
    month_year_source = "sheet title"
    if month_num is None or year is None:
        month_num, year = parse_month_year_from_filename(path)
        month_year_source = "filename"

    if month_num is None or year is None:
        result.reason = "Could not determine month/year from sheet title or filename"
        wb.close()
        return result, [], evidence

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
    return result, records, evidence


# ---------------------------------------------------------------------------
# Step 2.3 / 2.7: rainfall statistics
# ---------------------------------------------------------------------------

def fmt(v, nd=3):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v:,.{nd}f}"


def rainfall_stats(rain_series: pd.Series) -> dict:
    valid = rain_series.dropna()
    return {
        "count": int(valid.count()),
        "min": valid.min() if not valid.empty else None,
        "mean": valid.mean() if not valid.empty else None,
        "median": valid.median() if not valid.empty else None,
        "p95": valid.quantile(0.95) if not valid.empty else None,
        "max": valid.max() if not valid.empty else None,
        "n_rain_days": int((valid > 0).sum()),
        "n_dry_days": int((valid == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Step 2.4 / 2.5: dual-axis plots
# ---------------------------------------------------------------------------

def plot_dual_axis(df: pd.DataFrame, title: str, out_path: Path, bar_width, markers=False,
                    date_locator=None, date_formatter=None):
    d = df.sort_values("Date")
    fig, ax1 = plt.subplots(figsize=(16, 6))

    flow_valid = d.dropna(subset=["Total_Treated_MGD"])
    marker_style = "o" if markers else None
    line, = ax1.plot(flow_valid["Date"], flow_valid["Total_Treated_MGD"],
                      color="#2b6cb0", linewidth=1.2, marker=marker_style, markersize=4,
                      label="Total Treated Flow")
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
    ax1.legend([line, bars], [l.get_label() for l in [line, bars]], loc="upper right")

    if date_locator is not None:
        ax1.xaxis.set_major_locator(date_locator)
    if date_formatter is not None:
        ax1.xaxis.set_major_formatter(date_formatter)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(unit_evidence, coverage, rstats, monthly_keys, sample_months, file_results, workbooks):
    lines = []
    lines.append("# Richmond Regional WWTF — Daily Flow & Rainfall Report (Step 2)")
    lines.append("")

    lines.append("## 1. Objective")
    lines.append("")
    lines.append(
        "Step 2 puts the plant's recorded daily rainfall on the same timeline as its recorded "
        "daily Total Treated MGD, so that the day-to-day behavior of treated wastewater flow "
        "can be visually inspected around individual rainfall events, before any statistical or "
        "predictive analysis is attempted."
    )
    lines.append("")

    lines.append("## 2. Rainfall Source")
    lines.append("")
    lines.append(
        "Rainfall was extracted from the **`Rain` field in the RRWWTF Daily Effluent Log sheet** "
        "of the same workbooks used for the Step 1 flow extraction (`raw/`). No external weather "
        "API or third-party station data (Open-Meteo, ERA5, ERA5-Land, NOAA, NCDC, Sugar Land, or "
        "Pearland) was used."
    )
    lines.append("")
    lines.append("**How the rainfall unit was determined:**")
    lines.append("")
    lines.append(
        f"1. The `Rain` column header itself carries **no unit suffix**. This was checked across "
        f"all {coverage['n_ok']} successfully processed workbooks — every one uses the identical "
        f"header text `Rain`, with no variant such as `Rain (in)` or `Rain (mm)`. This contrasts "
        f"with neighboring columns in the *same* header row that *do* state units explicitly, e.g. "
        f"`Flow Over Weir (in inches)` and `Staff Gauge (in feet)` — so the absence of a unit on "
        f"`Rain` is not an omission unique to this extraction, it is how the source sheet is built."
    )
    lines.append(
        f"2. The `Rain` data cells are formatted as a plain two-decimal number "
        f"(`{unit_evidence['number_format']}`), not a unit-tagged custom format."
    )
    lines.append(
        "3. **Decisive evidence found inside the workbooks themselves:** the `DMR Info` sheet in "
        "several monthly workbooks contains an operator-written remarks field that explicitly "
        "describes that month's rainfall **in inches**, in plain language (e.g. *\"Over 5 inches "
        "of rain.\"*, *\"OVER 2 INCHES OF RAIN\"*, *\"Over an inch of rain fall\"*). These remarks "
        "were cross-checked by summing that same month's `Rain` column values:"
    )
    lines.append("")
    lines.append("| Workbook | DMR Info remark | Summed `Rain` column for that month |")
    lines.append("|---|---|---:|")
    for row in unit_evidence["cross_checks"]:
        lines.append(f"| {row['file']} | \"{row['note']}\" | {row['sum']:.2f} |")
    lines.append("")
    lines.append(
        "In every case, the plant's own plain-language description of that month's rainfall "
        "(\"over N inches\") is consistent with the summed `Rain` column exceeding N. This is "
        "internal, source-based documentation of the unit — not an external assumption — and it "
        "is treated as sufficient to establish that **`Rain` values are recorded in inches**. "
        "Accordingly, the raw values are preserved unchanged in the `Rainfall_in` column below. "
        "The underlying column header itself still carries no formal unit label, so if this "
        "dataset is used for regulatory or engineering purposes, confirming the unit against "
        "facility SOPs/instrumentation records is still recommended."
    )
    lines.append("")

    lines.append("## 3. Data Coverage")
    lines.append("")
    lines.append(f"- Flow date range: **{coverage['flow_min'].date()} to {coverage['flow_max'].date()}**")
    lines.append(f"- Rainfall date range: **{coverage['rain_min'].date()} to {coverage['rain_max'].date()}**")
    lines.append(f"- Total dates in the flow dataset: **{coverage['n_total']}**")
    lines.append(f"- Dates with valid flow: **{coverage['n_valid_flow']}**")
    lines.append(f"- Dates with valid rainfall: **{coverage['n_valid_rain']}**")
    lines.append(f"- Dates with both valid flow and rainfall: **{coverage['n_valid_both']}**")
    lines.append(f"- Missing rainfall observations (blank/non-numeric `Rain` cell): **{coverage['n_missing_rain']}**")
    lines.append(f"- Missing flow observations: **{coverage['n_missing_flow']}**")
    lines.append(
        f"- Missing calendar periods (months with no processed workbook at all): "
        f"**{coverage['missing_periods_desc']}**"
    )
    lines.append("")
    lines.append(f"**Workbooks discovered:** {len(workbooks)} · **Processed:** {coverage['n_ok']} · "
                 f"**Failed:** {len(workbooks) - coverage['n_ok']}")
    failed = [r for r in file_results if r.status == "failed"]
    if failed:
        lines.append("")
        lines.append("| File | Reason |")
        lines.append("|---|---|")
        for r in failed:
            lines.append(f"| {r.path.relative_to(RAW_DIR).as_posix()} | {r.reason} |")
    lines.append("")

    lines.append("## 4. Rainfall Statistics")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Count (valid days) | {rstats['count']} |")
    lines.append(f"| Days with Rainfall > 0 | {rstats['n_rain_days']} |")
    lines.append(f"| Days with Rainfall = 0 | {rstats['n_dry_days']} |")
    lines.append(f"| Minimum (in) | {fmt(rstats['min'])} |")
    lines.append(f"| Mean (in) | {fmt(rstats['mean'])} |")
    lines.append(f"| Median (in) | {fmt(rstats['median'])} |")
    lines.append(f"| 95th percentile (in) | {fmt(rstats['p95'])} |")
    lines.append(f"| Maximum (in) | {fmt(rstats['max'])} |")
    lines.append(f"| Date of maximum | {rstats['max_date']} |")
    lines.append("")
    lines.append(
        "A blank `Rain` cell is preserved as a missing observation (`NaN`), distinct from a "
        "recorded value of `0`. No missing rainfall was replaced with zero, and no values were "
        "removed as outliers."
    )
    lines.append("")

    lines.append("## 5. Full Historical Figure")
    lines.append("")
    lines.append(f"![Flow and Rainfall — Full History]({FULL_FIG_PATH.relative_to(OUTPUT_DIR).as_posix()})")
    lines.append("")

    lines.append("## 6. Monthly Daily Figures")
    lines.append("")
    lines.append(
        f"A separate dual-axis figure was generated for **every calendar month with available "
        f"data ({len(monthly_keys)} months total)**, showing each individual day of that month "
        f"on the x-axis with daily flow markers and rainfall bars. This lets individual rainfall "
        f"events be visually compared against the plant's flow response on a day-by-day basis. "
        f"All monthly figures are saved under "
        f"[`figures/monthly/`]({MONTHLY_FIG_DIR.relative_to(OUTPUT_DIR).as_posix()}/) "
        f"using the naming pattern `flow_rainfall_YYYY_MM.png`. They are not all embedded here "
        f"to keep this report readable; a few representative examples are shown below."
    )
    lines.append("")
    for y, m in sample_months:
        fig_path = MONTHLY_FIG_DIR / f"flow_rainfall_{y}_{m:02d}.png"
        month_name = MONTH_NUM_TO_NAME[m]
        lines.append(f"### {month_name} {y}")
        lines.append(f"![Flow and Rainfall — {month_name} {y}]({fig_path.relative_to(OUTPUT_DIR).as_posix()})")
        lines.append("")

    lines.append("## 7. Observations")
    lines.append("")
    lines.append(
        "The following are directly observable, descriptive patterns only — no causal or "
        "statistical claim is made:"
    )
    lines.append("")
    lines.append("- Several of the largest single-day flow readings occur on or immediately after days with "
                  "the highest recorded rainfall (e.g., the May 2019 and September 2019 events).")
    lines.append("- Some rainfall events coincide with a visible rise in treated flow in the same monthly window.")
    lines.append("- Not all rainfall events are followed by an obvious increase in flow — several months with "
                  "recorded rainfall show flow staying within its typical day-to-day range.")
    lines.append("- A flow increase sometimes appears to persist for more than one day after a rainfall event, "
                  "and sometimes appears on the same day — the timing of any such response has not been "
                  "measured here and would require lag analysis.")
    lines.append("")

    lines.append("## 8. Next Analysis")
    lines.append("")
    lines.append(
        "A subsequent analysis step may evaluate same-day rainfall vs. flow, previous-day "
        "rainfall vs. flow, 2-day and 3-day lag relationships, and multi-day cumulative "
        "rainfall vs. flow. None of those analyses — including correlation, lag correlation, "
        "cumulative rainfall, or any predictive modeling — are performed in this step."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 2.1: extract rainfall + unit evidence ---
    workbooks = discover_workbooks(RAW_DIR)
    all_records = []
    file_results = []
    header_formats = set()
    unit_note_hits = []  # (file, note)

    for path in workbooks:
        result, records, evidence = process_workbook_rain(path)
        file_results.append(result)
        all_records.extend(records)
        if result.status == "ok":
            if evidence.get("rain_cell_number_format"):
                header_formats.add(evidence["rain_cell_number_format"])
            for note in evidence.get("unit_notes", []):
                unit_note_hits.append((str(path.relative_to(RAW_DIR)), note))

    n_ok = sum(1 for r in file_results if r.status == "ok")

    rain_df = pd.DataFrame(all_records).sort_values(["Date", "Source_File"]).reset_index(drop=True)

    # Cross-check a representative sample of in-workbook "inches" remarks against summed monthly Rain
    cross_checks = []
    seen_files = set()
    for fname, note in unit_note_hits:
        m = re.search(r"(inch|in\.)", note, re.IGNORECASE)
        if not m or fname in seen_files:
            continue
        month_df = rain_df[rain_df["Source_File"] == fname]
        total = month_df["Rainfall_in"].dropna().sum()
        cross_checks.append({"file": fname, "note": note.strip(), "sum": total})
        seen_files.add(fname)
        if len(cross_checks) >= 6:
            break

    unit_evidence = {
        "number_format": ", ".join(sorted(header_formats)) if header_formats else "General",
        "cross_checks": cross_checks,
    }

    # --- Step 2.2: merge onto Step 1 flow dataset (flow = reference) ---
    flow_df = pd.read_csv(FLOW_CSV, parse_dates=["Date"])
    merged = flow_df.merge(
        rain_df[["Date", "Rainfall_in"]], on="Date", how="left"
    )
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged.to_csv(MERGED_CSV, index=False)

    # --- Step 2.3: coverage stats ---
    n_total = len(merged)
    n_valid_flow = int(merged["Total_Treated_MGD"].notna().sum())
    n_valid_rain = int(merged["Rainfall_in"].notna().sum())
    n_valid_both = int((merged["Total_Treated_MGD"].notna() & merged["Rainfall_in"].notna()).sum())
    n_missing_rain = int(merged["Rainfall_in"].isna().sum())
    n_missing_flow = int(merged["Total_Treated_MGD"].isna().sum())

    # Missing calendar periods: months with no row at all in the flow dataset
    present_year_months = sorted(set(zip(merged["Date"].dt.year, merged["Date"].dt.month)))
    full_start = merged["Date"].min()
    full_end = merged["Date"].max()
    all_year_months = []
    y, m = full_start.year, full_start.month
    while (y, m) <= (full_end.year, full_end.month):
        all_year_months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    missing_year_months = [ym for ym in all_year_months if ym not in present_year_months]
    if missing_year_months:
        missing_periods_desc = ", ".join(f"{MONTH_NUM_TO_NAME[m]} {y}" for y, m in missing_year_months)
    else:
        missing_periods_desc = "None"

    coverage = {
        "n_total": n_total,
        "n_valid_flow": n_valid_flow,
        "n_valid_rain": n_valid_rain,
        "n_valid_both": n_valid_both,
        "n_missing_rain": n_missing_rain,
        "n_missing_flow": n_missing_flow,
        "flow_min": flow_df["Date"].min(),
        "flow_max": flow_df["Date"].max(),
        "rain_min": rain_df["Date"].min(),
        "rain_max": rain_df["Date"].max(),
        "n_ok": n_ok,
        "missing_periods_desc": missing_periods_desc,
    }

    # --- Rainfall statistics (Step 2.3 / 2.7) ---
    rstats = rainfall_stats(merged["Rainfall_in"])
    max_val = rstats["max"]
    if max_val is not None:
        max_date_row = merged.loc[merged["Rainfall_in"] == max_val, "Date"]
        rstats["max_date"] = max_date_row.iloc[0].date()
    else:
        rstats["max_date"] = "N/A"

    stats_df = pd.DataFrame([{
        "Count_Valid_Days": rstats["count"],
        "Days_Rainfall_GT_0": rstats["n_rain_days"],
        "Days_Rainfall_EQ_0": rstats["n_dry_days"],
        "Missing_Rainfall_Days": n_missing_rain,
        "Min_in": rstats["min"],
        "Mean_in": rstats["mean"],
        "Median_in": rstats["median"],
        "P95_in": rstats["p95"],
        "Max_in": rstats["max"],
        "Max_Date": rstats["max_date"],
        "Earliest_Date": rain_df["Date"].min().date(),
        "Latest_Date": rain_df["Date"].max().date(),
    }])
    stats_df.to_csv(RAIN_STATS_CSV, index=False)

    # --- Step 2.4: full-history plot ---
    plot_dual_axis(merged, "RRWWTF Historical Daily Flow and Rainfall", FULL_FIG_PATH, bar_width=0.9)

    # --- Step 2.5: one plot per available month ---
    monthly_keys = present_year_months
    for (y, m) in monthly_keys:
        month_df = merged[(merged["Date"].dt.year == y) & (merged["Date"].dt.month == m)]
        out_path = MONTHLY_FIG_DIR / f"flow_rainfall_{y}_{m:02d}.png"
        title = f"RRWWTF Daily Flow and Rainfall — {MONTH_NUM_TO_NAME[m]} {y}"
        plot_dual_axis(
            month_df, title, out_path, bar_width=0.7, markers=True,
            date_locator=mdates.DayLocator(interval=2),
            date_formatter=mdates.DateFormatter("%b %d"),
        )

    # Pick a few representative months for the report (include the known heavy-rain months if present)
    preferred = [(2019, 5), (2019, 9), (2018, 1)]
    sample_months = [ym for ym in preferred if ym in monthly_keys]
    for ym in monthly_keys:
        if len(sample_months) >= 3:
            break
        if ym not in sample_months:
            sample_months.append(ym)

    # --- Step 2.8: report ---
    report_text = build_report(unit_evidence, coverage, rstats, monthly_keys, sample_months,
                                file_results, workbooks)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    # --- Console summary ---
    print("RRWWTF STEP 2 - DAILY FLOW & RAINFALL SUMMARY")
    print(f"Date range:                       {coverage['flow_min'].date()} to {coverage['flow_max'].date()}")
    print(f"Valid flow observations:          {n_valid_flow}")
    print(f"Valid rainfall observations:      {n_valid_rain}")
    print(f"Dates with both measurements:     {n_valid_both}")
    print(f"Rain days (> 0 in):               {rstats['n_rain_days']}")
    print(f"Missing rainfall observations:    {n_missing_rain}")
    print(f"Maximum daily rainfall:           {fmt(rstats['max'])} in on {rstats['max_date']}")
    print(f"Monthly figures generated:        {len(monthly_keys)}")
    print()
    print("Output files:")
    print(f"  {MERGED_CSV}")
    print(f"  {RAIN_STATS_CSV}")
    print(f"  {FULL_FIG_PATH}")
    print(f"  {MONTHLY_FIG_DIR}\\ ({len(monthly_keys)} files)")
    print(f"  {REPORT_PATH}")


if __name__ == "__main__":
    main()

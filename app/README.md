# RRWWTF Flow Prediction Dashboard

A single-page dashboard over the Phase 7 flow model and the soil-saturation model
for the Richmond Regional WWTF. React 18 + Vite + Tailwind + Recharts.

The page follows the Civitas water-demand forecast app: purple header, a "today"
card with the predicted flow, a strip of seven day cards (tap one to show it in
the top card), and the rest behind "View ..." rows. Tapping the top card opens
the plant inputs, chart and day-by-day table.

The page has two parts. The model outputs are a pre-computed bundle; only the
weather forecast is fetched live:

- **7-day flow forecast** (top): the Phase 7 production model, exported as its
  trees by `scripts/forecast_7day.py` and rolled forward in the browser
  (`src/lib/forecast.js`) - day 1 from the measured state, each later day from
  the previous day's prediction. Rain and minimum temperature for the past 31
  and next 7 days come from the Open-Meteo forecast API for Richmond and can be
  overwritten; the user enters the current dry-weather flow and the last three
  days' measured flow, because plant flow is not a live feed. The likely range
  per day is the 10th-90th percentile of the backtest error at that lead day
  (`output/23_seven_day_forecast/lead_time_scores.csv`), wet and dry separately.
  That backtest used recorded rainfall, so real rain-forecast error is extra.
- **How the model performed on past days**: the retrospective 2024 test view.

---

## Quick start

```bash
cd app
npm install
npm run data     # regenerate src/data/predictions.json from the model outputs
npm run dev      # http://localhost:5173 (or the next free port)
npm run build    # production bundle -> dist/
npm run preview  # serve dist/ locally
```

`npm run data` needs Python with `pandas` and `numpy`. Everything else is npm.

---

## Where the numbers come from

`scripts/build_data.py` reads three files from the project's `output/` tree and
writes one JSON bundle that the app imports at build time:

| Source | Used for |
|---|---|
| `output/26_phase7_mrms_rain/test_predictions_2024.csv` | actual vs predicted flow, rainfall, Tmin |
| `output/26_phase7_mrms_rain/test_scores.csv` | cross-check only — never displayed directly |
| `output/14_saturation_model/saturation_timeseries.csv` | saturation index, Wet/Dry label, precipitation |

The script **recomputes** every metric from the raw predictions rather than
copying the scorecard, then asserts the recomputed values match
`test_scores.csv`. If the model is retrained and the scorecard moves, the build
prints a warning instead of silently showing stale numbers. Current agreement:

```
R2 / RMSE / MAE : 0.527 / 0.326 / 0.215 MGD
within +/-0.3   : 77.1%
rain split      : wet R2=+0.543 (n=104)  dry R2=-0.241 (n=110)
```

Those match `test_scores.csv` to four decimals.

---

## Things worth knowing before quoting this dashboard

**The test window is 2024-01-04 to 2024-08-28, not the full year.** That is the
extent of `test_predictions_2024.csv`. 217 days are present; 21 calendar days
inside the span are missing from the record entirely. The charts use a real time
axis so those gaps stay visible instead of being closed up.

**There are two different Wet/Dry definitions in this project and they
disagree.** The dashboard shows both and always names which one is in use:

- **Rain rule** — wet if rain today or yesterday exceeded 0.1. This is the split
  the model itself scores, and it is where `R² = 0.543 wet / −0.241 dry` comes
  from. 104 wet / 110 dry days.
- **Saturation rule** — the saturation model's 14-day water-balance label, which
  also requires measurable precipitation. Only 34 wet / 180 dry days, and the dry
  subset scores `R² = +0.242` rather than −0.241.

A day can read `DRY` on the saturation panel and `WET` under the rain rule on the
same screen. That is the data, not a bug. Quoting one split's R² under the other's
name is the mistake this layout exists to prevent.

**"Accuracy 89.9%" is `100 − MAPE`, a percentage-error measure.** It is *not* the
±0.3 MGD hit rate, which is **77.1%**. Both are shown, separately labelled. (The
original dashboard spec listed 89.9% as the ±0.3 accuracy; that conflated the two
columns in `test_scores.csv`.)

**Dry-day predictions are worse than a flat average** (`R² = −0.241` under the
rain rule). The dashboard raises this as a standing alert. Dry-day output should
not drive capacity or compliance decisions.

**"Prediction confidence" is a presentation heuristic, not a model output.** The
Phase 7 model emits a point estimate with no per-day uncertainty. The confidence
chip keys off the one thing validation established — that accuracy splits by rain
state — and is labelled as such in the code (`src/lib/metrics.js`).

**Saturation data ends 2024-08-31**, three days after the prediction record.

---

## Layout

| Section | Contents |
|---|---|
| Today card | Predicted flow, likely range, forecast rain, Normal/High status; tap for the detailed breakdown |
| 7-day forecast | One card per day; tap a day to show it in the top card |
| Detailed breakdown | Why the flow is expected, comparison to dry-weather flow, plant inputs and per-day rain overrides |
| View Previous 5 Days Predictions | Recent days' predictions |
| View Actual vs Predicted Flow | Retrospective 2024 test chart with date-range controls |

Chart colors are validated for colorblind separation and surface contrast in both
light and dark mode; the two modes use different steps of the same hues rather
than one palette flipped.

---

## Updating with new predictions

1. Retrain / rerun the model so `output/26_phase7_mrms_rain/` and
   `output/14_saturation_model/` are refreshed. From `scripts/`:
   `python fetch_wet_weather_inputs.py` (new gauge data), then
   `python train_phase7_model.py` (also appends the run to the scorecard history; delete
   `output/26_phase7_mrms_rain/mrms_daily_9pt.csv` first to re-download MRMS rain),
   then `python rating_curve.py` and `python forecast_7day.py`.
2. `npm run data` — watch for `WARNING:` lines, which mean the recomputed metrics
   no longer match the model's own scorecard.
3. `npm run build`, then redeploy `dist/`.

Weekly or daily batch refresh is the intended cadence. Making this real-time
would mean standing the Phase 7 model up behind an endpoint and having the page
fetch instead of importing the bundle — the components already take their data as
props, so only `src/Dashboard.jsx` would change.

---

## Deploying

**Vercel**

```bash
npm i -g vercel
cd app
vercel            # preview
vercel --prod     # production
```

Framework preset: **Vite**. Build command `npm run build`, output directory
`dist`. Run `npm run data` before deploying — the bundle is committed as JSON, and
Vercel will not have Python or the `output/` tree.

**Static hosting / City portal**

`vite.config.js` sets `base: './'`, so `dist/` works from any sub-path. Copy the
folder to the web root of choice; there is no server-side component.

---

## Project layout

```
app/
├── scripts/build_data.py       data pipeline + metric validation
├── src/
│   ├── Dashboard.jsx           layout, filter state, theme plumbing
│   ├── lib/metrics.js          metric definitions, formatters, bands
│   ├── lib/forecast.js         7-day rollout of the model + weather fetch
│   ├── data/predictions.json   generated - do not edit by hand
│   └── components/             one file per panel, plus shared ui.jsx
├── tailwind.config.js          Pearland design tokens
└── index.html
```

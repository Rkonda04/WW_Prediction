# RRWWTF Flow Prediction Dashboard

A single-page dashboard over the Phase 5 flow model and the soil-saturation model
for the Richmond Regional WWTF. React 18 + Vite + Tailwind + Recharts.

The page is a **retrospective viewer over a pre-computed bundle**, not a live
feed. Nothing in it calls the model.

---

## Quick start

```bash
cd app
npm install
npm run data     # regenerate src/data/predictions.json from the model outputs
npm run dev      # http://localhost:5173
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
| `output/13_screened_model/test_predictions_2024.csv` | actual vs predicted flow, rainfall, Tmin |
| `output/13_screened_model/test_scores.csv` | cross-check only — never displayed directly |
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
Phase 5 model emits a point estimate with no per-day uncertainty. The confidence
chip keys off the one thing validation established — that accuracy splits by rain
state — and is labelled as such in the code (`src/lib/metrics.js`).

**Saturation data ends 2024-08-31**, three days after the prediction record.

---

## Layout

| Panel | Contents |
|---|---|
| Key metrics | Predicted flow, saturation index, confidence, data currency |
| Flow forecast | Actual (solid) vs predicted (dashed), optional ±0.3 MGD band |
| Saturation state | Gauge, classification, 90-day saturation trend |
| Error analysis | Residual scatter with ±0.1 / ±0.3 / ±0.5 reference lines |
| Model performance | Headline scores + switchable wet/dry breakdown |
| Alerts & notes | Standing caveats, provenance, validation vs `test_scores.csv` |

Interactivity: 30/60/90/All presets and a custom date range (applies to every
panel), a day selector that drives the saturation and confidence panels, a
tolerance-band toggle, hover tooltips on both charts, and a dark-mode toggle
persisted per browser. Cards stack to one column on phones.

Chart colors are validated for colorblind separation and surface contrast in both
light and dark mode; the two modes use different steps of the same hues rather
than one palette flipped.

---

## Updating with new predictions

1. Retrain / rerun the model so `output/13_screened_model/` and
   `output/14_saturation_model/` are refreshed.
2. `npm run data` — watch for `WARNING:` lines, which mean the recomputed metrics
   no longer match the model's own scorecard.
3. `npm run build`, then redeploy `dist/`.

Weekly or daily batch refresh is the intended cadence. Making this real-time
would mean standing the Phase 5 model up behind an endpoint and having the page
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
│   ├── data/predictions.json   generated - do not edit by hand
│   └── components/             one file per panel, plus shared ui.jsx
├── tailwind.config.js          Pearland design tokens
└── index.html
```

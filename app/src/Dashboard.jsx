import { useEffect, useMemo, useState } from 'react'
import bundle from './data/predictions.json'
import Alerts from './components/Alerts'
import Controls from './components/Controls'
import ErrorAnalysis from './components/ErrorAnalysis'
import FlowForecast from './components/FlowForecast'
import KeyMetrics from './components/KeyMetrics'
import Performance from './components/Performance'
import SaturationState from './components/SaturationState'
import { useTheme } from './components/ui'
import { cssVar } from './lib/metrics'

const {
  predictions,
  metrics,
  model,
  validation,
  saturation_history: saturationHistory,
  generated_at: generatedAt,
} = bundle

const MIN_DATE = predictions[0].date
const MAX_DATE = predictions[predictions.length - 1].date

/** Shift an ISO date by a whole number of days, staying in UTC. */
function shiftDate(iso, days) {
  const d = new Date(`${iso}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

function presetRange(key) {
  if (key === 'all') return { from: MIN_DATE, to: MAX_DATE }
  const from = shiftDate(MAX_DATE, -(Number(key) - 1))
  return { from: from < MIN_DATE ? MIN_DATE : from, to: MAX_DATE }
}

export default function Dashboard() {
  const [dark, setDark] = useTheme()
  const [preset, setPreset] = useState('60')
  const [range, setRange] = useState(() => presetRange('60'))
  const [showBands, setShowBands] = useState(true)
  const [selectedDate, setSelectedDate] = useState(MAX_DATE)

  // Chart marks are SVG, so they cannot inherit Tailwind's dark variants. Re-read
  // the themed custom properties whenever the theme flips.
  const [theme, setTheme] = useState(readTheme)
  useEffect(() => {
    setTheme(readTheme())
  }, [dark])

  const applyPreset = (key) => {
    setPreset(key)
    setRange(presetRange(key))
  }

  const applyRange = (next) => {
    setPreset('custom')
    setRange(next)
  }

  const rows = useMemo(
    () => predictions.filter((r) => r.date >= range.from && r.date <= range.to),
    [range],
  )

  // Error analysis keeps its own 90-day window per the panel spec, independent
  // of the forecast range, but never reaches outside the user's selection.
  const residualRows = useMemo(() => {
    const floor = shiftDate(range.to, -89)
    const from = floor > range.from ? floor : range.from
    return predictions.filter((r) => r.date >= from && r.date <= range.to)
  }, [range])

  const dayOptions = useMemo(() => rows.map((r) => r.date).reverse(), [rows])

  // Keep the selected day inside the visible range.
  useEffect(() => {
    if (!rows.length) return
    if (!rows.some((r) => r.date === selectedDate)) {
      setSelectedDate(rows[rows.length - 1].date)
    }
  }, [rows, selectedDate])

  const selected = useMemo(
    () => predictions.find((r) => r.date === selectedDate) ?? null,
    [selectedDate],
  )

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-700 dark:bg-surface-dark">
        <div className="mx-auto flex max-w-[1400px] flex-col gap-1 px-4 py-4 sm:px-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-50">
              RRWWTF Flow Prediction Dashboard
            </h1>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Richmond Regional WWTF {'·'} Phase 5 model
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Retrospective view of {model.name} against observed plant flow, with the 14-day
            soil-saturation state for the same days.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-4 sm:px-6">
        <Controls
          bounds={{ min: MIN_DATE, max: MAX_DATE }}
          range={range}
          onRange={applyRange}
          preset={preset}
          onPreset={applyPreset}
          dark={dark}
          onDark={setDark}
          selectedDate={selectedDate}
          onSelectedDate={setSelectedDate}
          dayOptions={dayOptions}
        />

        <KeyMetrics
          row={selected}
          isLatest={selectedDate === MAX_DATE}
          generatedAt={generatedAt.slice(0, 10)}
        />

        {rows.length === 0 ? (
          <div className="card mt-4 px-4 py-8 text-center text-sm text-slate-500 dark:text-slate-400">
            No days fall in the selected range.
          </div>
        ) : (
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <FlowForecast
                rows={rows}
                theme={theme}
                showBands={showBands}
                onToggleBands={setShowBands}
              />
            </div>
            <SaturationState row={selected} theme={theme} history={saturationHistory} />

            <div className="xl:col-span-2">
              <ErrorAnalysis rows={residualRows} theme={theme} />
            </div>
            <Performance metrics={metrics} model={model} />

            <div className="xl:col-span-2">
              <Alerts
                metrics={metrics}
                model={model}
                row={selected}
                validation={validation}
              />
            </div>
            <Sources />
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-6 text-[11px] text-slate-400 sm:px-6 dark:text-slate-500">
        Figures recomputed from the model's raw predictions at build time and cross-checked
        against output/13_screened_model/test_scores.csv.
      </footer>
    </div>
  )
}

function Sources() {
  return (
    <section className="card flex flex-col px-4 py-3">
      <h2 className="card-title">Data sources</h2>
      <ul className="mt-2 space-y-2 text-[11px] text-slate-500 dark:text-slate-400">
        {Object.entries(bundle.source_files).map(([k, v]) => (
          <li key={k}>
            <div className="font-medium capitalize text-slate-600 dark:text-slate-300">{k}</div>
            <code className="break-all text-[10px]">{v}</code>
          </li>
        ))}
      </ul>
      <dl className="mt-3 space-y-1 border-t border-slate-200/70 pt-2.5 text-[11px] dark:border-slate-700/60">
        {[
          ['Test days', model.n_test_days],
          ['Flagged (unscored)', model.n_screened_out],
          ['Scored days', metrics.all.n],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <dt className="text-slate-500 dark:text-slate-400">{k}</dt>
            <dd className="font-medium tabular-nums text-slate-700 dark:text-slate-200">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function readTheme() {
  return {
    actual: cssVar('--series-actual', '#1E88E5'),
    predicted: cssVar('--series-predicted', '#E8710A'),
    band: cssVar('--band-fill', 'rgba(30,136,229,0.10)'),
    grid: cssVar('--chart-grid', '#E6E8EB'),
    axis: cssVar('--chart-axis', '#8A9099'),
    ink: cssVar('--chart-ink', '#1F2933'),
    surface: cssVar('--chart-surface', '#FFFFFF'),
    good: '#43A047',
    bad: '#E53935',
  }
}

import { useEffect, useMemo, useState } from 'react'
import bundle from './data/predictions.json'
import Controls from './components/Controls'
import FlowForecast from './components/FlowForecast'
import { Card, useTheme } from './components/ui'
import { cssVar, fmt } from './lib/metrics'

const { predictions, model } = bundle

const MAX_DATE = predictions[predictions.length - 1].date

// Newest first, so the dropdown opens on the most recent day.
const DAY_OPTIONS = predictions.map((r) => r.date).reverse()

export default function Dashboard() {
  const [dark, setDark] = useTheme()
  const [showBands, setShowBands] = useState(true)
  const [selectedDate, setSelectedDate] = useState(MAX_DATE)

  // Chart marks are SVG, so they cannot inherit Tailwind's dark variants. Re-read
  // the themed custom properties whenever the theme flips.
  const [theme, setTheme] = useState(readTheme)
  useEffect(() => {
    setTheme(readTheme())
  }, [dark])

  const selected = useMemo(
    () => predictions.find((r) => r.date === selectedDate) ?? null,
    [selectedDate],
  )

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-700 dark:bg-surface-dark">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-baseline justify-between gap-2 px-4 py-4 sm:px-6">
          <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-50">
            RRWWTF Flow Prediction Dashboard
          </h1>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Richmond Regional WWTF {'·'} Phase 5 model
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-4 py-4 sm:px-6">
        <Controls
          dark={dark}
          onDark={setDark}
          selectedDate={selectedDate}
          onSelectedDate={setSelectedDate}
          dayOptions={DAY_OPTIONS}
        />

        <PredictedFlow row={selected} isLatest={selectedDate === MAX_DATE} />

        <div className="mt-4">
          <FlowForecast
            rows={predictions}
            theme={theme}
            showBands={showBands}
            onToggleBands={setShowBands}
            highlightDate={selectedDate}
          />
        </div>
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-6 text-[11px] text-slate-400 sm:px-6 dark:text-slate-500">
        Batch predictions, {model.test_start} to {model.test_end}. Not a live feed.
      </footer>
    </div>
  )
}

function PredictedFlow({ row, isLatest }) {
  return (
    <Card bodyClass="flex flex-wrap items-end justify-between gap-6">
      <div>
        <div className="stat-label">{isLatest ? 'Latest predicted flow' : 'Predicted flow'}</div>
        <div className="mt-1 flex items-baseline gap-2">
          <span className="text-5xl font-semibold tabular-nums text-primary dark:text-primary-light">
            {fmt.mgd(row?.predicted)}
          </span>
          <span className="text-base font-medium text-slate-500 dark:text-slate-400">MGD</span>
        </div>
        <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          {row ? fmt.date(row.date) : '—'}
        </div>
      </div>

      {row && (
        <dl className="flex gap-8">
          <div>
            <dt className="stat-label">Actual</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
              {fmt.mgd(row.actual)}
            </dd>
          </div>
          <div>
            <dt className="stat-label">Error</dt>
            <dd className="mt-1 text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
              {fmt.signed(row.error)}
            </dd>
          </div>
        </dl>
      )}
    </Card>
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

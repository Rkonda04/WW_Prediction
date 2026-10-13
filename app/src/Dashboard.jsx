import { useState } from 'react'
import bundle from './data/predictions.json'
import ActualVsPredicted from './components/ActualVsPredicted'
import SevenDayForecast from './components/SevenDayForecast'
import { cssVar } from './lib/metrics'

const { forecast, predictions, model } = bundle

export default function Dashboard() {
  const [page, setPage] = useState('forecast')

  // Chart marks are SVG, so they read their colors from the page's custom properties.
  const [theme] = useState(readTheme)

  return (
    <div className="flex min-h-screen flex-col">
      <header className="bg-gradient-to-br from-primary to-primary-dark px-6 py-5 shadow-[0_4px_20px_rgba(74,14,120,0.4)]">
        <div className="mx-auto flex max-w-[680px] items-center justify-between gap-4">
          <img
            src={`${import.meta.env.BASE_URL}civitas-logo.png`}
            alt="Civitas Engineering Group"
            className="h-[50px] w-auto object-contain brightness-0 invert"
          />
          <div className="text-right">
            <h1 className="text-xl font-bold leading-tight tracking-[-0.3px] text-white">Richmond Regional WWTF</h1>
            <p className="mt-1.5 text-[13px] tracking-[0.2px] text-white/85">Daily Wastewater Flow Forecast</p>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[680px] flex-1 p-5">
        {/* Kept mounted while the other page shows, so the plant inputs typed in are not lost. */}
        <div className={page === 'forecast' ? '' : 'hidden'}>
          <SevenDayForecast tool={forecast} theme={theme} />
          <button
            type="button"
            onClick={() => setPage('history')}
            className="card mt-4 flex w-full items-center justify-between px-7 py-6 text-left text-base font-bold text-primary dark:text-primary-light"
          >
            View Actual vs Predicted Flow
            <span aria-hidden="true">→</span>
          </button>
        </div>
        {page === 'history' && (
          <ActualVsPredicted rows={predictions} model={model} onBack={() => setPage('forecast')} />
        )}

      </main>

    </div>
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
    curveDry: cssVar('--curve-dry', '#90BFF0'),
    curveNormal: cssVar('--curve-normal', '#1E88E5'),
    curveWet: cssVar('--curve-wet', '#0B4A8F'),
    good: '#43A047',
    bad: '#E53935',
  }
}

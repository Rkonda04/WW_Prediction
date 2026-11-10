import { useState } from 'react'
import bundle from './data/predictions.json'
import ActualVsPredicted from './components/ActualVsPredicted'
import SevenDayForecast from './components/SevenDayForecast'

const { forecast, predictions, model } = bundle

export default function Dashboard() {
  const [page, setPage] = useState('forecast')

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
        {/* Kept mounted while the other page shows, so the fetched forecast is not refetched. */}
        <div className={page === 'forecast' ? '' : 'hidden'}>
          <SevenDayForecast tool={forecast} />
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

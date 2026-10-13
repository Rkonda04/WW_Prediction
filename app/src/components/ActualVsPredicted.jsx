import { useMemo, useState } from 'react'
import Controls from './Controls'
import { fmt } from '../lib/metrics'

/**
 * Second page: what the model predicted against what the plant measured, over
 * the 2024 test period: the figures for one chosen day.
 */
export default function ActualVsPredicted({ rows, model, onBack }) {
  const dayOptions = useMemo(() => rows.map((r) => r.date).reverse(), [rows])
  const [selectedDate, setSelectedDate] = useState(dayOptions[0])
  const row = rows.find((r) => r.date === selectedDate) ?? null

  return (
    <div>
      <button
        type="button"
        onClick={onBack}
        className="mb-3 text-[13px] font-semibold text-primary dark:text-primary-light"
      >
        ← Back to Forecast
      </button>
      <h2 className="text-2xl font-bold tracking-[-0.3px] text-slate-700 dark:text-slate-100">Actual vs Predicted Flow</h2>
      <p className="mb-5 mt-1 text-xs text-slate-500 dark:text-slate-400">
        {fmt.date(model.test_start)} to {fmt.date(model.test_end)}, the most recent stretch with measured plant flow.
        The model never saw these days in training.
      </p>

      <Controls selectedDate={selectedDate} onSelectedDate={setSelectedDate} dayOptions={dayOptions} />

      <section className="card grid grid-cols-3 gap-4 p-6">
        {[
          ['Predicted', fmt.mgd(row?.predicted), 'text-primary dark:text-primary-light'],
          ['Actual', fmt.mgd(row?.actual), 'text-slate-700 dark:text-slate-50'],
          ['Difference', fmt.signed(row?.error), 'text-slate-700 dark:text-slate-50'],
        ].map(([label, value, tone]) => (
          <div key={label}>
            <div className="stat-label">{label}</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className={`text-3xl font-extrabold tabular-nums ${tone}`}>{value}</span>
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">MGD</span>
            </div>
          </div>
        ))}
      </section>
    </div>
  )
}

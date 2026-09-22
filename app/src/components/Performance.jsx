import { useState } from 'react'
import { Card, Stat } from './ui'
import { fmt } from '../lib/metrics'

/**
 * Headline scores for the full test window, straight from the build script
 * (which reproduces output/13_screened_model/test_scores.csv exactly).
 *
 * The wet/dry breakdown is shown under two switchable definitions because the
 * project has two, and they disagree:
 *   - Rain: the split the model itself scores (rain today or yesterday > 0.1).
 *   - Saturation: the 14-day water-balance label from the saturation model.
 * Quoting one set of numbers under the other's name is the easy mistake here, so
 * the active definition is always named on screen.
 */
const SPLITS = {
  rain: {
    label: 'Rain',
    tab: 'By rain',
    wetKey: 'rain_wet',
    dryKey: 'rain_dry',
    note: 'Wet = measurable rain today or yesterday (>0.1). This is the split the model report quotes.',
  },
  saturation: {
    label: 'Saturation',
    tab: 'By saturation',
    wetKey: 'sat_wet',
    dryKey: 'sat_dry',
    note: 'Wet = the saturation model’s 14-day water-balance label. Far fewer days qualify.',
  },
}

export default function Performance({ metrics, model }) {
  const [split, setSplit] = useState('rain')
  const cfg = SPLITS[split]
  const all = metrics.all
  const wet = metrics[cfg.wetKey]
  const dry = metrics[cfg.dryKey]

  return (
    <Card
      title="Model performance"
      subtitle={`Full test window · ${model.test_start} to ${model.test_end} · n = ${all.n}`}
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Stat label="R²" value={fmt.r2(all.r2)} hint="Variance explained" />
        <Stat label="RMSE" value={fmt.mgd(all.rmse)} unit="MGD" hint="Root mean sq. error" />
        <Stat label="MAE" value={fmt.mgd(all.mae)} unit="MGD" hint="Mean abs. error" />
        <Stat
          label="Within ±0.3"
          value={fmt.pct(all.within_tolerance)}
          hint={`${Math.round(all.within_tolerance * all.n)} of ${all.n} days`}
        />
        <Stat
          label="Accuracy"
          value={`${(100 - all.mape).toFixed(1)}%`}
          hint="100 − MAPE"
          footnote="Percentage-error measure, not the ±0.3 MGD hit rate."
        />
        <Stat
          label="Mean error"
          value={fmt.signed(all.bias)}
          unit="MGD"
          hint={all.bias < 0 ? 'Slight under-prediction' : 'Slight over-prediction'}
        />
      </div>

      <div className="mt-4 border-t border-slate-200/70 pt-3 dark:border-slate-700/60">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-xs font-semibold text-slate-700 dark:text-slate-200">
            Wet vs dry breakdown
          </h3>
          <div
            className="flex gap-1"
            role="tablist"
            aria-label="Wet/dry definition"
          >
            {Object.entries(SPLITS).map(([key, s]) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={split === key}
                onClick={() => setSplit(key)}
                className={`btn ${split === key ? 'btn-active' : ''}`}
              >
                {s.tab}
              </button>
            ))}
          </div>
        </div>

        <table className="mt-2.5 w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500 dark:text-slate-400">
              <th className="py-1 font-medium">{cfg.label} state</th>
              <th className="py-1 text-right font-medium">Days</th>
              <th className="py-1 text-right font-medium">R{'²'}</th>
              <th className="py-1 text-right font-medium">RMSE</th>
              <th className="py-1 text-right font-medium">{'±'}0.3</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200/70 dark:divide-slate-700/60">
            {[
              ['Wet', wet],
              ['Dry', dry],
            ].map(([label, m]) => {
              const weak = m.r2 != null && m.r2 < 0
              return (
                <tr key={label}>
                  <td className="py-1.5 font-medium text-slate-700 dark:text-slate-200">
                    {label}
                    {weak && (
                      <span className="ml-1.5 rounded bg-danger/10 px-1.5 py-0.5 text-[10px] font-semibold text-danger-dark dark:text-danger-light">
                        WEAK
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300">
                    {m.n}
                  </td>
                  <td
                    className={`py-1.5 text-right font-semibold tabular-nums ${
                      weak
                        ? 'text-danger-dark dark:text-danger-light'
                        : 'text-slate-800 dark:text-slate-100'
                    }`}
                  >
                    {fmt.r2(m.r2)}
                  </td>
                  <td className="py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300">
                    {fmt.mgd(m.rmse)}
                  </td>
                  <td className="py-1.5 text-right tabular-nums text-slate-600 dark:text-slate-300">
                    {fmt.pct(m.within_tolerance, 0)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
          {cfg.note}
        </p>
      </div>
    </Card>
  )
}

import { Card } from './ui'
import { fmt } from '../lib/metrics'

const pct = (v) => `${v.toFixed(1)}%`
const pts = (a, b) => `${fmt.signed(b - a, 1)} pts`
const lower = (a, b) => `${Math.abs(((b - a) / a) * 100).toFixed(0)}% ${b <= a ? 'lower' : 'higher'}`
const smaller = (a, b) => `${Math.abs(Math.abs(a) - Math.abs(b)).toFixed(2)} MGD ${Math.abs(b) <= Math.abs(a) ? 'smaller' : 'larger'}`

// One row per scorecard metric: how to print each phase's value and the change between them.
const ROWS = [
  { key: 'mae_all', label: 'MAE, all days (MGD)', show: (v) => v.toFixed(3), change: lower },
  { key: 'r2_all', label: 'R², all days', show: (v) => fmt.signed(v).replace('+', ''), change: (a, b) => fmt.signed(b - a) },
  { key: 'accuracy_pct', label: 'Mean accuracy (100% − MAPE)', show: pct, change: pts },
  { key: 'within_0_3_pct', label: 'Days within ±0.3 MGD', show: pct, change: pts },
  { key: 'correct_band_pct', label: 'Counted correct (−0.2 / +0.3 MGD)', show: pct, change: pts },
  { key: 'mae_wet', label: 'MAE, wet days (MGD)', show: (v) => v.toFixed(3), change: lower },
  { key: 'r2_wet', label: 'R², wet days', show: (v) => fmt.signed(v).replace('+', ''), change: (a, b) => fmt.signed(b - a) },
  { key: 'bias_high_flow', label: 'Bias, days of 2.5 MGD or more', show: (v) => fmt.signed(v).replace('+', ''), change: smaller },
  { key: 'p99_gap_wet', label: 'Wet-day gap at 99th percentile (MGD)', show: (v) => fmt.signed(v).replace('+', ''), change: smaller },
]

/**
 * Phase 5 against Phase 6. Values come from the bundle: either this project's
 * pooled out-of-sample scorecard (scripts/scorecard.py), or, when
 * scripts/reported_scorecard.json exists, figures reported from a run made
 * elsewhere - the subtitle says which.
 */
export default function PhaseComparison({ comparison }) {
  const { phase5: a, phase6: b } = comparison
  return (
    <Card
      title="Model scorecard — Phase 5 vs Phase 6"
      subtitle={
        comparison.reported
          ? 'As reported from a training run on another machine'
          : `Each year predicted by a model that never saw it (${comparison.years}) · ${b.n_days.toLocaleString()} days scored`
      }
      bodyClass="!p-0"
    >
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="bg-slate-800 text-left text-white dark:bg-slate-700">
              <th className="px-4 py-2 font-semibold">Metric</th>
              <th className="px-4 py-2 text-right font-semibold">Phase 5</th>
              <th className="px-4 py-2 text-right font-semibold">Phase 6</th>
              <th className="px-4 py-2 text-right font-semibold">Change</th>
            </tr>
          </thead>
          <tbody className="tabular-nums text-slate-700 dark:text-slate-200">
            {ROWS.map((r, i) => (
              <tr
                key={r.key}
                className={`border-t border-slate-200/70 dark:border-slate-700/60 ${
                  i % 2 ? 'bg-slate-50 dark:bg-slate-800/40' : ''
                }`}
              >
                <td className="px-4 py-2">{r.label}</td>
                <td className="px-4 py-2 text-right">{r.show(a[r.key])}</td>
                <td className="px-4 py-2 text-right font-semibold text-slate-900 dark:text-slate-50">
                  {r.show(b[r.key])}
                </td>
                <td className="px-4 py-2 text-right">{r.change(a[r.key], b[r.key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

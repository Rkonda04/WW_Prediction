import { Card } from './ui'
import { dayConfidence, fmt } from '../lib/metrics'

const ICONS = {
  alert: (
    <path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
  ),
  info: <path d="M12 16v-4m0-4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />,
  note: <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2Z" />,
}

const TONES = {
  alert: {
    ring: 'border-danger/30 bg-danger/5',
    icon: 'text-danger-dark dark:text-danger-light',
    label: 'Alert',
  },
  warn: {
    ring: 'border-warning/30 bg-warning/5',
    icon: 'text-warning-dark dark:text-warning-light',
    label: 'Caution',
  },
  info: {
    ring: 'border-primary/25 bg-primary/5',
    icon: 'text-primary dark:text-primary-light',
    label: 'Info',
  },
}

export default function Alerts({ metrics, model, row, validation }) {
  const conf = dayConfidence(row)
  const dry = metrics.rain_dry
  const items = []

  if (dry.r2 != null && dry.r2 < 0) {
    items.push({
      tone: 'alert',
      icon: 'alert',
      title: `Dry-day performance is worse than a flat average (R² = ${fmt.r2(dry.r2)})`,
      body: `Across ${dry.n} dry days the model's errors exceed those of simply predicting the period mean. Dry-day figures should be treated as indicative only, not used for capacity or compliance decisions.`,
    })
  }

  if (row) {
    items.push({
      tone: conf.level === 'MODERATE' ? 'warn' : 'alert',
      icon: 'alert',
      title: `${fmt.date(row.date)} is a ${row.rain_wet ? 'WET' : 'DRY'} day by the rain rule — confidence ${conf.label.toUpperCase()}`,
      body: conf.note,
    })
  }

  items.push({
    tone: 'info',
    icon: 'info',
    title: 'Predictions are batch, not live',
    body: `The bundle covers ${model.test_start} to ${model.test_end}. Nothing on this page updates on its own — rerun npm run data after the model writes new predictions.`,
  })

  items.push({
    tone: 'info',
    icon: 'note',
    title: 'Scoring protocol',
    body: `${model.n_screened_out} of ${model.n_test_days} days were flagged by the screening rule and are excluded from every score shown. Saturation data ends ${model.saturation_end}.`,
  })

  return (
    <Card title="Alerts & notes" subtitle="Read before quoting any figure on this page">
      <ul className="space-y-2.5">
        {items.map((it, i) => {
          const tone = TONES[it.tone]
          return (
            <li key={i} className={`flex gap-2.5 rounded-lg border p-2.5 ${tone.ring}`}>
              <svg
                className={`mt-0.5 h-4 w-4 shrink-0 ${tone.icon}`}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                {ICONS[it.icon]}
              </svg>
              <div className="min-w-0">
                <span className="sr-only">{tone.label}: </span>
                <p className="text-xs font-semibold text-slate-800 dark:text-slate-100">
                  {it.title}
                </p>
                <p className="mt-0.5 text-[11px] leading-snug text-slate-600 dark:text-slate-400">
                  {it.body}
                </p>
              </div>
            </li>
          )
        })}
      </ul>

      <details className="mt-3 border-t border-slate-200/70 pt-2.5 dark:border-slate-700/60">
        <summary className="cursor-pointer text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-slate-100">
          Provenance & validation
        </summary>
        <dl className="mt-2 space-y-1 text-[11px] text-slate-500 dark:text-slate-400">
          <div className="flex justify-between gap-3">
            <dt>Model</dt>
            <dd className="text-right font-medium text-slate-700 dark:text-slate-200">
              {model.name}
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt>Protocol</dt>
            <dd className="text-right">{model.protocol}</dd>
          </div>
          {[
            ['All', validation.all],
            ['Wet (rain)', validation.rain_wet],
            ['Dry (rain)', validation.rain_dry],
          ].map(([label, v]) => (
            <div key={label} className="flex justify-between gap-3">
              <dt>{label} R{'²'} vs test_scores.csv</dt>
              <dd className="text-right tabular-nums">
                {fmt.r2(v.recomputed_r2)} vs {fmt.r2(v.published_r2)}
                <span className="ml-1 text-accent-dark dark:text-accent-light">
                  {Math.abs(v.recomputed_r2 - v.published_r2) < 0.001 ? '✓' : '⚠'}
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </details>
    </Card>
  )
}

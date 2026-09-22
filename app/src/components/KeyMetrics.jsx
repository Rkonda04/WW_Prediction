import { Card, Chip, Stat } from './ui'
import { BANDS, dayConfidence, fmt, saturationBand } from '../lib/metrics'

/**
 * Headline tiles for the selected day.
 *
 * The record ends 2024-08-28, so there is no "today" to report. The tile is
 * labelled for the day actually being shown rather than implying a live feed.
 */
export default function KeyMetrics({ row, isLatest, generatedAt }) {
  const band = saturationBand(row?.saturation)
  const conf = dayConfidence(row)
  const tone = { LOW: 'bad', MODERATE: 'warn', HIGH: 'good', UNKNOWN: 'default' }[conf.level]

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <Card
        className="xl:col-span-1"
        bodyClass="flex flex-col justify-between gap-2"
      >
        <div>
          <div className="stat-label">
            {isLatest ? 'Latest predicted flow' : 'Predicted flow'}
          </div>
          <div className="mt-1 flex items-baseline gap-1.5">
            <span className="text-4xl font-semibold tabular-nums text-primary dark:text-primary-light">
              {fmt.mgd(row?.predicted)}
            </span>
            <span className="text-sm font-medium text-slate-500 dark:text-slate-400">MGD</span>
          </div>
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-400">
          {row ? fmt.date(row.date) : '—'}
          {row && (
            <span className="ml-1.5">
              {'·'} actual {fmt.mgd(row.actual)} ({fmt.signed(row.error)})
            </span>
          )}
        </div>
      </Card>

      <Card bodyClass="flex flex-col justify-between gap-2">
        <Stat
          label="Saturation index"
          value={row?.saturation == null ? '—' : row.saturation.toFixed(3)}
          hint={`Band: ${band.label} (${band.range})`}
        />
        <div className="flex items-center gap-2">
          {row?.classification && (
            <Chip color={row.classification === 'WET' ? BANDS.wet.color : BANDS.dry.color}>
              {row.classification}
            </Chip>
          )}
          <SatBar value={row?.saturation ?? 0} color={band.color} />
        </div>
      </Card>

      <Card bodyClass="flex flex-col justify-between gap-2">
        <Stat
          label="Prediction confidence"
          value={conf.label}
          tone={tone}
          hint={row?.rain_wet ? 'Wet day (rain rule)' : 'Dry day (rain rule)'}
        />
        <p className="text-[11px] leading-snug text-slate-500 dark:text-slate-400">{conf.note}</p>
      </Card>

      <Card bodyClass="flex flex-col justify-between gap-2">
        <Stat
          label="Data current through"
          value={fmt.dateShort(row ? row.date : '2024-08-28')}
          hint="Model output is batch, not live"
        />
        <p className="text-[11px] leading-snug text-slate-500 dark:text-slate-400">
          Bundle built {generatedAt}
        </p>
      </Card>
    </div>
  )
}

function SatBar({ value, color }) {
  const pct = Math.min(100, Math.max(0, value * 100))
  return (
    <div
      className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"
      role="img"
      aria-label={`Saturation index ${value.toFixed(3)} of 1`}
    >
      <div
        className="h-full rounded-full transition-all"
        style={{ width: `${Math.max(pct, 1.5)}%`, backgroundColor: color }}
      />
    </div>
  )
}

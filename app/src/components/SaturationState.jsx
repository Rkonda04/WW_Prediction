import { useMemo } from 'react'
import { Card, Chip } from './ui'
import { BANDS, fmt, saturationBand } from '../lib/metrics'

/**
 * Saturation state for the selected day.
 *
 * Two different things are shown side by side on purpose and must not be
 * conflated: `saturation_index` is a continuous 14-day water-balance figure,
 * while `classification` is the saturation model's own Wet/Dry label, which
 * additionally requires measurable precipitation. A day can therefore sit in a
 * high index band and still be labelled Dry.
 */
export default function SaturationState({ row, theme, history }) {
  const index = row?.saturation ?? null
  const band = saturationBand(index)
  const cls = row?.classification ?? null

  // 90 days of context ending on the selected day, so a reading of 0.00 is
  // legible as "still dry" rather than as a possible data gap.
  const trend = useMemo(() => {
    if (!row || !history?.length) return []
    const end = history.findIndex((h) => h.date === row.date)
    if (end < 0) return []
    return history.slice(Math.max(0, end - 89), end + 1)
  }, [row, history])

  return (
    <Card
      title="Saturation state"
      subtitle={row ? fmt.date(row.date) : 'No day selected'}
      bodyClass="flex flex-col"
    >
      <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-center sm:gap-5">
        <Gauge value={index} band={band} theme={theme} />

        <div className="min-w-0 flex-1 space-y-3">
          <div>
            <div className="stat-label">Classification</div>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              {cls ? (
                <Chip color={cls === 'WET' ? BANDS.wet.color : BANDS.dry.color}>{cls}</Chip>
              ) : (
                <Chip color={BANDS.unknown.color}>No data</Chip>
              )}
              <Chip color={band.color} title={`Index band ${band.range}`}>
                {band.label}
              </Chip>
            </div>
          </div>

          <div>
            <div className="stat-label">Saturation index</div>
            <div className="mt-0.5 text-xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
              {index == null ? '—' : index.toFixed(3)}
              <span className="ml-1 text-xs font-normal text-slate-500 dark:text-slate-400">
                of 1.000
              </span>
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
            <dt className="text-slate-500 dark:text-slate-400">14-day balance</dt>
            <dd className="text-right font-medium tabular-nums text-slate-700 dark:text-slate-200">
              {index == null ? '—' : `${(index * 40).toFixed(1)} mm`}
            </dd>
            <dt className="text-slate-500 dark:text-slate-400">Precipitation</dt>
            <dd className="text-right font-medium tabular-nums text-slate-700 dark:text-slate-200">
              {row?.precip_mm == null ? '—' : `${row.precip_mm.toFixed(2)} mm`}
            </dd>
            <dt className="text-slate-500 dark:text-slate-400">Rain (model input)</dt>
            <dd className="text-right font-medium tabular-nums text-slate-700 dark:text-slate-200">
              {row?.rain_in == null ? '—' : row.rain_in.toFixed(2)}
            </dd>
          </dl>
        </div>
      </div>

      {trend.length > 1 && <Trend rows={trend} theme={theme} />}

      <ul className="mt-3 flex flex-wrap gap-x-3 gap-y-1 border-t border-slate-200/70 pt-2.5 text-[11px] text-slate-500 dark:border-slate-700/60 dark:text-slate-400">
        {[BANDS.veryDry, BANDS.dry, BANDS.moderate, BANDS.wet].map((b) => (
          <li key={b.key} className="flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: b.color }}
            />
            {b.label} <span className="tabular-nums opacity-70">{b.range}</span>
          </li>
        ))}
      </ul>

      <p className="mt-2 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
        Index is the 14-day cumulative water balance over a 40&nbsp;mm reference capacity. The
        Wet label additionally requires precipitation above 0.1&nbsp;mm, so a dry-labelled day can
        still carry a non-zero index.
      </p>
    </Card>
  )
}

/**
 * 90-day saturation trend. A single-series area needs no legend box - the
 * heading names it - and carries only endpoint labels rather than per-point
 * values.
 */
function Trend({ rows, theme }) {
  const w = 100
  const h = 26
  const n = rows.length
  const x = (i) => (i / (n - 1)) * w
  const y = (v) => h - Math.min(1, Math.max(0, v)) * (h - 2) - 1

  const line = rows.map((r, i) => `${i ? 'L' : 'M'} ${x(i).toFixed(2)} ${y(r.saturation).toFixed(2)}`)
  const area = `${line.join(' ')} L ${w} ${h} L 0 ${h} Z`
  const peak = rows.reduce((m, r) => Math.max(m, r.saturation), 0)

  return (
    <figure className="m-0 mt-3 border-t border-slate-200/70 pt-2.5 dark:border-slate-700/60">
      <figcaption className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="font-medium text-slate-600 dark:text-slate-300">
          Saturation, previous 90 days
        </span>
        <span className="tabular-nums text-slate-400 dark:text-slate-500">
          peak {peak.toFixed(2)}
        </span>
      </figcaption>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        preserveAspectRatio="none"
        className="h-[26px] w-full"
        role="img"
        aria-label={`Saturation index over the previous 90 days, peaking at ${peak.toFixed(2)}`}
      >
        <path d={area} fill={theme.actual} fillOpacity="0.16" />
        <path
          d={line.join(' ')}
          fill="none"
          stroke={theme.actual}
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
        />
      </svg>
      <div className="mt-0.5 flex justify-between text-[10px] tabular-nums text-slate-400 dark:text-slate-500">
        <span>{fmt.dateShort(rows[0].date)}</span>
        <span>{fmt.dateShort(rows[n - 1].date)}</span>
      </div>
    </figure>
  )
}

/**
 * Semicircular gauge. The arc is a single sweep with the four status bands laid
 * behind it, so the reading is positional first and color second.
 */
function Gauge({ value, band, theme }) {
  const size = 148
  const cx = size / 2
  const cy = size / 2 + 8
  const r = 58
  const stroke = 12
  const v = value == null ? 0 : Math.min(1, Math.max(0, value))

  const bands = [BANDS.veryDry, BANDS.dry, BANDS.moderate, BANDS.wet]
  const circ = Math.PI * r

  return (
    <figure className="m-0 shrink-0" style={{ width: size }}>
      <svg
        width={size}
        height={cy + stroke}
        role="img"
        aria-label={
          value == null
            ? 'Saturation index unavailable'
            : `Saturation index ${value.toFixed(3)} of 1, ${band.label}`
        }
      >
        {/* Band track: four arc segments with a 2px surface gap between them. */}
        {bands.map((b, i) => {
          const seg = circ / 4
          return (
            <path
              key={b.key}
              d={arc(cx, cy, r)}
              fill="none"
              stroke={b.color}
              strokeOpacity={0.28}
              strokeWidth={stroke}
              strokeDasharray={`${seg - 2} ${circ}`}
              strokeDashoffset={-i * seg}
              strokeLinecap="butt"
            />
          )
        })}

        {/* Value arc. */}
        {value != null && (
          <path
            d={arc(cx, cy, r)}
            fill="none"
            stroke={band.color}
            strokeWidth={stroke}
            strokeDasharray={`${v * circ} ${circ}`}
            strokeLinecap="round"
          />
        )}

        {/* Needle tip, ringed in the surface color so it reads over any band. */}
        {value != null && (
          <circle
            cx={cx + r * Math.cos(Math.PI - v * Math.PI)}
            cy={cy - r * Math.sin(Math.PI - v * Math.PI)}
            r={5}
            fill={band.color}
            stroke={theme.surface}
            strokeWidth={2}
          />
        )}

        <text
          x={cx}
          y={cy - 14}
          textAnchor="middle"
          style={{ fill: theme.ink, fontSize: 26, fontWeight: 600 }}
          className="tabular-nums"
        >
          {value == null ? '—' : value.toFixed(2)}
        </text>
        <text
          x={cx}
          y={cy + 4}
          textAnchor="middle"
          style={{ fill: theme.axis, fontSize: 10, letterSpacing: 0.4 }}
        >
          SATURATION
        </text>
        <text x={cx - r} y={cy + 18} textAnchor="middle" style={{ fill: theme.axis, fontSize: 10 }}>
          0
        </text>
        <text x={cx + r} y={cy + 18} textAnchor="middle" style={{ fill: theme.axis, fontSize: 10 }}>
          1
        </text>
      </svg>
    </figure>
  )
}

/** Half-circle path from 180deg to 0deg. */
function arc(cx, cy, r) {
  return `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`
}

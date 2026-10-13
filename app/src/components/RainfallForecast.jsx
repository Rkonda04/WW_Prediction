import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, Legend, TooltipShell } from './ui'
import { fmt } from '../lib/metrics'

const KEYS = ['dry', 'normal', 'wet']
const SHORT = { dry: 'Dry month', normal: 'Normal month', wet: 'Wet month' }

/**
 * Forward-looking use of the model: forecast rainfall in, predicted plant flow out.
 *
 * Reads the rating curve distilled from the Phase 6 production model
 * (scripts/rating_curve.py): median flow added above the dry-weather level for a
 * given day's rainfall, by how wet the prior 30 days were. Predicted flow is the
 * current dry-weather level plus that added flow.
 */
export default function RainfallForecast({ curve, theme }) {
  const { rain_steps_in: steps, classes, p99_rain_in: p99, latest_state: latest } = curve
  const maxRain = steps[steps.length - 1]

  const [rainText, setRainText] = useState('1.00')
  const [anteText, setAnteText] = useState(latest.ante_30d_in.toFixed(2))
  const [levelText, setLevelText] = useState(latest.dry_level_mgd.toFixed(2))

  const rain = clampNum(rainText, 0, maxRain)
  const ante = clampNum(anteText, 0, 60)
  const level = clampNum(levelText, 0, 20)
  const valid = rain != null && ante != null && level != null

  const ci = ante == null ? 1 : classes.findIndex((c) => ante >= c.ante_lo_in && (c.ante_hi_in == null || ante < c.ante_hi_in))
  const cls = classes[ci]
  const key = KEYS[ci]

  const result = useMemo(() => {
    if (!valid) return null
    const at = (arr) => level + interp(steps, arr, rain)
    return {
      dayOf: at(cls.day_of),
      dayOfRange: [at(cls.day_of_p25), at(cls.day_of_p75)],
      dayAfter: at(cls.day_after),
      dayAfterRange: [at(cls.day_after_p25), at(cls.day_after_p75)],
    }
  }, [valid, level, rain, cls, steps])

  const data = useMemo(
    () =>
      steps.map((r, i) => ({
        rain: r,
        ...Object.fromEntries(classes.map((c, j) => [KEYS[j], (level ?? latest.dry_level_mgd) + c.day_of[i]])),
      })),
    [steps, classes, level, latest.dry_level_mgd],
  )

  const colors = { dry: theme.curveDry, normal: theme.curveNormal, wet: theme.curveWet }
  const legend = classes.map((c, j) => ({ label: c.name, color: colors[KEYS[j]] }))
  const typedRain = Number.parseFloat(rainText)
  const clipped = Number.isFinite(typedRain) && typedRain > maxRain

  return (
    <Card
      title="Rainfall forecast → predicted flow"
      subtitle="Enter forecast rainfall for a day to get the expected plant flow for that day and the next"
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-4">
          <Field
            id="fc-rain"
            label="Forecast rainfall for the day"
            unit="in"
            value={rainText}
            onChange={setRainText}
            step="0.05"
            hint="Sewershed-average daily total (plant day, morning to morning)"
          />
          <Field
            id="fc-ante"
            label="Rainfall over the prior 30 days"
            unit="in"
            value={anteText}
            onChange={setAnteText}
            step="0.1"
            hint={`${cls.name.split(' (')[0]} catchment · record ends ${fmt.date(latest.date)} at ${latest.ante_30d_in.toFixed(2)} in`}
          />
          <Field
            id="fc-level"
            label="Current dry-weather flow"
            unit="MGD"
            value={levelText}
            onChange={setLevelText}
            step="0.01"
            hint={`60-day dry-day median · ${latest.dry_level_mgd.toFixed(2)} MGD at end of record`}
          />

          <dl className="grid grid-cols-2 gap-4 border-t border-slate-200/70 pt-4 dark:border-slate-700/60">
            <Outcome label="Predicted flow, day of rain" value={result?.dayOf} range={result?.dayOfRange} strong />
            <Outcome label="Day after (no more rain)" value={result?.dayAfter} range={result?.dayAfterRange} />
          </dl>

          {!valid && (
            <p className="text-xs text-danger-dark dark:text-danger-light">Enter a number in each field.</p>
          )}
          {clipped && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              The curve stops at {maxRain} in; the prediction shown is for {maxRain} in.
            </p>
          )}
          {valid && rain > p99 && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Fewer than 1% of recorded wet days exceeded {p99} in. Above that the model has few examples,
              so treat this as a lower bound.
            </p>
          )}
        </div>

        <div className="min-w-0">
          <Legend items={legend} />
          <div className="mt-2 h-[300px] w-full sm:h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
                <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" />
                <ReferenceArea x1={p99} x2={maxRain} fill={theme.grid} fillOpacity={0.6} ifOverflow="hidden" />
                <XAxis
                  dataKey="rain"
                  type="number"
                  domain={[0, maxRain]}
                  ticks={[0, 1, 2, 3, 4, 5].filter((t) => t <= maxRain)}
                  tick={{ fill: theme.axis, fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: theme.grid }}
                  label={{
                    value: 'Forecast daily rainfall (in)',
                    position: 'insideBottom',
                    offset: -12,
                    style: { fill: theme.axis, fontSize: 11 },
                  }}
                />
                <YAxis
                  tick={{ fill: theme.axis, fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={52}
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => v.toFixed(1)}
                  label={{
                    value: 'Predicted flow (MGD)',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 18,
                    style: { fill: theme.axis, fontSize: 11, textAnchor: 'middle' },
                  }}
                />
                <Tooltip
                  content={<CurveTooltip colors={colors} classes={classes} />}
                  cursor={{ stroke: theme.axis, strokeWidth: 1, strokeDasharray: '3 3' }}
                />
                {valid && <ReferenceLine x={rain} stroke={theme.axis} strokeDasharray="4 3" />}
                {KEYS.map((k) => (
                  <Line
                    key={k}
                    dataKey={k}
                    name={SHORT[k]}
                    type="linear"
                    stroke={colors[k]}
                    strokeWidth={k === key ? 3 : 1.5}
                    strokeOpacity={k === key ? 1 : 0.55}
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 2, stroke: theme.surface }}
                    isAnimationActive={false}
                  />
                ))}
                {result && (
                  <ReferenceDot
                    x={rain}
                    y={result.dayOf}
                    r={6}
                    fill={colors[key]}
                    stroke={theme.surface}
                    strokeWidth={2}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
            Median of the model's response across {`>`}1,400 real starting days; the range under each
            prediction is the middle 50%. Shaded area: rainfall beyond 99% of recorded wet days. {curve.model}.
          </p>
        </div>
      </div>
    </Card>
  )
}

function Field({ id, label, unit, value, onChange, step, hint }) {
  return (
    <div>
      <label htmlFor={id} className="stat-label">
        {label}
      </label>
      <div className="mt-1 flex items-center gap-2">
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min="0"
          step={step}
          className="input w-28 tabular-nums"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <span className="text-xs text-slate-500 dark:text-slate-400">{unit}</span>
      </div>
      {hint && <div className="mt-1 text-[11px] leading-snug text-slate-400 dark:text-slate-500">{hint}</div>}
    </div>
  )
}

function Outcome({ label, value, range, strong }) {
  return (
    <div>
      <dt className="stat-label">{label}</dt>
      <dd className="mt-1 flex items-baseline gap-1">
        <span
          className={`font-semibold tabular-nums ${
            strong ? 'text-4xl text-primary dark:text-primary-light' : 'text-2xl text-slate-900 dark:text-slate-50'
          }`}
        >
          {value == null ? '—' : fmt.mgd(value)}
        </span>
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">MGD</span>
      </dd>
      {range && (
        <dd className="mt-0.5 text-xs tabular-nums text-slate-500 dark:text-slate-400">
          typical {fmt.mgd(range[0])} – {fmt.mgd(range[1])}
        </dd>
      )}
    </div>
  )
}

function CurveTooltip({ active, payload, colors, classes }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <TooltipShell
      title={`${d.rain.toFixed(2)} in of rain`}
      rows={KEYS.map((k, j) => ({
        label: classes[j].name,
        value: `${fmt.mgd(d[k])} MGD`,
        color: colors[k],
      }))}
    />
  )
}

function clampNum(text, lo, hi) {
  const v = Number.parseFloat(text)
  if (!Number.isFinite(v) || v < lo) return null
  return Math.min(v, hi)
}

/** Linear interpolation of ys over ascending xs; clamps at both ends. */
function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0]
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) {
      const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
      return ys[i - 1] + t * (ys[i] - ys[i - 1])
    }
  }
  return ys[ys.length - 1]
}

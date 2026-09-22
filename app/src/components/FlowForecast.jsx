import { useMemo } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, Legend, TooltipShell, Toggle } from './ui'
import { TOLERANCE_MGD, fmt, niceTicks } from '../lib/metrics'

const DAY = 86400000

/**
 * Actual vs predicted flow over the selected range.
 *
 * The x axis is a real time scale rather than a row index: the test record skips
 * 21 days inside its own span, and a categorical axis would close those gaps and
 * draw a continuous line through days the model never scored.
 */
export default function FlowForecast({ rows, theme, showBands, onToggleBands }) {
  const data = useMemo(
    () =>
      rows.map((r) => ({
        ...r,
        ts: Date.parse(`${r.date}T00:00:00Z`),
        band: [r.predicted - TOLERANCE_MGD, r.predicted + TOLERANCE_MGD],
      })),
    [rows],
  )

  const domain = useMemo(() => {
    if (!data.length) return ['auto', 'auto']
    let lo = Infinity
    let hi = -Infinity
    for (const d of data) {
      lo = Math.min(lo, d.actual, showBands ? d.band[0] : d.predicted)
      hi = Math.max(hi, d.actual, showBands ? d.band[1] : d.predicted)
    }
    const pad = Math.max((hi - lo) * 0.08, 0.1)
    return [Math.max(0, lo - pad), hi + pad]
  }, [data, showBands])

  const ticks = useMemo(() => monthTicks(data), [data])
  const yTicks = useMemo(
    () => (typeof domain[0] === 'number' ? niceTicks(domain[0], domain[1]) : undefined),
    [domain],
  )

  const legend = [
    { label: 'Actual', color: theme.actual },
    { label: 'Predicted', color: theme.predicted, dash: '6 4' },
    ...(showBands
      ? [{ label: `Predicted ±${TOLERANCE_MGD} MGD`, color: theme.band, shape: 'band' }]
      : []),
  ]

  return (
    <Card
      title="Flow forecast"
      subtitle={
        data.length
          ? `${fmt.date(rows[0].date)} – ${fmt.date(rows[rows.length - 1].date)} · ${data.length} days`
          : 'No days in range'
      }
      actions={<Toggle checked={showBands} onChange={onToggleBands} label="Tolerance band" />}
    >
      <Legend items={legend} />
      <div className="mt-2 h-[300px] w-full sm:h-[340px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
            <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              ticks={ticks}
              tickFormatter={(t) => fmt.dateShort(new Date(t).toISOString().slice(0, 10))}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: theme.grid }}
              minTickGap={12}
            />
            <YAxis
              domain={domain}
              ticks={yTicks}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={52}
              tickFormatter={(v) => v.toFixed(1)}
              label={{
                value: 'Flow (MGD)',
                angle: -90,
                position: 'insideLeft',
                offset: 18,
                style: { fill: theme.axis, fontSize: 11, textAnchor: 'middle' },
              }}
            />
            <Tooltip
              content={<FlowTooltip theme={theme} showBands={showBands} />}
              cursor={{ stroke: theme.axis, strokeWidth: 1, strokeDasharray: '3 3' }}
            />
            {showBands && (
              <Area
                dataKey="band"
                stroke="none"
                fill={theme.band}
                isAnimationActive={false}
                connectNulls={false}
                activeDot={false}
              />
            )}
            <Line
              dataKey="actual"
              name="Actual"
              stroke={theme.actual}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: theme.surface }}
              isAnimationActive={false}
              connectNulls={false}
            />
            <Line
              dataKey="predicted"
              name="Predicted"
              stroke={theme.predicted}
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2, stroke: theme.surface }}
              isAnimationActive={false}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function FlowTooltip({ active, payload, theme, showBands }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const within = Math.abs(d.error) <= TOLERANCE_MGD
  return (
    <TooltipShell
      title={fmt.date(d.date)}
      rows={[
        { label: 'Actual', value: `${fmt.mgd(d.actual)} MGD`, color: theme.actual },
        { label: 'Predicted', value: `${fmt.mgd(d.predicted)} MGD`, color: theme.predicted },
        { label: 'Error', value: `${fmt.signed(d.error)} MGD` },
        ...(showBands
          ? [
              {
                label: 'Tolerance band',
                value: `${fmt.mgd(d.band[0])} – ${fmt.mgd(d.band[1])}`,
                color: theme.band,
              },
            ]
          : []),
        { label: 'Rain', value: d.rain_in == null ? '—' : d.rain_in.toFixed(2) },
      ]}
      footer={
        <span className="flex flex-wrap items-center gap-x-2">
          <span style={{ color: within ? theme.good : theme.bad }}>
            {within ? `Within ±${TOLERANCE_MGD}` : `Outside ±${TOLERANCE_MGD}`}
          </span>
          <span>{'·'} {d.rain_wet ? 'Wet day' : 'Dry day'} (rain rule)</span>
          {d.screened_out && (
            <span className="font-medium">{'·'} Flagged, not scored</span>
          )}
        </span>
      }
    />
  )
}

/** One tick per month start present in the data, so labels never collide. */
function monthTicks(data) {
  if (!data.length) return []
  const seen = new Set()
  const out = []
  for (const d of data) {
    const key = d.date.slice(0, 7)
    if (!seen.has(key)) {
      seen.add(key)
      out.push(d.ts)
    }
  }
  // A very short range gets endpoints instead of a single lonely month tick.
  if (out.length < 2) return [data[0].ts, data[data.length - 1].ts]
  const span = data[data.length - 1].ts - data[0].ts
  if (span < 45 * DAY) {
    const step = Math.max(1, Math.floor(data.length / 6))
    return data.filter((_, i) => i % step === 0).map((d) => d.ts)
  }
  return out
}

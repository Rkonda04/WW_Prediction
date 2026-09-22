import { useMemo } from 'react'
import {
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, Legend, TooltipShell } from './ui'
import { TOLERANCE_MGD, computeMetrics, fmt, scored } from '../lib/metrics'

/**
 * Residuals over the selected range, one dot per scored day.
 *
 * Dots are colored by a reserved status pair (within / outside tolerance) rather
 * than by a categorical hue, and the reference lines carry the same meaning, so
 * the split is readable without relying on color alone.
 */
export default function ErrorAnalysis({ rows, theme }) {
  const data = useMemo(
    () =>
      scored(rows).map((r) => ({
        ...r,
        ts: Date.parse(`${r.date}T00:00:00Z`),
        within: Math.abs(r.error) <= TOLERANCE_MGD,
      })),
    [rows],
  )

  const stats = useMemo(() => computeMetrics(scored(rows)), [rows])

  const bound = useMemo(() => {
    const max = data.reduce((m, d) => Math.max(m, Math.abs(d.error)), 0.55)
    return Math.ceil(max * 10) / 10
  }, [data])

  const outside = data.length - data.filter((d) => d.within).length

  return (
    <Card
      title="Error analysis"
      subtitle={`Residuals (predicted − actual) · ${data.length} scored days`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Legend
          items={[
            { label: `Within ±${TOLERANCE_MGD} MGD`, color: theme.good },
            { label: `Outside ±${TOLERANCE_MGD} MGD`, color: theme.bad },
          ]}
        />
        <div className="text-xs text-slate-600 dark:text-slate-300">
          <span className="font-semibold tabular-nums text-slate-900 dark:text-slate-50">
            {fmt.pct(stats.within_tolerance ?? 0)}
          </span>{' '}
          within tolerance
          <span className="text-slate-400 dark:text-slate-500"> ({outside} outside)</span>
        </div>
      </div>

      <div className="mt-2 h-[240px] w-full sm:h-[268px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 34, bottom: 4, left: -8 }}>
            <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(t) => fmt.dateShort(new Date(t).toISOString().slice(0, 10))}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: theme.grid }}
              minTickGap={28}
            />
            <YAxis
              dataKey="error"
              type="number"
              domain={[-bound, bound]}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={52}
              tickFormatter={(v) => v.toFixed(1)}
              label={{
                value: 'Error (MGD)',
                angle: -90,
                position: 'insideLeft',
                offset: 18,
                style: { fill: theme.axis, fontSize: 11, textAnchor: 'middle' },
              }}
            />

            <ReferenceLine y={0} stroke={theme.axis} strokeWidth={1.5} />
            {[0.1, 0.3, 0.5].map((t) => (
              <ReferenceLine
                key={t}
                y={t}
                stroke={t === TOLERANCE_MGD ? theme.bad : theme.grid}
                strokeDasharray={t === TOLERANCE_MGD ? '5 4' : '2 4'}
                strokeWidth={t === TOLERANCE_MGD ? 1.5 : 1}
                label={{
                  value: `+${t}`,
                  position: 'right',
                  style: { fill: theme.axis, fontSize: 10 },
                }}
              />
            ))}
            {[0.1, 0.3, 0.5].map((t) => (
              <ReferenceLine
                key={`-${t}`}
                y={-t}
                stroke={t === TOLERANCE_MGD ? theme.bad : theme.grid}
                strokeDasharray={t === TOLERANCE_MGD ? '5 4' : '2 4'}
                strokeWidth={t === TOLERANCE_MGD ? 1.5 : 1}
                label={{
                  value: `−${t}`,
                  position: 'right',
                  style: { fill: theme.axis, fontSize: 10 },
                }}
              />
            ))}

            <Tooltip
              content={<ErrorTooltip theme={theme} />}
              cursor={{ stroke: theme.axis, strokeWidth: 1, strokeDasharray: '3 3' }}
            />
            <Scatter data={data} isAnimationActive={false}>
              {data.map((d) => (
                <Cell
                  key={d.date}
                  fill={d.within ? theme.good : theme.bad}
                  fillOpacity={0.85}
                  stroke={theme.surface}
                  strokeWidth={1.5}
                  r={4}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-200/70 pt-2.5 text-xs sm:grid-cols-4 dark:border-slate-700/60">
        {[
          ['Mean error', fmt.signed(stats.bias ?? 0)],
          ['MAE', fmt.mgd(stats.mae ?? 0)],
          ['RMSE', fmt.mgd(stats.rmse ?? 0)],
          ['R² (range)', fmt.r2(stats.r2)],
        ].map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-2 sm:block">
            <dt className="text-slate-500 dark:text-slate-400">{k}</dt>
            <dd className="font-semibold tabular-nums text-slate-800 dark:text-slate-100">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  )
}

function ErrorTooltip({ active, payload, theme }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <TooltipShell
      title={fmt.date(d.date)}
      rows={[
        { label: 'Error', value: `${fmt.signed(d.error)} MGD`, color: d.within ? theme.good : theme.bad },
        { label: 'Actual', value: `${fmt.mgd(d.actual)} MGD` },
        { label: 'Predicted', value: `${fmt.mgd(d.predicted)} MGD` },
        {
          label: 'Relative',
          value: `${fmt.signed((d.error / d.actual) * 100, 1)}%`,
        },
      ]}
      footer={d.within ? `Within ±${TOLERANCE_MGD} MGD` : `Outside ±${TOLERANCE_MGD} MGD`}
    />
  )
}

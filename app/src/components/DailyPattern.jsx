import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { TooltipShell } from './ui'
import { fmt } from '../lib/metrics'

/*
 * Textbook domestic-sewage hourly pattern (flow at each hour ÷ daily average):
 * low overnight, a morning peak as people wake, a smaller evening peak.
 * Not measured at Richmond: the plant records daily totals only.
 */
const SHAPE_RAW = [
  0.8, 0.68, 0.58, 0.52, 0.5, 0.55, 0.75, 1.05, 1.3, 1.4, 1.38, 1.3,
  1.22, 1.15, 1.08, 1.02, 1.0, 1.05, 1.15, 1.25, 1.28, 1.2, 1.05, 0.92,
]
const SHAPE_MEAN = SHAPE_RAW.reduce((a, b) => a + b, 0) / SHAPE_RAW.length
const SHAPE = SHAPE_RAW.map((v) => v / SHAPE_MEAN)

const MORNING = [6, 12]
const EVENING = [17, 23]

const hourLabel = (h) => {
  const hh = h % 24
  if (hh === 0) return '12AM'
  if (hh === 12) return '12PM'
  return hh < 12 ? `${hh}AM` : `${hh - 12}PM`
}

/**
 * Spreads a day's predicted flow over 24 hours. Only the dry-weather part follows
 * the household pattern; flow added by rain (inflow and infiltration) is spread
 * evenly, so a wet day's curve is flatter. The hourly average equals the daily flow.
 */
export function hourlyCurve(flow, dryLevel) {
  const base = Math.min(flow, dryLevel)
  const extra = flow - base
  const pts = SHAPE.map((m, h) => ({ hour: h, flow: base * m + extra }))
  return [...pts, { hour: 24, flow: pts[0].flow }]
}

const peakIn = (pts, [lo, hi]) =>
  pts.filter((p) => p.hour >= lo && p.hour <= hi).reduce((a, b) => (b.flow > a.flow ? b : a))

export default function DailyPattern({ flow, dryLevel, color, theme }) {
  const data = hourlyCurve(flow, dryLevel)
  const am = peakIn(data, MORNING)
  const pm = peakIn(data, EVENING)
  const low = data.reduce((a, b) => (b.flow < a.flow ? b : a))
  const gid = 'pattern-fill'
  // Headroom above the higher peak for its label, rounded up to a half MGD.
  const yTop = Math.ceil((Math.max(am.flow, pm.flow) * 1.1) * 2) / 2

  return (
    <section className="card mb-5 p-6">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-bold text-slate-700 dark:text-slate-100">Daily Flow Pattern</h3>
          <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">Typical hourly distribution</p>
        </div>
        <span className="rounded-lg bg-slate-100 px-2.5 py-1.5 text-[11px] font-bold text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          MGD
        </span>
      </div>

      <div className="mt-4 h-[200px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.35} />
                <stop offset="100%" stopColor={color} stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
            <ReferenceArea x1={MORNING[0] + 1} x2={MORNING[1] - 1} fill={color} fillOpacity={0.1} stroke="none" />
            <ReferenceArea x1={EVENING[0] + 1} x2={EVENING[1] - 1} fill={color} fillOpacity={0.1} stroke="none" />
            <XAxis
              dataKey="hour"
              type="number"
              domain={[0, 24]}
              ticks={[0, 6, 12, 18, 24]}
              tickFormatter={hourLabel}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: theme.grid }}
            />
            <YAxis
              domain={[0, yTop]}
              ticks={Array.from({ length: yTop * 2 + 1 }, (_, i) => i / 2)}
              tick={{ fill: theme.axis, fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={44}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <Tooltip content={<HourTooltip color={color} />} cursor={{ stroke: theme.axis, strokeDasharray: '3 3' }} />
            <Area
              dataKey="flow"
              type="monotone"
              stroke={color}
              strokeWidth={2.5}
              fill={`url(#${gid})`}
              isAnimationActive={false}
            />
            {[am, pm].map((p) => (
              <ReferenceDot
                key={p.hour}
                x={p.hour}
                y={p.flow}
                r={5}
                fill={theme.surface}
                stroke={color}
                strokeWidth={2}
                label={{ value: fmt.mgd(p.flow), position: 'top', fill: color, fontSize: 11, fontWeight: 700 }}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex justify-center gap-5 text-[11px] text-slate-500 dark:text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: color, opacity: 0.25 }} /> Peak hours
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border-2" style={{ borderColor: color }} /> Peak flow
        </span>
      </div>

      <div className="mt-4 flex items-start gap-2 rounded-xl bg-slate-50 px-4 py-3 text-xs text-slate-600 dark:bg-slate-800/50 dark:text-slate-300">
        <span aria-hidden="true">💡</span>
        <span>
          Morning peak ({hourLabel(am.hour)}): ~{fmt.mgd(am.flow)} MGD • Evening peak ({hourLabel(pm.hour)}): ~
          {fmt.mgd(pm.flow)} MGD • Overnight low ({hourLabel(low.hour)}): ~{fmt.mgd(low.flow)} MGD
        </span>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
        Illustrative only: a standard household wastewater pattern scaled to this day's predicted flow, not measured
        at Richmond (the plant records daily totals). Flow added by rain is spread evenly over the day. Pump cycling at
        the plant makes real effluent flow much spikier than this.
      </p>
    </section>
  )
}

function HourTooltip({ active, payload, color }) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload
  return <TooltipShell title={hourLabel(p.hour)} rows={[{ label: 'Typical flow', value: `${fmt.mgd(p.flow)} MGD`, color }]} />
}

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Collapsible, Legend, TooltipShell } from './ui'
import { fmt } from '../lib/metrics'
import { addDays, fallbackWeather, fetchWeather, hindcast, plantToday, rollForward } from '../lib/forecast'

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const weekday = (iso) => WEEKDAYS[new Date(`${iso}T00:00:00Z`).getUTCDay()]
const sum = (a) => a.reduce((s, v) => s + v, 0)
const FULL_DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const longDate = (iso) =>
  `${FULL_DAYS[new Date(`${iso}T00:00:00Z`).getUTCDay()]}, ${MONTHS[Number(iso.slice(5, 7)) - 1]} ${Number(iso.slice(8))}`

// Flow status, by how far the day sits above the dry-weather flow. Always shown with its label.
const LEVELS = [
  { key: 'normal', label: 'Normal', range: 'under 0.3 MGD above dry-weather', color: '#22C55E', below: 0.3 },
  { key: 'elevated', label: 'Elevated', range: '0.3 – 1.0 MGD above', color: '#EAB308', below: 1.0 },
  { key: 'high', label: 'High', range: 'over 1.0 MGD above', color: '#EF4444', below: Infinity },
]
const levelOf = (d, dry) => LEVELS.find((l) => d.flow - dry < l.below)
// Plain-language reasons for a day's flow, read off the same inputs the model used.
function reasonsFor(d) {
  const rain =
    d.rain > 0.1
      ? `${d.rain.toFixed(2)} in forecast — rain gets into the sewers and raises plant flow`
      : d.rainLag1 > 0.1
        ? `No rain this day, but ${d.rainLag1.toFixed(2)} in the day before is still draining to the plant`
        : 'No meaningful rain forecast — flow stays near the dry-weather level'
  const total = `${d.ante30.toFixed(2)} in over the prior 30 days`
  const ground =
    d.ante30 < 1
      ? `${total} — dry ground soaks up more, so a storm adds less flow`
      : d.ante30 < 4
        ? `${total} — normal ground conditions`
        : `${total} — wet ground, so a storm adds more flow`
  return [
    { icon: '🌧️', title: 'Rainfall', text: rain },
    { icon: '💧', title: 'Ground Wetness', text: ground },
  ]
}

const skyOf = (rain) => (rain > 0.1 ? '\u{1F327}\uFE0F' : rain > 0 ? '\u{1F326}\uFE0F' : '\u2600\uFE0F')

/**
 * Predicted plant flow for today and the next six days.
 *
 * Runs the Phase 7 production model forward in the browser (src/lib/forecast.js,
 * mirroring scripts/forecast_7day.py). Rain and temperature come from the
 * Open-Meteo forecast for Richmond and can be overwritten; the plant's current
 * flows are entered by the user because the flow record is not a live feed.
 */
export default function SevenDayForecast({ tool, theme }) {
  const { latest_state: latest, backtest } = tool

  const [weather, setWeather] = useState(null)
  const [status, setStatus] = useState('loading')
  const [rainText, setRainText] = useState(() => Array(tool.horizon).fill('0.00'))
  const [anteText, setAnteText] = useState('0.00')
  const [levelText, setLevelText] = useState(latest.dry_level_mgd.toFixed(2))
  const [flowText, setFlowText] = useState(['', '', ''])
  const [picked, setPicked] = useState(0)
  const [open, setOpen] = useState(false)

  const load = useCallback((signal) => {
    setStatus('loading')
    fetchWeather(tool, signal)
      .then((w) => {
        setWeather(w)
        setRainText(w.rainNext.map((v) => v.toFixed(2)))
        setAnteText(sum(w.rainPast.slice(-30)).toFixed(2))
        setStatus('live')
      })
      .catch((e) => {
        if (e.name !== 'AbortError') setStatus('offline')
      })
  }, [tool])

  useEffect(() => {
    const ctl = new AbortController()
    load(ctl.signal)
    return () => ctl.abort()
  }, [load])

  const level = num(levelText, 0.1, 20)
  const ante = num(anteText, 0, 60)
  const rains = rainText.map((t) => num(t, 0, 30))
  // A blank recent-flow box means "about the dry-weather flow", i.e. no storm still draining.
  const flows = flowText.map((t) => (t.trim() === '' ? level : num(t, 0.1, 30)))
  const valid = level != null && ante != null && rains.every((v) => v != null) && flows.every((v) => v != null)

  const [days, prev] = useMemo(() => {
    if (!valid) return [null, null]
    const base = weather ?? fallbackWeather(tool, plantToday(), ante)
    // Editing the prior-30-day total rescales the fetched history, keeping its day-to-day pattern.
    const fetched = sum(base.rainPast.slice(-30))
    let rainPast = base.rainPast
    if (Math.abs(fetched - ante) > 0.005) {
      rainPast = fetched > 0 ? base.rainPast.map((v) => (v * ante) / fetched) : fallbackWeather(tool, base.start, ante).rainPast
    }
    const next = rollForward(tool, {
      ...base,
      rainPast,
      rainNext: rains,
      level,
      flowLag1: flows[0],
      excess: flows.map((f) => f - level),
    })
    return [next, hindcast(tool, { ...base, rainPast, level }).reverse()]
  }, [valid, weather, tool, ante, level, rains.join(), flows.join()])

  const peak = days?.reduce((a, b) => (b.flow > a.flow ? b : a))
  const data = days?.map((d) => ({ ...d, label: `${weekday(d.date)} ${fmt.dateShort(d.date)}`, band: [d.lo, d.hi] }))
  const start = weather?.start ?? plantToday()
  const flat = rains.some((v) => v != null && v > tool.p99_rain_in)

  const setAt = (setter) => (i, v) => setter((prev) => prev.map((p, j) => (j === i ? v : p)))
  const setRain = setAt(setRainText)
  const setFlow = setAt(setFlowText)

  const day = days?.[picked]
  const dayDate = day?.date ?? addDays(start, picked)
  const dayLevel = day && levelOf(day, level)

  if (open) {
    const vsDry = day ? ((day.flow - level) / level) * 100 : null
    return (
      <div>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="mb-3 text-[13px] font-semibold text-primary dark:text-primary-light"
        >
          ← Back to Forecast
        </button>
        <h2 className="text-2xl font-bold tracking-[-0.3px] text-slate-700 dark:text-slate-100">{longDate(dayDate)}</h2>
        {day && (
          <div className="mb-5 mt-2 flex items-center gap-3">
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Forecast Rain</span>
            <span className="text-lg font-bold tabular-nums text-slate-700 dark:text-slate-100">{day.rain.toFixed(2)} in</span>
            <span className="text-2xl leading-none" aria-hidden="true">
              {skyOf(day.rain)}
            </span>
          </div>
        )}

        {day && (
          <section className="card mb-5 border-l-4 p-6" style={{ borderLeftColor: dayLevel.color }}>
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-semibold text-slate-600 dark:text-slate-300">Expected Daily Flow (MGD)</span>
              <span
                className="rounded-full px-3 py-1 text-[11px] font-bold uppercase tracking-[0.5px] text-slate-700 dark:text-slate-100"
                style={{ backgroundColor: `${dayLevel.color}33` }}
              >
                {dayLevel.label}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-6">
              <div>
                <div className="text-[56px] font-extrabold leading-none tracking-[-1.5px] tabular-nums text-slate-700 dark:text-slate-50">
                  {fmt.mgd(day.flow)}
                </div>
                <div className="mt-1 text-sm text-slate-500 dark:text-slate-400">Million Gallons</div>
              </div>
              <div className="min-w-[200px] flex-1 rounded-xl border border-slate-200 px-4 py-3 dark:border-slate-700">
                <div className="text-xs text-slate-500 dark:text-slate-400">Expected Range (80% confidence):</div>
                <div className="mt-0.5 text-base font-bold tabular-nums text-slate-700 dark:text-slate-100">
                  {fmt.mgd(day.lo)} – {fmt.mgd(day.hi)} MGD
                </div>
              </div>
            </div>
          </section>
        )}

        {day && (
          <section className="card mb-5 p-6">
            <h3 className="text-base font-bold text-slate-700 dark:text-slate-100">Why {dayLevel.label} Flow Expected?</h3>
            <ul className="mt-4 flex flex-col gap-4">
              {reasonsFor(day).map((r) => (
                <li key={r.title} className="flex items-start gap-4">
                  <span className="text-2xl leading-none" aria-hidden="true">
                    {r.icon}
                  </span>
                  <div>
                    <div className="text-sm font-bold text-slate-700 dark:text-slate-100">{r.title}</div>
                    <div className="mt-0.5 text-[13px] text-slate-500 dark:text-slate-400">{r.text}</div>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}

        {day && (
          <section className="card mb-5 flex items-center justify-between gap-4 p-6">
            <div>
              <div className="text-sm font-bold text-slate-700 dark:text-slate-100">Compared to Dry-Weather Flow</div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                ({fmt.mgd(level)} MGD typical on days without rain)
              </div>
            </div>
            <div className="text-3xl font-extrabold tabular-nums" style={{ color: dayLevel.key === 'normal' ? undefined : dayLevel.color }}>
              {fmt.signed(vsDry, 0)}%
            </div>
          </section>
        )}

        <section className="card mb-5 p-6">
          <h3 className="text-base font-bold text-slate-700 dark:text-slate-100">This Week and Plant Inputs</h3>
          <p className="mb-4 mt-0.5 text-xs text-slate-500 dark:text-slate-400">
            Set the plant's current flows and adjust each day's rain; every figure on this page follows.
          </p>
          <div className="flex flex-col gap-6">
            <div className="flex flex-col gap-4">
              <Field
                id="f7-level"
                label="Current dry-weather flow"
                unit="MGD"
                value={levelText}
                onChange={setLevelText}
                step="0.01"
                hint={`Typical flow on recent days without rain. Pre-filled with ${latest.dry_level_mgd.toFixed(2)} MGD from the end of the record (${fmt.date(latest.date)}) — replace with today's figure.`}
              />
              <div>
                <div className="stat-label">Measured flow, last 3 days</div>
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  {['Yesterday', '2 days ago', '3 days ago'].map((label, i) => (
                    <input
                      key={label}
                      aria-label={`Measured flow ${label.toLowerCase()} (MGD)`}
                      type="number"
                      inputMode="decimal"
                      min="0"
                      step="0.01"
                      placeholder={label}
                      className="input w-24 tabular-nums"
                      value={flowText[i]}
                      onChange={(e) => setFlow(i, e.target.value)}
                    />
                  ))}
                  <span className="text-xs text-slate-500 dark:text-slate-400">MGD</span>
                </div>
                <div className="mt-1 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
                  Leave blank if flow has been at the dry-weather level. Fill in after a storm so the forecast
                  starts from the elevated flow.
                </div>
              </div>
              <Field
                id="f7-ante"
                label="Rainfall over the prior 30 days"
                unit="in"
                value={anteText}
                onChange={setAnteText}
                step="0.1"
                hint={
                  weather
                    ? 'From Open-Meteo. Overwrite with the gauge total if it differs.'
                    : 'Assumed to have fallen more than 5 days ago.'
                }
              />

              <dl className="grid grid-cols-2 gap-4 border-t border-slate-200/70 pt-4 dark:border-slate-700/60">
                <div>
                  <dt className="stat-label">Peak day</dt>
                  <dd className="mt-1 flex items-baseline gap-1">
                    <span className="text-4xl font-semibold tabular-nums text-primary dark:text-primary-light">
                      {fmt.mgd(peak?.flow)}
                    </span>
                    <span className="text-xs font-medium text-slate-500 dark:text-slate-400">MGD</span>
                  </dd>
                  <dd className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                    {peak ? `${weekday(peak.date)} ${fmt.dateShort(peak.date)}` : '—'}
                  </dd>
                </div>
                <div>
                  <dt className="stat-label">7-day average</dt>
                  <dd className="mt-1 flex items-baseline gap-1">
                    <span className="text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-50">
                      {fmt.mgd(days && sum(days.map((d) => d.flow)) / days.length)}
                    </span>
                    <span className="text-xs font-medium text-slate-500 dark:text-slate-400">MGD</span>
                  </dd>
                </div>
              </dl>

              {!valid && (
                <p className="text-xs text-danger-dark dark:text-danger-light">
                  Enter a number for the dry-weather flow, the 30-day rainfall and each day's rain.
                </p>
              )}
              {flat && (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Fewer than 1% of recorded wet days exceeded {tool.p99_rain_in} in. Above that the model has few
                  examples, so treat that day's flow as a lower bound.
                </p>
              )}
            </div>

            <div className="min-w-0">
              <Legend
                items={[
                  { label: 'Predicted flow', color: theme.predicted },
                  { label: 'Range of 8 in 10 past forecasts', color: theme.predicted + '55', shape: 'band' },
                  { label: 'Dry-weather flow', color: theme.axis, dash: '4 3' },
                  { label: 'Forecast rain (right axis)', color: theme.curveDry, shape: 'band' },
                ]}
              />
              <div className="mt-2 h-[280px] w-full sm:h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={data ?? []} margin={{ top: 8, right: 0, bottom: 0, left: -8 }}>
                    <CartesianGrid stroke={theme.grid} strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="label"
                      tick={{ fill: theme.axis, fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: theme.grid }}
                      interval={0}
                    />
                    <YAxis
                      yAxisId="flow"
                      tick={{ fill: theme.axis, fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={52}
                      domain={[(min) => Math.floor(min * 5) / 5, (max) => Math.ceil(max * 5) / 5]}
                      tickFormatter={(v) => v.toFixed(1)}
                      label={{
                        value: 'Flow (MGD)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 18,
                        style: { fill: theme.axis, fontSize: 11, textAnchor: 'middle' },
                      }}
                    />
                    <YAxis
                      yAxisId="rain"
                      orientation="right"
                      tick={{ fill: theme.axis, fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={40}
                      domain={[0, (max) => Math.max(2, Math.ceil(max))]}
                      tickFormatter={(v) => `${v} in`}
                    />
                    <Tooltip content={<DayTooltip theme={theme} />} cursor={{ fill: theme.grid, fillOpacity: 0.4 }} />
                    <Bar yAxisId="rain" dataKey="rain" fill={theme.curveDry} barSize={14} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                    <Area yAxisId="flow" dataKey="band" stroke="none" fill={theme.predicted} fillOpacity={0.16} isAnimationActive={false} />
                    {level != null && <ReferenceLine yAxisId="flow" y={level} stroke={theme.axis} strokeDasharray="4 3" />}
                    <Line
                      yAxisId="flow"
                      dataKey="flow"
                      type="linear"
                      stroke={theme.predicted}
                      strokeWidth={2.5}
                      dot={{ r: 3.5, fill: theme.predicted, stroke: theme.surface, strokeWidth: 1.5 }}
                      activeDot={{ r: 5, strokeWidth: 2, stroke: theme.surface }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-3 overflow-x-auto">
                <table className="w-full min-w-[440px] text-xs">
                  <thead>
                    <tr className="text-left text-slate-500 dark:text-slate-400">
                      <th className="py-1 pr-3 font-medium">Day</th>
                      <th className="py-1 pr-3 font-medium">Forecast rain (in)</th>
                      <th className="py-1 pr-3 text-right font-medium">Predicted flow (MGD)</th>
                      <th className="py-1 text-right font-medium">Likely range (MGD)</th>
                    </tr>
                  </thead>
                  <tbody className="tabular-nums text-slate-700 dark:text-slate-200">
                    {rainText.map((text, i) => {
                      const d = days?.[i]
                      const date = d?.date ?? addDays(start, i)
                      return (
                        <tr key={date} className="border-t border-slate-200/70 dark:border-slate-700/60">
                          <td className="py-1.5 pr-3 whitespace-nowrap">
                            {weekday(date)} {fmt.dateShort(date)}
                            {i === 0 && <span className="text-slate-400 dark:text-slate-500"> · today</span>}
                          </td>
                          <td className="py-1.5 pr-3">
                            <input
                              aria-label={`Forecast rain ${fmt.dateShort(date)} (inches)`}
                              type="number"
                              inputMode="decimal"
                              min="0"
                              step="0.05"
                              className="input w-20 tabular-nums"
                              value={text}
                              onChange={(e) => setRain(i, e.target.value)}
                            />
                          </td>
                          <td className="py-1.5 pr-3 text-right text-sm font-semibold text-slate-900 dark:text-slate-50">
                            {fmt.mgd(d?.flow)}
                          </td>
                          <td className="py-1.5 text-right text-slate-500 dark:text-slate-400">
                            {d ? `${fmt.mgd(d.lo)} – ${fmt.mgd(d.hi)}` : '—'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              <p className="mt-2 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
                Each day is predicted from the day before, starting from the flows entered here. Tested on{' '}
                {backtest.n_start_days.toLocaleString()} past start days ({backtest.years.join(', ')}) using the rain
                that actually fell: average miss {backtest.mae_day1.toFixed(2)} MGD on day 1 and{' '}
                {backtest.mae_day7.toFixed(2)} MGD on day 7, against {backtest.mae_persistence_day7.toFixed(2)} MGD
                for assuming flow stays where it is. A wrong rain forecast adds to that. The likely range is where 8
                in 10 of those past forecasts landed, for wet and dry days separately. Rain is a calendar-day total
                from a weather model, not the plant gauge. {tool.model}.
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-slate-50 p-5 dark:border-slate-700 dark:bg-slate-800/40">
          <div className="text-xs font-bold text-slate-700 dark:text-slate-200">About This Prediction</div>
          <p className="mt-1 text-xs leading-snug text-slate-500 dark:text-slate-400">
            Daily totals predicted by the Phase 7 model, trained on Richmond plant flow from 2018 to 2024. The range
            is where 8 in 10 past predictions landed around the measured flow. Hourly flow is not predicted.
          </p>
        </section>
      </div>
    )
  }

  return (
    <div>
      <section className="card mb-6 p-6">
        <button type="button" className="block w-full text-left" onClick={() => setOpen(true)}>
          <div className="mb-5 flex items-center justify-between">
            <span className="rounded-[14px] bg-gradient-to-br from-primary to-[#6B1FA3] px-3.5 py-1.5 text-[11px] font-bold uppercase tracking-[1px] text-white">
              {picked === 0 ? 'Today' : weekday(dayDate)}
            </span>
            <span className="text-[13px] font-medium text-slate-500 dark:text-slate-400">{longDate(dayDate)}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <div className="flex flex-col items-center gap-1">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Forecast Rain</span>
              <span className="text-xl font-bold tabular-nums text-slate-700 dark:text-slate-100">
                {day ? `${day.rain.toFixed(2)} in` : '—'}
              </span>
              <span className="text-[52px] leading-none" aria-hidden="true">
                {skyOf(day?.rain ?? 0)}
              </span>
            </div>
            <div className="text-right">
              {dayLevel && (
                <div
                  className="mb-3 inline-flex items-center gap-2 rounded-full border-2 px-3.5 py-1.5 text-xs font-semibold text-slate-700 dark:text-slate-100"
                  style={{ borderColor: dayLevel.color, backgroundColor: `${dayLevel.color}26` }}
                >
                  <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: dayLevel.color }} />
                  {dayLevel.label}
                </div>
              )}
              <div className="flex items-baseline justify-end gap-1.5">
                <span className="text-[52px] font-extrabold leading-none tracking-[-1.5px] tabular-nums text-slate-700 dark:text-slate-50">
                  {fmt.mgd(day?.flow)}
                </span>
                <span className="text-xl font-semibold text-slate-500 dark:text-slate-400">MGD</span>
              </div>
              <span className="mt-1 block text-xs text-slate-500 dark:text-slate-400">
                Million Gallons per Day Expected
                {day && ` · likely ${fmt.mgd(day.lo)} – ${fmt.mgd(day.hi)}`}
              </span>
            </div>
          </div>
          <div className="mt-5 border-t border-slate-200 pt-4 text-center text-[13px] font-semibold text-primary dark:border-slate-700 dark:text-primary-light">
            Tap for detailed breakdown →
          </div>
        </button>
      </section>

      <h2 className="mb-4 ml-1 text-lg font-bold tracking-[-0.3px] text-slate-700 dark:text-slate-100">7-Day Forecast</h2>
      <div className="flex snap-x gap-2.5 overflow-x-auto pb-2">
        {rainText.map((_, i) => {
          const d = days?.[i]
          const date = d?.date ?? addDays(start, i)
          const on = i === picked
          const lv = d && levelOf(d, level)
          return (
            <button
              key={date}
              type="button"
              aria-pressed={on}
              onClick={() => setPicked(i)}
              className={`flex min-w-[76px] flex-1 snap-start flex-col items-center gap-1.5 rounded-2xl px-2.5 py-3.5 transition-transform hover:-translate-y-0.5 ${
                on
                  ? 'bg-gradient-to-b from-[#6B1FA3] to-primary text-white shadow-[0_4px_20px_rgba(74,14,120,0.3)]'
                  : 'bg-white text-slate-700 shadow-[0_2px_12px_rgba(0,0,0,0.06)] dark:bg-surface-dark dark:text-slate-100'
              }`}
            >
              <span className={`text-[10px] font-bold uppercase tracking-[0.5px] ${on ? 'text-white/90' : 'text-slate-500 dark:text-slate-400'}`}>
                {i === 0 ? 'Today' : weekday(date)}
              </span>
              <span className="text-lg font-extrabold">{Number(date.slice(8))}</span>
              <span className="my-1 text-[26px] leading-none" aria-hidden="true">
                {skyOf(d?.rain ?? 0)}
              </span>
              <span className={`text-[13px] font-semibold tabular-nums ${on ? 'text-white/80' : 'text-slate-500 dark:text-slate-400'}`}>
                {d ? `${d.rain.toFixed(2)} in` : '—'}
              </span>
              <span
                title={lv?.label}
                className="mt-1.5 h-3 w-3 rounded-full"
                style={{ backgroundColor: lv?.color ?? '#CBD5E1', boxShadow: on && lv ? `0 0 8px ${lv.color}` : 'none' }}
              />
              <span className="text-base font-extrabold tabular-nums">{fmt.mgd(d?.flow)}</span>
              <span className={`-mt-1 text-[10px] font-medium ${on ? 'text-white/70' : 'text-slate-500 dark:text-slate-400'}`}>MGD</span>
              {d && (
                <span
                  title="Likely range (80%)"
                  className={`whitespace-nowrap text-[10px] font-semibold tabular-nums ${on ? 'text-white/85' : 'text-slate-500 dark:text-slate-400'}`}
                >
                  {fmt.mgd(d.lo)}–{fmt.mgd(d.hi)}
                </span>
              )}
            </button>
          )
        })}
      </div>

      <div className="mt-5 text-center">
        {status === 'offline' && (
          <div className="mb-3 text-xs text-slate-500 dark:text-slate-400">
            Weather feed unavailable — enter rainfall by hand in the breakdown
          </div>
        )}
        <button
          type="button"
          onClick={() => load()}
          disabled={status === 'loading'}
          className="rounded-xl border-2 border-primary bg-white px-8 py-2.5 text-sm font-bold text-primary transition-colors hover:bg-primary hover:text-white disabled:opacity-50 dark:border-primary-light dark:bg-transparent dark:text-primary-light"
        >
          ↻ Refresh Forecast
        </button>
      </div>

      <div className="mt-8">
        <Collapsible title="View Previous 5 Days Predictions">
          <section className="card overflow-hidden">
            <div className="overflow-x-auto px-5 pt-5">
              <table className="w-full min-w-[420px] text-sm">
                <thead>
                  <tr className="border-b-2 border-slate-200 bg-slate-50 text-left text-[13px] text-slate-700 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-200">
                    <th className="px-3 py-3 font-semibold">Date</th>
                    <th className="px-3 py-3 font-semibold">Predicted Flow</th>
                    <th className="px-3 py-3 font-semibold">Confidence Interval (80%)</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {(prev ?? []).map((d) => (
                    <tr key={d.date} className="border-b border-slate-200 dark:border-slate-700">
                      <td className="whitespace-nowrap px-3 py-3 text-slate-700 dark:text-slate-200">{fmt.date(d.date)}</td>
                      <td className="px-3 py-3 text-base font-bold text-primary dark:text-primary-light">
                        {fmt.mgd(d.flow)} MGD
                      </td>
                      <td className="px-3 py-3 text-[13px] text-slate-500 dark:text-slate-400">
                        {fmt.mgd(d.lo)} – {fmt.mgd(d.hi)} MGD
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="px-5 pb-5 pt-3 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
              {prev
                ? `Calculated now from the rain recorded for these days and a dry-weather flow of ${fmt.mgd(level)} MGD; no measured plant flow is available for them. The interval is where 8 in 10 past predictions landed around the measured flow.`
                : 'Enter the plant inputs in the breakdown above to see these days.'}
            </p>
          </section>
        </Collapsible>
      </div>
    </div>
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

function DayTooltip({ active, payload, theme }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <TooltipShell
      title={`${d.label} · day ${d.lead}`}
      rows={[
        { label: 'Predicted flow', value: `${fmt.mgd(d.flow)} MGD`, color: theme.predicted },
        { label: 'Likely range', value: `${fmt.mgd(d.lo)} – ${fmt.mgd(d.hi)} MGD` },
        { label: 'Forecast rain', value: `${d.rain.toFixed(2)} in`, color: theme.curveDry },
      ]}
    />
  )
}

function num(text, lo, hi) {
  const v = Number.parseFloat(text)
  return Number.isFinite(v) && v >= lo && v <= hi ? v : null
}

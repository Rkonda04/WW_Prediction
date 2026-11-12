import { useCallback, useEffect, useMemo, useState } from 'react'
import { Collapsible } from './ui'
import { fmt } from '../lib/metrics'
import { fallbackWeather, fetchWeather, hindcast, plantToday, rollForward } from '../lib/forecast'

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
const skyOf = (rain) => (rain > 0.1 ? '\u{1F327}\uFE0F' : rain > 0 ? '\u{1F326}\uFE0F' : '\u2600\uFE0F')

/**
 * Predicted plant flow for today and the next six days.
 *
 * Runs the Phase 7 production model forward in the browser (src/lib/forecast.js,
 * mirroring scripts/forecast_7day.py) on the live Open-Meteo rain, temperature and
 * soil moisture (ECMWF) feeds for Richmond.
 * The plant flow record is not a live feed, so the run starts from the dry-weather
 * flow at the end of the record, with no storm still draining.
 */
export default function SevenDayForecast({ tool }) {
  const level = tool.latest_state.dry_level_mgd

  const [weather, setWeather] = useState(null)
  const [status, setStatus] = useState('loading')
  const [picked, setPicked] = useState(0)

  const load = useCallback((signal) => {
    setStatus('loading')
    fetchWeather(tool, signal)
      .then((w) => {
        setWeather(w)
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

  const [days, prev] = useMemo(() => {
    const base = weather ?? fallbackWeather(tool, plantToday())
    const next = rollForward(tool, { ...base, level, flowLag1: level, excess: [0, 0, 0] })
    return [next, hindcast(tool, { ...base, level }).reverse()]
  }, [weather, tool, level])

  const day = days[picked]
  const dayDate = day.date
  const dayLevel = levelOf(day, level)

  return (
    <div>
      <section className="card mb-6 p-6">
        <div>
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
        </div>
      </section>

      <h2 className="mb-4 ml-1 text-lg font-bold tracking-[-0.3px] text-slate-700 dark:text-slate-100">7-Day Forecast</h2>
      <div className="flex snap-x gap-2.5 overflow-x-auto pb-2">
        {days.map((d, i) => {
          const date = d.date
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
            {weather
              ? 'Weather feed did not respond — showing the forecast loaded earlier'
              : 'Weather feed unavailable — showing a no-rain forecast'}
          </div>
        )}
        {status !== 'loading' && weather?.soil && !weather.soil.live && (
          <div className="mb-3 text-xs text-slate-500 dark:text-slate-400">
            Soil moisture feed incomplete — using monthly normal soil moisture for missing days
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
                : 'Not available.'}
            </p>
          </section>
        </Collapsible>
      </div>
    </div>
  )
}

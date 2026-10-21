// Seven-day flow forecast: the Phase 7 production model, rolled forward in the
// browser. Mirrors rollout() in scripts/forecast_7day.py - day 1 is predicted
// from the measured state, each later day from the previous day's prediction.
// The dry-weather level and its slope are held at their start-day values.

const DAY_MS = 86400000

/** Model output (flow added above the dry-weather level) for one feature row. */
export function predictAdded(tool, x) {
  let out = tool.base_score
  for (const t of tool.trees) {
    let n = 0
    // XGBoost compares in float32; leaves carry their value in the threshold slot.
    while (t.l[n] !== -1) n = Math.fround(x[t.f[n]]) < Math.fround(t.t[n]) ? t.l[n] : t.r[n]
    out += t.t[n]
  }
  return out
}

/**
 * @param tool      bundle.forecast
 * @param start     ISO date of forecast day 1
 * @param level     current dry-weather flow (MGD)
 * @param slope     30-day change in that level (MGD)
 * @param flowLag1  measured flow the day before `start`
 * @param excess    flow minus level for the 3 days before `start`, most recent first
 * @param rainPast  daily rain (in) ending the day before `start`, oldest first, >= 30 days
 * @param rainNext  forecast rain for the 7 days
 * @param tminPrev  minimum temperature (F) the day before `start`
 * @param tminNext  forecast minimum temperature for the 7 days
 */
export function rollForward(tool, { start, level, slope = 0, flowLag1, excess, rainPast, rainNext, tminPrev, tminNext }) {
  const rain = [...rainPast, ...rainNext]
  const tmin = [tminPrev, ...tminNext]
  const p0 = rainPast.length
  const wet = (v) => v > tool.wet_rain_in
  const sum = (from, to) => rain.slice(from, to).reduce((s, v) => s + v, 0)

  let lagFlow = flowLag1
  let [e1, e2, e3] = excess
  const days = []
  for (let k = 0; k < rainNext.length; k++) {
    const i = p0 + k
    let dry = 0
    while (dry < 30 && i - 1 - dry >= 0 && !wet(rain[i - 1 - dry])) dry++
    if (i - 1 - dry < 0) dry = 30
    const date = addDays(start, k)
    const doy = dayOfYear(date)
    const ante30 = sum(i - 30, i)
    const f = {
      flow_lag1: lagFlow,
      excess_lag1: e1,
      excess_lag2: e2,
      excess_lag3: e3,
      excess_mean3: (e1 + e2 + e3) / 3,
      rainfall_t: rain[i],
      rain_lag1: rain[i - 1],
      rain_lag2: rain[i - 2],
      rain_sqrt_t: Math.sqrt(Math.max(rain[i], 0)),
      ante_5d: sum(i - 5, i),
      ante_30d: ante30,
      days_since_rain: dry,
      recent_level: level,
      level_slope30: slope,
      tmin_t: tmin[k + 1],
      tmin_lag1: tmin[k],
      freeze_2d: tmin[k + 1] < tool.freeze_f || tmin[k] < tool.freeze_f ? 1 : 0,
      doy_sin: Math.sin((2 * Math.PI * doy) / 365.25),
      doy_cos: Math.cos((2 * Math.PI * doy) / 365.25),
      rain_x_ante30: rain[i] * ante30,
      rain_lag1_x_ante30: rain[i - 1] * ante30,
      // Phase 7 also reads MRMS rain; there is one rain forecast, so it feeds both inputs.
      mrms_t: rain[i],
      mrms_lag1: rain[i - 1],
    }
    const added = predictAdded(tool, tool.features.map((name) => f[name]))
    const flow = level + added
    const isWet = wet(rain[i]) || wet(rain[i - 1])
    // Spread of (measured - predicted) at this lead day in the backtest.
    const r = tool.ranges[isWet ? 'wet' : 'dry'][k] ?? tool.ranges.all[k]
    days.push({
      date,
      lead: k + 1,
      rain: rain[i],
      tmin: tmin[k + 1],
      flow,
      added,
      wet: isWet,
      rainLag1: rain[i - 1],
      ante30,
      freeze: f.freeze_2d === 1,
      lo: flow + r.q10,
      hi: flow + r.q90,
    })
    lagFlow = flow
    ;[e1, e2, e3] = [added, e1, e2]
  }
  return days
}

export function addDays(iso, n) {
  return new Date(Date.parse(`${iso}T00:00:00Z`) + n * DAY_MS).toISOString().slice(0, 10)
}

function dayOfYear(iso) {
  return Math.round((Date.parse(`${iso}T00:00:00Z`) - Date.parse(`${iso.slice(0, 4)}-01-01T00:00:00Z`)) / DAY_MS) + 1
}

/** Today's date at the plant (US Central), independent of the viewer's time zone. */
export function plantToday() {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago' }).format(new Date())
}

const WEATHER_URL =
  'https://api.open-meteo.com/v1/forecast?latitude=29.5818&longitude=-95.7608' +
  '&daily=precipitation_sum,temperature_2m_min&past_days=36&forecast_days=7' +
  '&timezone=America%2FChicago&precipitation_unit=inch&temperature_unit=fahrenheit'

/**
 * Past 36 days and next 7 days of rain and minimum temperature for Richmond, TX
 * (same point and units as scripts/openmeteo_richmond_rainfall.py). Rainfall is
 * required; any missing minimum temperature falls back to the month's normal,
 * since the freeze effect it drives rarely matters here.
 */
export async function fetchWeather(tool, signal) {
  const res = await fetch(WEATHER_URL, { signal })
  if (!res.ok) throw new Error(`weather feed returned ${res.status}`)
  const { daily } = await res.json()
  const n = daily.time.length
  if (n < 43 || daily.precipitation_sum.some((v) => v == null)) {
    throw new Error('weather feed returned an incomplete rainfall series')
  }
  const tmin = daily.time.map(
    (iso, j) => daily.temperature_2m_min?.[j] ?? tool.tmin_normal_f_by_month[Number(iso.slice(5, 7)) - 1],
  )
  return {
    start: daily.time[n - 7],
    rainPast: daily.precipitation_sum.slice(0, n - 7),
    rainNext: daily.precipitation_sum.slice(n - 7),
    tminPast: tmin.slice(0, n - 7),
    tminPrev: tmin[n - 8],
    tminNext: tmin.slice(n - 7),
  }
}

/**
 * The model run over the days just before `start`, from the weather recorded for
 * them. No measured flow exists for those days, so the run starts `count` days
 * back at the dry-weather flow and each day follows from the one before.
 */
export function hindcast(tool, { start, level, rainPast, tminPast }, count = 5) {
  return rollForward(tool, {
    start: addDays(start, -count),
    level,
    flowLag1: level,
    excess: [0, 0, 0],
    rainPast: rainPast.slice(0, -count),
    rainNext: rainPast.slice(-count),
    tminPrev: tminPast[tminPast.length - count - 1],
    tminNext: tminPast.slice(-count),
  })
}

/**
 * Stand-in weather when the feed is unreachable: no rain in the forecast, normal
 * temperatures for the month, and the prior-30-day total spread evenly over days
 * 6-30 back (so nothing is assumed about the last 5 days).
 */
export function fallbackWeather(tool, start, ante30 = 0) {
  const month = (iso) => Number(iso.slice(5, 7)) - 1
  const normal = (iso) => tool.tmin_normal_f_by_month[month(iso)]
  return {
    start,
    rainPast: Array.from({ length: 36 }, (_, j) => (j >= 6 && j <= 30 ? ante30 / 25 : 0)),
    tminPast: Array.from({ length: 36 }, (_, j) => normal(addDays(start, j - 36))),
    rainNext: Array(tool.horizon).fill(0),
    tminPrev: normal(addDays(start, -1)),
    tminNext: Array.from({ length: tool.horizon }, (_, k) => normal(addDays(start, k))),
  }
}

// Metric helpers. The build script computes the headline numbers for the full
// test window; these recompute the same quantities for whatever range the user
// has filtered to, using identical definitions so the two never disagree.

export const TOLERANCE_MGD = 0.3

/** Coefficient of determination. Returns null when it is not defined. */
export function r2(rows) {
  if (rows.length < 2) return null
  const mean = rows.reduce((s, r) => s + r.actual, 0) / rows.length
  let ssRes = 0
  let ssTot = 0
  for (const r of rows) {
    ssRes += (r.actual - r.predicted) ** 2
    ssTot += (r.actual - mean) ** 2
  }
  if (ssTot === 0) return null
  return 1 - ssRes / ssTot
}

/** Full metric bundle for a set of rows. error = predicted - actual. */
export function computeMetrics(rows) {
  if (!rows.length) return { n: 0 }
  const n = rows.length
  let se = 0
  let ae = 0
  let bias = 0
  let ape = 0
  let within = 0
  for (const r of rows) {
    const e = r.error
    se += e * e
    ae += Math.abs(e)
    bias += e
    ape += Math.abs(e / r.actual)
    if (Math.abs(e) <= TOLERANCE_MGD) within += 1
  }
  return {
    n,
    r2: r2(rows),
    rmse: Math.sqrt(se / n),
    mae: ae / n,
    bias: bias / n,
    mape: (ape / n) * 100,
    within_tolerance: within / n,
  }
}

/** Rows the model actually scores: flagged (screened-out) days are excluded. */
export const scored = (rows) => rows.filter((r) => !r.screened_out)

/**
 * Saturation band for an index in [0,1]. Thresholds come straight from the
 * dashboard spec; `key` is a reserved status slot, not a categorical hue.
 */
export function saturationBand(index) {
  if (index == null) return BANDS.unknown
  if (index < 0.25) return BANDS.veryDry
  if (index < 0.5) return BANDS.dry
  if (index < 0.75) return BANDS.moderate
  return BANDS.wet
}

export const BANDS = {
  veryDry: { key: 'veryDry', label: 'Very Dry', color: '#E53935', range: '0.00 - 0.25' },
  dry: { key: 'dry', label: 'Dry', color: '#FB8C00', range: '0.25 - 0.50' },
  moderate: { key: 'moderate', label: 'Moderate', color: '#F9A825', range: '0.50 - 0.75' },
  wet: { key: 'wet', label: 'Wet', color: '#43A047', range: '0.75 - 1.00' },
  unknown: { key: 'unknown', label: 'No data', color: '#9AA1AB', range: '-' },
}

/**
 * Confidence in a given day's prediction.
 *
 * This is a presentation-layer heuristic, NOT a model output: the Phase 5 model
 * emits a point estimate with no per-day uncertainty. It keys off the one thing
 * the validation actually established - that accuracy splits by rain state, and
 * that dry days score worse than wet ones (R2 -0.241 vs 0.543).
 */
export function dayConfidence(row) {
  if (!row) return { level: 'UNKNOWN', label: 'Unknown', color: BANDS.unknown.color, note: '' }
  if (row.screened_out) {
    return {
      level: 'LOW',
      label: 'Low',
      color: '#E53935',
      note: 'Day flagged by the screening rule and excluded from scoring.',
    }
  }
  if (row.rain_wet) {
    return {
      level: 'MODERATE',
      label: 'Moderate',
      color: '#FB8C00',
      note: 'Wet day. The model explains about half the variance here (R² = 0.54).',
    }
  }
  return {
    level: 'LOW',
    label: 'Low',
    color: '#E53935',
    note: 'Dry day. The model scores worse than a flat mean on dry days (R² = −0.24).',
  }
}

export const fmt = {
  mgd: (v, d = 2) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(d)),
  signed: (v, d = 2) =>
    v == null || Number.isNaN(v) ? '—' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`,
  pct: (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(d)}%`),
  r2: (v) =>
    v == null || Number.isNaN(v) ? '—' : `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(3)}`,
  date: (iso) => {
    const [y, m, d] = iso.split('-')
    return `${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][
      Number(m) - 1
    ]} ${Number(d)}, ${y}`
  },
  dateShort: (iso) => {
    const [, m, d] = iso.split('-')
    return `${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][
      Number(m) - 1
    ]} ${Number(d)}`
  },
}

/**
 * Axis ticks on round numbers covering [lo, hi]. Recharts' automatic ticks land
 * on values like 2.7 and 0.8 once the domain is padded, which reads as noise on
 * a flow axis.
 */
export function niceTicks(lo, hi, target = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return undefined
  const raw = (hi - lo) / target
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag
  const ticks = []
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
    ticks.push(Number(t.toFixed(6)))
  }
  return ticks.length >= 2 ? ticks : undefined
}

/** Read a themed CSS custom property, so SVG marks follow the active theme. */
export function cssVar(name, fallback = '#000') {
  if (typeof window === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name)
  return v ? v.trim() : fallback
}

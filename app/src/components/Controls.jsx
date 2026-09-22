import { Toggle } from './ui'

const PRESETS = [
  { key: '30', label: '30d', days: 30 },
  { key: '60', label: '60d', days: 60 },
  { key: '90', label: '90d', days: 90 },
  { key: 'all', label: 'All', days: null },
]

/** Filters sit in one row above the charts and apply to every panel at once. */
export default function Controls({
  bounds,
  range,
  onRange,
  preset,
  onPreset,
  dark,
  onDark,
  selectedDate,
  onSelectedDate,
  dayOptions,
}) {
  return (
    <div className="card mb-4 flex flex-col gap-3 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-1.5">
          <span className="stat-label mr-1">Range</span>
          {PRESETS.map((p) => (
            <button
              key={p.key}
              type="button"
              onClick={() => onPreset(p.key)}
              className={`btn ${preset === p.key ? 'btn-active' : ''}`}
              aria-pressed={preset === p.key}
            >
              {p.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <label htmlFor="from" className="stat-label">
            From
          </label>
          <input
            id="from"
            type="date"
            className="input"
            value={range.from}
            min={bounds.min}
            max={range.to}
            onChange={(e) => onRange({ ...range, from: e.target.value })}
          />
          <label htmlFor="to" className="stat-label">
            To
          </label>
          <input
            id="to"
            type="date"
            className="input"
            value={range.to}
            min={range.from}
            max={bounds.max}
            onChange={(e) => onRange({ ...range, to: e.target.value })}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-1.5">
          <label htmlFor="day" className="stat-label">
            Day
          </label>
          <select
            id="day"
            className="input"
            value={selectedDate}
            onChange={(e) => onSelectedDate(e.target.value)}
          >
            {dayOptions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>
        <Toggle checked={dark} onChange={onDark} label="Dark" />
      </div>
    </div>
  )
}

export { PRESETS }

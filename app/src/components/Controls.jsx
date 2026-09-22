import { Toggle } from './ui'

/** One control: the day being inspected. Everything else was removed. */
export default function Controls({ dark, onDark, selectedDate, onSelectedDate, dayOptions }) {
  return (
    <div className="card mb-4 flex flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-2">
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
  )
}

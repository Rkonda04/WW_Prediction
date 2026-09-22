import { useEffect, useState } from 'react'

export function Card({ title, subtitle, actions, children, className = '', bodyClass = '' }) {
  return (
    <section className={`card flex flex-col ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-200/70 px-4 py-3 dark:border-slate-700/60">
          <div className="min-w-0">
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-subtitle mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={`flex-1 px-4 py-3 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, unit, hint, tone = 'default', footnote }) {
  const tones = {
    default: 'text-slate-900 dark:text-slate-50',
    good: 'text-accent-dark dark:text-accent-light',
    warn: 'text-warning-dark dark:text-warning-light',
    bad: 'text-danger-dark dark:text-danger-light',
  }
  return (
    <div className="min-w-0">
      <div className="stat-label">{label}</div>
      <div className={`mt-1 flex items-baseline gap-1 ${tones[tone]}`}>
        <span className="text-2xl font-semibold tabular-nums">{value}</span>
        {unit && <span className="text-xs font-medium opacity-70">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{hint}</div>}
      {footnote && (
        <div className="mt-1 text-[11px] leading-snug text-slate-400 dark:text-slate-500">
          {footnote}
        </div>
      )}
    </div>
  )
}

/** Status chips ship with a label, never color alone. */
export function Chip({ color, children, title }) {
  return (
    <span
      className="chip"
      title={title}
      style={{ backgroundColor: `${color}1A`, color, border: `1px solid ${color}55` }}
    >
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {children}
    </span>
  )
}

/** Shared tooltip shell so every chart's hover layer reads the same. */
export function TooltipShell({ title, rows, footer }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white/97 px-3 py-2 text-xs shadow-card backdrop-blur dark:border-slate-600 dark:bg-slate-800/97">
      <div className="mb-1.5 font-semibold text-slate-700 dark:text-slate-100">{title}</div>
      <table className="w-full border-separate border-spacing-y-0.5">
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td className="pr-3 align-middle">
                <span className="flex items-center gap-1.5 text-slate-500 dark:text-slate-400">
                  {r.color && (
                    <span
                      aria-hidden="true"
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ backgroundColor: r.color }}
                    />
                  )}
                  {r.label}
                </span>
              </td>
              <td className="text-right font-medium tabular-nums text-slate-800 dark:text-slate-100">
                {r.value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {footer && (
        <div className="mt-1.5 border-t border-slate-200 pt-1.5 text-[11px] text-slate-500 dark:border-slate-600 dark:text-slate-400">
          {footer}
        </div>
      )}
    </div>
  )
}

export function Legend({ items }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <li
          key={it.label}
          className="flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-300"
        >
          <svg width="16" height="8" aria-hidden="true" className="shrink-0">
            {it.shape === 'band' ? (
              <rect x="0" y="1" width="16" height="6" rx="1" fill={it.color} />
            ) : (
              <line
                x1="0"
                y1="4"
                x2="16"
                y2="4"
                stroke={it.color}
                strokeWidth="2"
                strokeDasharray={it.dash || undefined}
                strokeLinecap="round"
              />
            )}
          </svg>
          {it.label}
        </li>
      ))}
    </ul>
  )
}

export function Toggle({ checked, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="group flex items-center gap-2.5 text-xs font-medium text-slate-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary dark:text-slate-300"
    >
      <span
        className={`relative h-4 w-7 shrink-0 rounded-full transition-colors ${
          checked ? 'bg-primary' : 'bg-slate-300 dark:bg-slate-600'
        }`}
      >
        <span
          className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
            checked ? 'translate-x-3.5' : 'translate-x-0.5'
          }`}
        />
      </span>
      {label}
    </button>
  )
}

/** Theme state, persisted per viewer. Storage can throw, so every access is guarded. */
export function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof document === 'undefined') return false
    return document.documentElement.classList.contains('dark')
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try {
      localStorage.setItem('rrwwtf-theme', dark ? 'dark' : 'light')
    } catch (e) {
      /* storage blocked - theme still applies for this session */
    }
  }, [dark])

  return [dark, setDark]
}

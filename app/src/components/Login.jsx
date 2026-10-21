import { useState } from 'react'

// Credentials come from app/.env.local (git-ignored; see app/.env.example), so they
// stay out of the repository. They are still checked in the browser and end up in
// the built bundle: this keeps casual visitors out, but it is not real access control.
const USERNAME = import.meta.env.VITE_DASHBOARD_USER
const PASSWORD = import.meta.env.VITE_DASHBOARD_PASSWORD
const CONFIGURED = Boolean(USERNAME && PASSWORD)
// Login lives only in page state, so a refresh asks again.

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  function submit(e) {
    e.preventDefault()
    if (CONFIGURED && username === USERNAME && password === PASSWORD) {
      onLogin()
    } else {
      setError(true)
    }
  }

  const input =
    'mt-1.5 w-full rounded-lg border border-gray-300 bg-white px-3 py-2.5 text-base text-gray-900 outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 dark:border-gray-600 dark:bg-surface-dark dark:text-gray-100'

  return (
    <form onSubmit={submit} className="card mx-auto mt-8 max-w-[400px] px-7 py-8">
      <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Sign in</h2>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Enter your credentials to view the flow forecast.</p>

      <label className="mt-6 block text-xs font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        Username
        <input
          className={input}
          autoComplete="username"
          autoFocus
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </label>
      <label className="mt-4 block text-xs font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
        Password
        <div className="relative">
          <input
            className={`${input} pr-16`}
            type={showPassword ? 'text' : 'password'}
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <button
            type="button"
            onClick={() => setShowPassword((v) => !v)}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            aria-pressed={showPassword}
            className="absolute right-2 top-1/2 mt-[3px] -translate-y-1/2 rounded px-2 py-1 text-xs font-semibold normal-case tracking-normal text-primary hover:bg-primary/10 dark:text-primary-light"
          >
            {showPassword ? 'Hide' : 'Show'}
          </button>
        </div>
      </label>

      {!CONFIGURED && (
        <p role="alert" className="mt-4 text-sm font-medium text-danger">
          Sign-in is not configured: set VITE_DASHBOARD_USER and VITE_DASHBOARD_PASSWORD in app/.env.local.
        </p>
      )}
      {CONFIGURED && error && (
        <p role="alert" className="mt-4 text-sm font-medium text-danger">
          Incorrect username or password.
        </p>
      )}

      <button
        type="submit"
        className="mt-6 w-full rounded-lg bg-primary px-4 py-3 text-base font-bold text-white hover:bg-primary-dark"
      >
        Log in
      </button>
    </form>
  )
}

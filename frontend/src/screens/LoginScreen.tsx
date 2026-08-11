/**
 * Login / register. Both modes on one screen, matching the existing behaviour in
 * App.tsx (register then immediately login, because /auth/register returns a user
 * rather than a token).
 */

import { useState, type FormEvent } from 'react'
import { login as apiLogin, register as apiRegister } from '../lib/api'
import { Button, Field, Rule } from '../components/ui'
import { InlineError } from '../components/StateViews'

export function LoginScreen({ onLoggedIn }: { onLoggedIn: (token: string) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'register') await apiRegister(email, password)
      onLoggedIn(await apiLogin(email, password))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[minmax(0,1fr)_480px]">
      <div className="flex flex-col bg-accent p-8 text-ground sm:p-10">
        <div className="text-lg font-extrabold">SkinAIBot</div>
        <h1 className="mt-16 max-w-md text-4xl sm:text-5xl lg:mt-auto">
          Photograph a spot. Get an AI-assisted read in under a minute.
        </h1>
        <p className="mt-5 max-w-md text-sm leading-relaxed">
          SkinAIBot is a clinical decision-support prototype, not a doctor. It tells you what a spot
          most resembles and how confident the model is, so you know whether to book an appointment —
          and how soon.
        </p>
      </div>

      <div className="flex flex-col border-rule p-8 sm:p-10 lg:border-l-2">
        <div className="flex self-start border border-rule">
          {(['login', 'register'] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => {
                setMode(option)
                setError(null)
              }}
              aria-pressed={mode === option}
              className={`px-3 py-1.5 text-[13px] ${
                mode === option
                  ? 'bg-accent text-ground'
                  : 'bg-transparent text-ink hover:bg-neutral-200'
              } ${option === 'register' ? 'border-l border-rule' : ''}`}
            >
              {option === 'login' ? 'Log in' : 'Create account'}
            </button>
          ))}
        </div>

        <h2 className="mt-6 text-2xl">{mode === 'login' ? 'Welcome back' : 'Create your account'}</h2>

        <form className="mt-5 flex flex-col gap-3.5" onSubmit={handleSubmit}>
          <Field
            id="email"
            label="Email"
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
          <Field
            id="password"
            label="Password"
            type="password"
            required
            minLength={8}
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            hint="At least 8 characters."
          />
          {error ? <InlineError message={error} /> : null}
          <Button type="submit" variant="primary" disabled={submitting} className="justify-start">
            {submitting ? 'Please wait…' : mode === 'login' ? 'Log in' : 'Create account'}
          </Button>
        </form>

        <Rule className="mt-auto mb-4 hidden lg:block" />
        <p className="mt-8 text-xs leading-relaxed text-muted lg:mt-0">
          Photographs are stored by the application backend and are never used to train models
          without your written consent. You can delete every upload from Settings.
        </p>
      </div>
    </div>
  )
}

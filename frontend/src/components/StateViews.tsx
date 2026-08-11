import type { ReactNode } from 'react'
import { Button, IndeterminateBar } from './ui'

/** Nothing here yet — always says what to do next, never just "no data". */
export function EmptyState({
  title,
  body,
  action,
}: {
  title: string
  body: string
  action?: ReactNode
}) {
  return (
    <div className="border-2 border-rule p-8">
      <h2 className="text-xl">{title}</h2>
      <p className="mt-2 max-w-prose text-sm text-muted">{body}</p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  )
}

/**
 * A failure. `variant="config"` is for the missing-image-URL case: a setup gap,
 * not a medical or model outcome, so it must not read as an alarming result.
 */
export function ErrorState({
  title,
  message,
  variant = 'error',
  onRetry,
}: {
  title: string
  message: string
  variant?: 'error' | 'config'
  onRetry?: () => void
}) {
  const config = variant === 'config'
  return (
    <div
      role="alert"
      className={`border-2 p-6 ${config ? 'border-rule bg-surface' : 'border-accent bg-accent-100'}`}
    >
      <h2 className={`text-lg ${config ? 'text-ink' : 'text-accent-800'}`}>{title}</h2>
      <p className={`mt-2 max-w-prose text-sm ${config ? 'text-muted' : 'text-accent-900'}`}>
        {message}
      </p>
      {onRetry ? (
        <Button className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      ) : null}
    </div>
  )
}

export function InlineError({ message }: { message: string }) {
  return (
    <p role="alert" className="text-sm text-accent-700">
      {message}
    </p>
  )
}

/** Loading with a named step, so the wait is legible rather than a bare spinner. */
export function LoadingState({ title, steps }: { title: string; steps: string[] }) {
  return (
    <div className="border-2 border-rule p-6" aria-live="polite" aria-busy="true">
      <IndeterminateBar />
      <h2 className="mt-4 text-base">{title}</h2>
      <ul className="mt-2 list-disc pl-5 text-xs leading-relaxed text-muted">
        {steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ul>
    </div>
  )
}

/**
 * Image placeholder. Used wherever a real photograph is not available yet —
 * hatched, labelled, and the same box the real <img> occupies, so dropping real
 * images in later needs no layout change.
 */
export function PhotoPlaceholder({
  label = 'Lesion photograph',
  className = '',
  style,
}: {
  label?: string
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <div
      className={`grid place-items-center bg-neutral-300 text-center ${className}`}
      style={{
        backgroundImage:
          'repeating-linear-gradient(135deg, color-mix(in srgb, #201e1d 9%, transparent) 0 1px, transparent 1px 9px)',
        ...style,
      }}
    >
      <span className="kicker px-3 text-muted">{label}</span>
    </div>
  )
}

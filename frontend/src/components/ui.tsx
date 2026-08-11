/**
 * Shared primitives. Small on purpose: the design system is mostly rules,
 * alignment and type, so there is little to abstract.
 */

import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'
import type { ConfidenceStatus } from '../lib/api'

type ButtonVariant = 'primary' | 'secondary' | 'ghost'

export function Button({
  variant = 'secondary',
  className = '',
  ...props
}: { variant?: ButtonVariant } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`btn btn-${variant} ${className}`} {...props} />
}

export function Kicker({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`kicker text-accent ${className}`}>{children}</div>
}

export function SectionLabel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <h2 className={`kicker text-muted mb-3 ${className}`} style={{ fontWeight: 800 }}>
      {children}
    </h2>
  )
}

export function Rule({ className = '' }: { className?: string }) {
  return <hr className={`rule ${className}`} />
}

export function Field({
  label,
  hint,
  id,
  ...props
}: { label: string; hint?: string } & InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <input id={id} className="input" {...props} />
      {hint ? <p className="mt-1 text-xs text-muted">{hint}</p> : null}
    </div>
  )
}

const CONFIDENCE_TAG: Record<ConfidenceStatus, string> = {
  high: 'tag tag-neutral',
  moderate: 'tag tag-accent',
  low: 'tag tag-outline',
}

export function ConfidenceTag({ status }: { status: ConfidenceStatus }) {
  return (
    <span className={CONFIDENCE_TAG[status]}>
      {status.charAt(0).toUpperCase() + status.slice(1)} confidence
    </span>
  )
}

/** Determinate bar. `accent` marks the leading prediction; the rest stay ink. */
export function Bar({ value, accent = false }: { value: number; accent?: boolean }) {
  const width = `${Math.max(0, Math.min(1, value)) * 100}%`
  return (
    <div className="h-2.5 w-full bg-neutral-300" role="presentation">
      <div
        className={`h-full transition-[width] duration-500 ease-out ${accent ? 'bg-accent' : 'bg-neutral-800'}`}
        style={{ width }}
      />
    </div>
  )
}

/** Indeterminate bar for the analyzing state. */
export function IndeterminateBar() {
  return (
    <div className="h-2 w-full overflow-hidden bg-neutral-300">
      <div className="animate-indeterminate h-full w-1/3 bg-accent" />
    </div>
  )
}

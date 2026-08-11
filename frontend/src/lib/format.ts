import type { ConfidenceStatus, Prediction } from './api'

/**
 * Human-readable names for model labels. The mock AI service returns
 * `mock_class_a|b|c`; a trained model will return its own label set. Unknown
 * labels fall through to `prettifyLabel`, so a new class never breaks the UI.
 */
const LABEL_NAMES: Record<string, string> = {
  mock_class_a: 'Mock class A',
  mock_class_b: 'Mock class B',
  mock_class_c: 'Mock class C',
}

export function prettifyLabel(label: string): string {
  const spaced = label.replace(/[_-]+/g, ' ').trim()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export function labelName(label: string): string {
  return LABEL_NAMES[label] ?? prettifyLabel(label)
}

export function formatConfidence(confidence: number): string {
  return confidence.toFixed(2)
}

export function formatPercent(confidence: number): string {
  return `${Math.round(confidence * 100)}%`
}

export const CONFIDENCE_COPY: Record<ConfidenceStatus, string> = {
  high: 'The model strongly favours one possibility. It can still be wrong.',
  moderate:
    'One possibility leads, but the alternatives are not ruled out. A clinician should confirm.',
  low: 'The model could not settle on one possibility. Treat this read as inconclusive.',
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatDateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDate(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' })
}

/** Highest-confidence prediction, or null for an empty list. */
export function topPrediction(predictions: Prediction[]): Prediction | null {
  if (predictions.length === 0) return null
  return predictions.reduce((best, p) => (p.confidence > best.confidence ? p : best))
}

/** Descending by confidence, without mutating the input. */
export function sortedPredictions(predictions: Prediction[]): Prediction[] {
  return [...predictions].sort((a, b) => b.confidence - a.confidence)
}

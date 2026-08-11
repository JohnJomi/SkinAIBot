/**
 * The differential: every prediction the model returned, ordered, with the bar
 * carrying the comparison and the number carrying the precision. Wording is
 * deliberately "possible condition" / "model prediction" — never a diagnosis.
 */

import type { Prediction } from '../lib/api'
import { formatConfidence, labelName, sortedPredictions } from '../lib/format'
import { Bar } from './ui'

export function PredictionList({ predictions }: { predictions: Prediction[] }) {
  const ordered = sortedPredictions(predictions)
  if (ordered.length === 0) {
    return <p className="text-sm text-muted">The model returned no predictions for this photograph.</p>
  }
  return (
    <ul className="flex list-none flex-col gap-3 p-0">
      {ordered.map((prediction, index) => (
        <li key={prediction.label}>
          <div className="mb-1.5 flex items-baseline justify-between gap-4 text-sm">
            <span className={index === 0 ? 'font-extrabold' : ''}>{labelName(prediction.label)}</span>
            <span className="tabular-nums">{formatConfidence(prediction.confidence)}</span>
          </div>
          <Bar value={prediction.confidence} accent={index === 0} />
        </li>
      ))}
    </ul>
  )
}

/** Compact variant for table cells and rails. */
export function TopPredictionLine({ predictions }: { predictions: Prediction[] }) {
  const ordered = sortedPredictions(predictions)
  if (ordered.length === 0) return <span className="text-muted">No read</span>
  return (
    <span>
      {labelName(ordered[0].label)}{' '}
      <span className="text-muted tabular-nums">{formatConfidence(ordered[0].confidence)}</span>
    </span>
  )
}

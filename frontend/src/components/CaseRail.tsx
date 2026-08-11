/**
 * The fact rail for a single analysis. Shared by the result screen and the chat
 * screen so the chat is visibly part of the same case rather than a widget
 * floating next to it.
 */

import type { AnalysisRecord } from '../lib/analysisStore'
import { formatDateTime, labelName, topPrediction } from '../lib/format'
import { routeToHash } from '../hooks/useHashRoute'
import { ImagePreview } from './ImagePreview'
import { Kicker, Rule } from './ui'

function Row({ term, value }: { term: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-rule-soft py-2 last:border-b-0">
      <dt className="text-muted">{term}</dt>
      <dd className="m-0 text-right">{value}</dd>
    </div>
  )
}

export function CaseRail({
  record,
  imageSrc,
  footer,
}: {
  record: AnalysisRecord
  imageSrc?: string | null
  footer?: React.ReactNode
}) {
  const top = record.result ? topPrediction(record.result.predictions) : null
  return (
    <div>
      <ImagePreview
        src={imageSrc}
        heightClass="h-56"
        placeholderLabel="Lesion photograph"
        filename={record.upload.original_filename}
        sizeBytes={record.upload.size_bytes}
      />
      <Kicker className="mt-4">Case {record.analysis_id}</Kicker>
      {top ? (
        <>
          <h2 className="mt-3 text-xl">{labelName(top.label)}</h2>
          <p className="mt-1 text-xs text-muted">
            Model prediction · {top.confidence.toFixed(2)} ·{' '}
            {record.result?.confidence_status} confidence
          </p>
        </>
      ) : (
        <h2 className="mt-3 text-xl">No usable read</h2>
      )}
      <Rule className="my-5" />
      <dl className="m-0 text-[13px]">
        <Row term="Uploaded" value={formatDateTime(record.created_at)} />
        <Row term="Site" value={record.body_site ?? 'Not recorded'} />
        <Row term="Status" value={record.result?.status ?? 'failed'} />
        <Row term="Model" value={record.result?.model_version ?? '—'} />
      </dl>
      {footer ? <div className="mt-5">{footer}</div> : null}
      <Rule className="my-5" />
      <a
        href={routeToHash({ name: 'results' })}
        className="text-sm text-ink no-underline hover:text-accent"
      >
        All results →
      </a>
    </div>
  )
}

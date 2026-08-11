/**
 * A single analysis — the approved case-sheet layout: fact rail on the left,
 * verdict, differential, explanation, recommendation and the chat entry on the
 * right. Wording never asserts a diagnosis.
 */

import { RailLayout } from '../components/AppShell'
import { CaseRail } from '../components/CaseRail'
import { PredictionList } from '../components/PredictionList'
import { EmptyState, ErrorState } from '../components/StateViews'
import { Button, ConfidenceTag, Rule, SectionLabel } from '../components/ui'
import { ChatPanel } from '../components/ChatPanel'
import { useChat } from '../hooks/useChat'
import type { Route } from '../hooks/useHashRoute'
import type { AnalysisRecord } from '../lib/analysisStore'
import { CONFIDENCE_COPY, labelName, topPrediction } from '../lib/format'

export function ResultScreen({
  record,
  token,
  navigate,
}: {
  record: AnalysisRecord | null
  token: string | null
  navigate: (route: Route) => void
}) {
  const chat = useChat(record?.analysis_id ?? null, token)

  if (!record) {
    return (
      <div className="p-8">
        <EmptyState
          title="That analysis is not on this device"
          body="Analyses are indexed in this browser until the backend exposes an analyses endpoint. Open the Results list to see what is available here."
          action={<Button onClick={() => navigate({ name: 'results' })}>Go to Results</Button>}
        />
      </div>
    )
  }

  const result = record.result
  const top = result ? topPrediction(result.predictions) : null

  return (
    <RailLayout
      rail={
        <CaseRail
          record={record}
          footer={
            <Button
              className="w-full justify-start"
              onClick={() => navigate({ name: 'chat', analysisId: record.analysis_id })}
            >
              Open in full chat
            </Button>
          }
        />
      }
    >
      {!result ? (
        <ErrorState
          title="No usable read for this photograph"
          message={
            record.error ??
            'The model returned no result. This usually means the photograph was blurred, too far away, or poorly lit.'
          }
          onRetry={() => navigate({ name: 'upload' })}
        />
      ) : (
        <>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <SectionLabel>Possible condition · model prediction</SectionLabel>
              <h1 className="text-3xl sm:text-4xl">{top ? labelName(top.label) : 'No prediction'}</h1>
            </div>
            <ConfidenceTag status={result.confidence_status} />
          </div>
          <p className="mt-3.5 max-w-2xl text-[15px]">{CONFIDENCE_COPY[result.confidence_status]}</p>

          <Rule className="my-6" />
          <SectionLabel>What the model considered</SectionLabel>
          <div className="max-w-xl">
            <PredictionList predictions={result.predictions} />
          </div>
          <p className="mt-3 text-[11px] text-muted">
            Model {result.model_version} · analysis {result.analysis_id} · status {result.status}.
            Confidence describes the model, not your risk.
          </p>

          <Rule className="my-6" />
          <div className="grid gap-8 sm:grid-cols-2">
            <div>
              <SectionLabel>In plain terms</SectionLabel>
              <p className="text-sm">{result.explanation}</p>
            </div>
            <div>
              <SectionLabel>What to do next</SectionLabel>
              <p className="text-sm">{result.recommendation}</p>
            </div>
          </div>

          <div className="mt-6 border-2 border-accent bg-accent-100 p-4">
            <p className="text-xs leading-relaxed text-accent-900">{result.disclaimer}</p>
          </div>

          <Rule className="my-6" />
          <SectionLabel>Ask about this result</SectionLabel>
          <ChatPanel
            messages={chat.messages}
            sending={chat.sending}
            error={chat.error}
            onSend={chat.send}
            emptyTitle="Questions about this read?"
            emptyBody="The assistant can explain what the prediction and confidence figure mean, and what changes are worth acting on. It cannot examine you or diagnose."
          />
        </>
      )}
    </RailLayout>
  )
}

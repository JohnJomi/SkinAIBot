/**
 * Chat as a full screen, anchored to a case when one is selected. Same rail as
 * the result screen, so the assistant reads as part of the application.
 */

import { Page, RailLayout } from '../components/AppShell'
import { CaseRail } from '../components/CaseRail'
import { ChatPanel } from '../components/ChatPanel'
import { EmptyState } from '../components/StateViews'
import { Button, SectionLabel } from '../components/ui'
import { useChat } from '../hooks/useChat'
import type { Route } from '../hooks/useHashRoute'
import { routeToHash } from '../hooks/useHashRoute'
import type { AnalysisRecord } from '../lib/analysisStore'
import { formatDate } from '../lib/format'

export function ChatScreen({
  record,
  records,
  token,
  navigate,
}: {
  record: AnalysisRecord | null
  records: AnalysisRecord[]
  token: string | null
  navigate: (route: Route) => void
}) {
  const chat = useChat(record?.analysis_id ?? null, token)

  if (!record) {
    return (
      <Page>
        <h1 className="text-3xl">Chat</h1>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          The assistant answers questions about an analysis it has already produced. Pick a case to
          give it context.
        </p>
        <div className="mt-6">
          {records.length === 0 ? (
            <EmptyState
              title="No analyses to talk about yet"
              body="Upload a photograph first — the assistant needs a result to explain."
              action={
                <Button variant="primary" onClick={() => navigate({ name: 'upload' })}>
                  Upload a photograph
                </Button>
              }
            />
          ) : (
            <ul className="m-0 flex max-w-xl list-none flex-col p-0">
              {records.map((item) => (
                <li key={item.analysis_id} className="border-b border-rule-soft">
                  <a
                    href={routeToHash({ name: 'chat', analysisId: item.analysis_id })}
                    className="flex justify-between gap-4 py-3 text-sm text-ink no-underline hover:text-accent"
                  >
                    <span>
                      {formatDate(item.created_at)} · {item.body_site ?? 'Site not recorded'}
                    </span>
                    <span className="text-muted">{item.analysis_id}</span>
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      </Page>
    )
  }

  return (
    <RailLayout
      rail={
        <CaseRail
          record={record}
          imageSrc={record.display_image_url}
          footer={
            <Button
              className="w-full justify-start"
              onClick={() => navigate({ name: 'result', analysisId: record.analysis_id })}
            >
              Back to the full result
            </Button>
          }
        />
      }
    >
      <SectionLabel>Talking about case {record.analysis_id}</SectionLabel>
      <div className="flex min-h-[520px] flex-col">
        <ChatPanel
          messages={chat.messages}
          sending={chat.sending}
          error={chat.error}
          onSend={chat.send}
        />
      </div>
    </RailLayout>
  )
}

/**
 * Chat thread and composer.
 *
 * Every assistant reply renders `disclaimer` from the response — the contract
 * requires it to accompany the reply, so it is part of the message, not a
 * one-off banner. Messages are ruled blocks with an author label rather than
 * chat bubbles: same visual language as the rest of the application.
 */

import { useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../hooks/useChat'
import { Button } from './ui'
import { InlineError } from './StateViews'

const SUGGESTIONS = [
  'What does this confidence figure mean?',
  'What should I watch for?',
  'Should I see someone sooner?',
]

function timeOf(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime())
    ? ''
    : date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

export function ChatPanel({
  messages,
  sending,
  error,
  onSend,
  emptyTitle = 'Ask about this analysis',
  emptyBody = 'The assistant can explain the prediction, what the confidence figure means, and what to watch for. It cannot examine you or give a diagnosis.',
}: {
  messages: ChatMessage[]
  sending: boolean
  error: string | null
  onSend: (text: string) => void
  emptyTitle?: string
  emptyBody?: string
}) {
  const [draft, setDraft] = useState('')
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.parentElement?.scrollTo({
      top: endRef.current.parentElement.scrollHeight,
      behavior: 'smooth',
    })
  }, [messages.length, sending])

  const submit = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || sending) return
    onSend(trimmed)
    setDraft('')
  }

  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex-1 overflow-y-auto" aria-live="polite">
        {messages.length === 0 ? (
          <div className="border-2 border-rule p-6">
            <h2 className="text-lg">{emptyTitle}</h2>
            <p className="mt-2 max-w-prose text-sm text-muted">{emptyBody}</p>
          </div>
        ) : (
          <ul className="flex list-none flex-col gap-5 p-0">
            {messages.map((message) => (
              <li
                key={message.id}
                className="animate-enter border-b border-rule-soft pb-4 last:border-b-0"
              >
                <h3
                  className={`kicker mb-1.5 ${message.author === 'assistant' ? 'text-accent' : 'text-muted'}`}
                >
                  {message.author === 'assistant' ? 'SkinAIBot' : 'You'} · {timeOf(message.at)}
                </h3>
                <p className="text-[15px]">{message.text}</p>
                {message.disclaimer ? (
                  <p className="mt-2.5 text-[11px] leading-relaxed text-muted">
                    {message.disclaimer}
                  </p>
                ) : null}
              </li>
            ))}
            {sending ? (
              <li className="text-sm text-muted" aria-busy="true">
                SkinAIBot is writing a reply…
              </li>
            ) : null}
          </ul>
        )}
        <div ref={endRef} />
      </div>

      <div className="mt-6">
        {error ? <div className="mb-3">{<InlineError message={error} />}</div> : null}
        {messages.length === 0 ? (
          <div className="mb-3 flex flex-wrap gap-2">
            {SUGGESTIONS.map((suggestion) => (
              <Button key={suggestion} className="text-[13px]" onClick={() => submit(suggestion)}>
                {suggestion}
              </Button>
            ))}
          </div>
        ) : null}
        <form
          className="flex gap-2.5"
          onSubmit={(event) => {
            event.preventDefault()
            submit(draft)
          }}
        >
          <label className="sr-only" htmlFor="chat-message">
            Your question
          </label>
          <input
            id="chat-message"
            className="input"
            placeholder="Ask a question about this analysis"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            disabled={sending}
          />
          <Button type="submit" variant="primary" disabled={sending || draft.trim().length === 0}>
            Send
          </Button>
        </form>
      </div>
    </div>
  )
}

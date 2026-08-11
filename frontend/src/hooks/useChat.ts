/**
 * Chat against POST /api/v1/chat.
 *
 * The contract is stateless per request: `{session_id, message, analysis_id?}` in,
 * `{response, disclaimer}` out. There is no chat-history endpoint and no
 * session-creation endpoint, so threads live in component state for the life of
 * the page and `session_id` is generated client-side. Reloading loses them.
 *
 * State is keyed by analysis id. The screen stays mounted while the selected
 * case changes, so a single thread would show case A's transcript under case B
 * and reuse A's session id for B's questions. Keying also means returning to A
 * restores A's own thread rather than inheriting B's.
 */

import { useCallback, useMemo, useRef, useState } from 'react'
import { ApiError, RequestTimeoutError, sendChatMessage } from '../lib/api'
import { USE_MOCK_DATA } from '../lib/config'
import { mockChatReplies, MOCK_DISCLAIMER } from '../lib/mockData'
import { newAnalysisId } from '../lib/analysisStore'

export interface ChatMessage {
  id: string
  author: 'user' | 'assistant'
  text: string
  /** Present on assistant messages: the contract requires it on every reply. */
  disclaimer?: string
  at: string
}

export interface ChatState {
  messages: ChatMessage[]
  sending: boolean
  error: string | null
  send: (text: string) => Promise<void>
  sessionId: string
}

/** Key for the thread with no case selected, so a null id is still safe. */
const NO_ANALYSIS_KEY = '__no_analysis__'

export function useChat(analysisId: string | null, token: string | null): ChatState {
  const key = analysisId ?? NO_ANALYSIS_KEY

  // One session id per analysis, created on first use and stable for the rest
  // of the page's life — held in a ref so switching cases never regenerates it.
  const sessionIds = useRef<Record<string, string>>({})
  const sessionId = useMemo(() => {
    const existing = sessionIds.current[key]
    if (existing) return existing
    const created = newAnalysisId()
    sessionIds.current[key] = created
    return created
  }, [key])

  const [threads, setThreads] = useState<Record<string, ChatMessage[]>>({})
  const [errors, setErrors] = useState<Record<string, string | null>>({})
  // Which case has a request in flight, so a send in A does not disable B.
  const [pendingKey, setPendingKey] = useState<string | null>(null)
  const mockIndex = useRef(0)

  const messages = threads[key] ?? []
  const error = errors[key] ?? null
  const sending = pendingKey === key

  const append = useCallback(
    (threadKey: string, message: ChatMessage) => {
      setThreads((current) => ({
        ...current,
        [threadKey]: [...(current[threadKey] ?? []), message],
      }))
    },
    [],
  )

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || sending) return
      setErrors((current) => ({ ...current, [key]: null }))
      append(key, {
        id: newAnalysisId(),
        author: 'user',
        text: trimmed,
        at: new Date().toISOString(),
      })
      setPendingKey(key)
      try {
        if (USE_MOCK_DATA) {
          await new Promise((resolve) => setTimeout(resolve, 700))
          const reply = mockChatReplies[mockIndex.current % mockChatReplies.length]
          mockIndex.current += 1
          append(key, {
            id: newAnalysisId(),
            author: 'assistant',
            text: reply,
            disclaimer: MOCK_DISCLAIMER,
            at: new Date().toISOString(),
          })
          return
        }
        const reply = await sendChatMessage(
          { session_id: sessionId, message: trimmed, analysis_id: analysisId ?? undefined },
          token,
        )
        append(key, {
          id: newAnalysisId(),
          author: 'assistant',
          text: reply.response,
          disclaimer: reply.disclaimer,
          at: new Date().toISOString(),
        })
      } catch (err) {
        const message =
          err instanceof RequestTimeoutError
            ? `${err.message} Please try again in a moment.`
            : err instanceof ApiError
              ? err.message
              : err instanceof Error
                ? err.message
                : 'The assistant could not be reached.'
        setErrors((current) => ({ ...current, [key]: message }))
      } finally {
        // Only clear if this send is still the in-flight one.
        setPendingKey((current) => (current === key ? null : current))
      }
    },
    [analysisId, append, key, sending, sessionId, token],
  )

  return { messages, sending, error, send, sessionId }
}

/**
 * Chat against POST /api/v1/chat.
 *
 * The contract is stateless per request: `{session_id, message, analysis_id?}` in,
 * `{response, disclaimer}` out. There is no chat-history endpoint and no
 * session-creation endpoint, so the thread lives in component state for the life
 * of the page and `session_id` is generated client-side. Reloading loses the
 * transcript — noted in ASSUMPTIONS_AND_BLOCKERS.md.
 */

import { useCallback, useMemo, useRef, useState } from 'react'
import { ApiError, sendChatMessage } from '../lib/api'
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

export function useChat(analysisId: string | null, token: string | null): ChatState {
  const sessionId = useMemo(() => newAnalysisId(), [])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mockIndex = useRef(0)

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || sending) return
      setError(null)
      setMessages((current) => [
        ...current,
        { id: newAnalysisId(), author: 'user', text: trimmed, at: new Date().toISOString() },
      ])
      setSending(true)
      try {
        if (USE_MOCK_DATA) {
          await new Promise((resolve) => setTimeout(resolve, 700))
          const reply = mockChatReplies[mockIndex.current % mockChatReplies.length]
          mockIndex.current += 1
          setMessages((current) => [
            ...current,
            {
              id: newAnalysisId(),
              author: 'assistant',
              text: reply,
              disclaimer: MOCK_DISCLAIMER,
              at: new Date().toISOString(),
            },
          ])
          return
        }
        const reply = await sendChatMessage(
          { session_id: sessionId, message: trimmed, analysis_id: analysisId ?? undefined },
          token,
        )
        setMessages((current) => [
          ...current,
          {
            id: newAnalysisId(),
            author: 'assistant',
            text: reply.response,
            disclaimer: reply.disclaimer,
            at: new Date().toISOString(),
          },
        ])
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : err instanceof Error
              ? err.message
              : 'The assistant could not be reached.',
        )
      } finally {
        setSending(false)
      }
    },
    [analysisId, sending, sessionId, token],
  )

  return { messages, sending, error, send, sessionId }
}

/**
 * API client.
 *
 * The first four functions (register, login, getCurrentUser, uploadImage) are
 * the repository's existing implementation, apart from importing API_BASE_URL
 * from lib/config.ts and going through `fetchWithTimeout`. Everything below the
 * marked section maps 1:1 onto endpoints that already exist in
 * backend/app/main.py — no endpoint, field or status code is invented here.
 *
 * Every request is bounded. A hung backend or a dropped connection otherwise
 * leaves `fetch` pending forever and the UI stuck in its loading state with no
 * way out.
 */

import { API_BASE_URL } from './config'

export interface User {
  id: string
  email: string
  is_active: boolean
}

export interface UploadRecord {
  id: string
  original_filename: string
  content_type: string
  size_bytes: number
  created_at: string
  /**
   * Absolute URL for POST /api/v1/analyze. The AI service fetches this
   * server-side, so it resolves on the server network - the browser generally
   * cannot load it. Optional so an older backend that omits it is a handled
   * error rather than an undefined slipping into the request.
   */
  analysis_image_url?: string
  /** Absolute URL the browser can load in an <img>. Never sent to analyze. */
  display_image_url?: string
}

/**
 * Request budgets, in milliseconds.
 *
 * DEFAULT covers the cheap authenticated calls. UPLOAD is longer because the
 * backend accepts images up to 10 MB and a slow uplink is not an error. AI is
 * longer still, and deliberately above the backend's own 30s AI_REQUEST_TIMEOUT
 * (backend/app/ai/client.py): the backend should be allowed to time out first
 * and answer with a structured error, so this abort is only the last resort
 * for a backend that never replies at all.
 */
const DEFAULT_TIMEOUT_MS = 30_000
const UPLOAD_TIMEOUT_MS = 60_000
const AI_TIMEOUT_MS = 45_000

/** A request that was aborted because it exceeded its budget. */
export class RequestTimeoutError extends Error {
  readonly timeoutMs: number

  constructor(timeoutMs: number) {
    super(
      `The request took longer than ${Math.round(timeoutMs / 1000)} seconds and was stopped.`,
    )
    this.name = 'RequestTimeoutError'
    this.timeoutMs = timeoutMs
  }
}

/**
 * `fetch` with an abort budget. The timer is always cleared, so a fast response
 * never leaves a pending timeout behind.
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } catch (error) {
    // Only this controller can abort these requests, so an AbortError here is
    // always our own timeout rather than a caller cancelling.
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new RequestTimeoutError(timeoutMs)
    }
    throw error
  } finally {
    clearTimeout(timer)
  }
}

export class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    return body?.error?.message ?? body?.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

export async function register(email: string, password: string): Promise<User> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/auth/register`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    },
    DEFAULT_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function login(email: string, password: string): Promise<string> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/auth/login`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username: email, password }),
    },
    DEFAULT_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  const data = await response.json()
  return data.access_token
}

export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/auth/me`,
    { headers: { Authorization: `Bearer ${token}` } },
    DEFAULT_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function uploadImage(token: string, file: File): Promise<UploadRecord> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/uploads`,
    {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    },
    UPLOAD_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

/* ─────────────────────────────────────────────────────────────────────────────
   New below this line. Types mirror backend/app/ai/schemas.py and
   contracts/ai-api/*.schema.json exactly.
   ───────────────────────────────────────────────────────────────────────── */

export interface Prediction {
  label: string
  /** 0..1 inclusive. */
  confidence: number
}

export type AnalysisStatus = 'completed' | 'failed' | 'processing'
export type ConfidenceStatus = 'high' | 'moderate' | 'low'

export interface AnalyzeRequest {
  analysis_id: string
  /** Must be an absolute URL the AI service can fetch (pydantic HttpUrl). */
  image_url: string
  model_version?: string
}

export interface AnalyzeResponse {
  analysis_id: string
  status: AnalysisStatus
  model_version: string
  predictions: Prediction[]
  confidence_status: ConfidenceStatus
  explanation: string
  recommendation: string
  disclaimer: string
}

export interface ChatRequest {
  session_id: string
  message: string
  analysis_id?: string
}

export interface ChatResponse {
  response: string
  disclaimer: string
}

/**
 * POST /api/v1/analyze (backend/app/main.py::analyze_image_endpoint).
 *
 * Note: this endpoint does NOT declare `Depends(get_current_user)` today, so no
 * Authorization header is required. `token` is accepted and sent when present so
 * that adding auth on the backend later needs no frontend change.
 */
export async function analyzeImage(
  req: AnalyzeRequest,
  token?: string | null,
): Promise<AnalyzeResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/analyze`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(req),
    },
    AI_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

/** POST /api/v1/chat (backend/app/main.py::chat_endpoint). Also unauthenticated today. */
export async function sendChatMessage(
  req: ChatRequest,
  token?: string | null,
): Promise<ChatResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE_URL}/api/v1/chat`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(req),
    },
    AI_TIMEOUT_MS,
  )
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

/** GET /health — used by the connection banner. */
export async function getHealth(): Promise<{ status: string; version: string }> {
  const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, DEFAULT_TIMEOUT_MS)
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

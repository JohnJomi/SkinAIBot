/**
 * API client.
 *
 * The first four functions (register, login, getCurrentUser, uploadImage) are
 * the repository's existing implementation, unchanged apart from importing
 * API_BASE_URL from lib/config.ts. Everything below the marked section is new
 * and maps 1:1 onto endpoints that already exist in backend/app/main.py — no
 * endpoint, field or status code is invented here.
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
   * Absolute URL the backend serves this image from, suitable as `image_url`
   * on POST /api/v1/analyze. Optional so an older backend that omits it is a
   * handled error rather than an undefined slipping into the request.
   */
  image_url?: string
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
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function login(email: string, password: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username: email, password }),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  const data = await response.json()
  return data.access_token
}

export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function uploadImage(token: string, file: File): Promise<UploadRecord> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}/api/v1/uploads`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  })
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
  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(req),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

/** POST /api/v1/chat (backend/app/main.py::chat_endpoint). Also unauthenticated today. */
export async function sendChatMessage(
  req: ChatRequest,
  token?: string | null,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(req),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

/** GET /health — used by the connection banner. */
export async function getHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

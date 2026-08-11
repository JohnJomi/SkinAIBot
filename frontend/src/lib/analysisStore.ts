/**
 * Client-side analysis index.
 *
 * BLOCKER: there is no analysis persistence in the backend — no `Analysis`
 * model, no `GET /api/v1/analyses`, no `GET /api/v1/analyses/{id}`. The Results
 * and Dashboard screens need a list, so this module keeps one in localStorage
 * and hides it behind `AnalysisStore` so it can be swapped for real endpoints by
 * replacing the implementation only (no screen changes).
 *
 * Consequences to be aware of while this stand-in is in place: the list is
 * per-browser, lost when storage is cleared, and not shared between devices.
 */

import type { AnalyzeResponse, UploadRecord } from './api'
import { ANALYSIS_STORE_KEY_PREFIX } from './config'

export interface AnalysisRecord {
  analysis_id: string
  created_at: string
  body_site: string | null
  upload: Pick<UploadRecord, 'id' | 'original_filename' | 'content_type' | 'size_bytes'>
  /** Absolute URL the analysis was run against (server-reachable). */
  analysis_image_url: string | null
  /** Absolute URL the browser can load for display, when one was supplied. */
  display_image_url: string | null
  /** null while processing, or when the analysis failed before returning. */
  result: AnalyzeResponse | null
  /** Set when the request itself failed (network, 5xx, misconfiguration). */
  error: string | null
}

export interface AnalysisStore {
  list(): AnalysisRecord[]
  get(analysisId: string): AnalysisRecord | null
  save(record: AnalysisRecord): void
  remove(analysisId: string): void
  clear(): void
}

/** localStorage key holding one user's index. */
export function analysisStoreKey(userId: string): string {
  return `${ANALYSIS_STORE_KEY_PREFIX}:${userId}`
}

function read(key: string): AnalysisRecord[] {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as AnalysisRecord[]) : []
  } catch {
    return []
  }
}

function write(key: string, records: AnalysisRecord[]): void {
  try {
    localStorage.setItem(key, JSON.stringify(records))
  } catch {
    /* Quota or private-mode failure: the index is a convenience, not a source of truth. */
  }
}

/**
 * An index for one authenticated user.
 *
 * Keyed by the account's stable id, not its email: an email can be changed, and
 * a renamed account must keep its own history rather than inherit someone
 * else's. Two accounts on one browser therefore never share a key, and logging
 * out leaves nothing readable by the next account.
 */
function persistentStore(userId: string): AnalysisStore {
  const key = analysisStoreKey(userId)
  return {
    list() {
      return read(key).sort((a, b) => b.created_at.localeCompare(a.created_at))
    },
    get(analysisId) {
      return read(key).find((r) => r.analysis_id === analysisId) ?? null
    },
    save(record) {
      const records = read(key).filter((r) => r.analysis_id !== record.analysis_id)
      records.push(record)
      write(key, records)
    },
    remove(analysisId) {
      write(key, read(key).filter((r) => r.analysis_id !== analysisId))
    },
    clear() {
      write(key, [])
    },
  }
}

/**
 * A store for when no user is established yet (auth still loading, or demo
 * mode). Held in memory only: an unattributed record must never be written to
 * a key another account could later read.
 */
function memoryStore(): AnalysisStore {
  let records: AnalysisRecord[] = []
  return {
    list() {
      return [...records].sort((a, b) => b.created_at.localeCompare(a.created_at))
    },
    get(analysisId) {
      return records.find((r) => r.analysis_id === analysisId) ?? null
    },
    save(record) {
      records = [...records.filter((r) => r.analysis_id !== record.analysis_id), record]
    },
    remove(analysisId) {
      records = records.filter((r) => r.analysis_id !== analysisId)
    },
    clear() {
      records = []
    },
  }
}

/** The only supported way to obtain a store. */
export function createAnalysisStore(userId: string | null | undefined): AnalysisStore {
  return userId ? persistentStore(userId) : memoryStore()
}

/** Stable id for a new analysis. `crypto.randomUUID` needs a secure context. */
export function newAnalysisId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `a-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

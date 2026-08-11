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
import { ANALYSIS_STORE_KEY } from './config'

export interface AnalysisRecord {
  analysis_id: string
  created_at: string
  body_site: string | null
  upload: Pick<UploadRecord, 'id' | 'original_filename' | 'content_type' | 'size_bytes'>
  /** Absolute URL the analysis was run against, when one could be resolved. */
  image_url: string | null
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

function read(): AnalysisRecord[] {
  try {
    const raw = localStorage.getItem(ANALYSIS_STORE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? (parsed as AnalysisRecord[]) : []
  } catch {
    return []
  }
}

function write(records: AnalysisRecord[]): void {
  try {
    localStorage.setItem(ANALYSIS_STORE_KEY, JSON.stringify(records))
  } catch {
    /* Quota or private-mode failure: the index is a convenience, not a source of truth. */
  }
}

export const localAnalysisStore: AnalysisStore = {
  list() {
    return read().sort((a, b) => b.created_at.localeCompare(a.created_at))
  },
  get(analysisId) {
    return read().find((r) => r.analysis_id === analysisId) ?? null
  },
  save(record) {
    const records = read().filter((r) => r.analysis_id !== record.analysis_id)
    records.push(record)
    write(records)
  },
  remove(analysisId) {
    write(read().filter((r) => r.analysis_id !== analysisId))
  },
  clear() {
    write([])
  },
}

/** Stable id for a new analysis. `crypto.randomUUID` needs a secure context. */
export function newAnalysisId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `a-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

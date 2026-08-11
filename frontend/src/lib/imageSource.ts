/**
 * Turning an upload into an `image_url` for POST /api/v1/analyze.
 *
 * ── The problem ──────────────────────────────────────────────────────────────
 * `POST /api/v1/analyze` requires `image_url`, a pydantic `HttpUrl` that the AI
 * service fetches server-side. The backend today stores uploads via
 * `LocalFileStorage` under `UPLOAD_DIR` and exposes **no** route that serves or
 * signs them: `POST /api/v1/uploads` returns metadata only, and there is no
 * `GET /api/v1/uploads/{id}`, no static mount, and no presigned-URL endpoint
 * (Sprint 4 in docs/ROADMAP.md).
 *
 * A browser `blob:`/`data:` URL cannot be used — it is not reachable from the
 * AI container and is not a valid `HttpUrl`.
 *
 * ── The seam ─────────────────────────────────────────────────────────────────
 * This module is the single place that resolves an upload to a URL. Point
 * `VITE_UPLOAD_URL_TEMPLATE` at whatever the backend eventually exposes (any
 * `{id}` / `{filename}` placeholders are substituted) and nothing else in the
 * frontend changes. When it is unset, `resolveUploadImageUrl` throws
 * `MissingImageUrlError`, which UploadScreen renders as an explicit,
 * non-scary configuration error rather than a failed analysis.
 */

import type { UploadRecord } from './api'
import { UPLOAD_URL_TEMPLATE } from './config'

export class MissingImageUrlError extends Error {
  constructor() {
    super(
      'The backend does not yet expose a fetchable URL for uploaded images. ' +
        'Set VITE_UPLOAD_URL_TEMPLATE (e.g. http://localhost:8000/static/uploads/{id}) ' +
        'once an upload-serving endpoint exists.',
    )
    this.name = 'MissingImageUrlError'
  }
}

export function hasImageUrlSupport(): boolean {
  return Boolean(UPLOAD_URL_TEMPLATE)
}

export function resolveUploadImageUrl(upload: UploadRecord): string {
  if (!UPLOAD_URL_TEMPLATE) throw new MissingImageUrlError()
  return UPLOAD_URL_TEMPLATE.replace('{id}', encodeURIComponent(upload.id)).replace(
    '{filename}',
    encodeURIComponent(upload.original_filename),
  )
}

/**
 * Local preview URL for a freshly chosen file. Object URLs must be revoked by
 * the caller (see useObjectUrl in hooks/useObjectUrl.ts) or they leak.
 */
export function createPreviewUrl(file: File): string {
  return URL.createObjectURL(file)
}

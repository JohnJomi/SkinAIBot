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

/**
 * Whether an analysis can obtain an image URL at all.
 *
 * The backend now serves stored uploads (GET /api/v1/uploads/{stored_filename})
 * and returns an absolute `image_url` on the upload response, so the capability
 * exists without any client configuration. The template below remains as an
 * override for deployments that serve uploads from somewhere else.
 */
export function hasImageUrlSupport(): boolean {
  return true
}

/** Whether the optional override template is configured. */
export function hasUploadUrlTemplate(): boolean {
  return Boolean(UPLOAD_URL_TEMPLATE)
}

export function resolveUploadImageUrl(upload: UploadRecord): string {
  // Explicit override first, so a deployment serving images elsewhere keeps
  // control.
  if (UPLOAD_URL_TEMPLATE) {
    return UPLOAD_URL_TEMPLATE.replace('{id}', encodeURIComponent(upload.id)).replace(
      '{filename}',
      encodeURIComponent(upload.original_filename),
    )
  }
  // Otherwise use what the backend gave us. Still guarded: a backend that
  // predates the upload-serving route returns no image_url, and sending
  // `undefined` would fail deep inside the analyze call instead of here.
  if (upload.image_url) return upload.image_url
  throw new MissingImageUrlError()
}

/**
 * Local preview URL for a freshly chosen file. Object URLs must be revoked by
 * the caller (see useObjectUrl in hooks/useObjectUrl.ts) or they leak.
 */
export function createPreviewUrl(file: File): string {
  return URL.createObjectURL(file)
}

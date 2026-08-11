/**
 * Choosing the right URL for an uploaded image.
 *
 * The backend serves stored uploads from `GET /api/v1/uploads/{stored_filename}`
 * and returns two absolute URLs for the same file, because two different
 * clients fetch it and they do not resolve the same hostnames:
 *
 *   analysis_image_url  the AI service fetches this server-side. Inside Docker
 *                       it resolves to the backend's Compose service name, which
 *                       a browser cannot resolve at all.
 *   display_image_url   the browser loads this in an <img>. It uses the host the
 *                       browser can actually reach.
 *
 * Sending the wrong one is silent: the analysis fails, or the photograph renders
 * broken. This module is the single place that picks between them, so the two
 * resolvers below are the only supported way to obtain either.
 */

import type { UploadRecord } from './api'
import { UPLOAD_URL_TEMPLATE } from './config'

/**
 * The upload response carried no URL the AI service could fetch.
 *
 * Expected only against a backend older than the upload-serving route; a
 * current backend always supplies `analysis_image_url`.
 */
export class MissingImageUrlError extends Error {
  constructor() {
    super(
      'This upload did not come back with an address the analysis service can ' +
        'reach, so it cannot be analysed. The application backend may be out of date.',
    )
    this.name = 'MissingImageUrlError'
  }
}

/**
 * Whether an analysis can obtain an image URL at all.
 *
 * Always true: the backend serves stored uploads and returns
 * `analysis_image_url` on every upload response, so no client configuration is
 * needed. Kept as the single place to express that, rather than scattering the
 * assumption across screens.
 */
export function hasImageUrlSupport(): boolean {
  return true
}

/** Whether the optional override template is configured. */
export function hasUploadUrlTemplate(): boolean {
  return Boolean(UPLOAD_URL_TEMPLATE)
}

/**
 * URL for `POST /api/v1/analyze`, fetched server-side by the AI service.
 *
 * Never render this in an <img>: inside Docker it names a host only the server
 * network can resolve.
 */
export function resolveUploadImageUrl(upload: UploadRecord): string {
  // Explicit override first, so a deployment serving images elsewhere keeps
  // control. See VITE_UPLOAD_URL_TEMPLATE in lib/config.ts - it overrides this
  // server-side URL only, never the display URL below.
  if (UPLOAD_URL_TEMPLATE) {
    return UPLOAD_URL_TEMPLATE.replace('{id}', encodeURIComponent(upload.id)).replace(
      '{filename}',
      encodeURIComponent(upload.original_filename),
    )
  }
  // Otherwise use what the backend gave us. Still guarded: a backend that
  // predates the upload-serving route returns no analysis_image_url, and
  // sending `undefined` would fail deep inside the analyze call instead of here.
  if (upload.analysis_image_url) return upload.analysis_image_url
  throw new MissingImageUrlError()
}

/**
 * URL for showing the stored image to the user.
 *
 * Deliberately not the analyze URL: that one resolves on the server network
 * and would render as a broken image. Null when the backend did not supply
 * one, which the caller shows as a placeholder rather than a broken <img>.
 */
export function resolveDisplayImageUrl(upload: UploadRecord): string | null {
  return upload.display_image_url ?? null
}

/**
 * Local preview URL for a freshly chosen file. Object URLs must be revoked by
 * the caller (see useObjectUrl in hooks/useObjectUrl.ts) or they leak.
 */
export function createPreviewUrl(file: File): string {
  return URL.createObjectURL(file)
}

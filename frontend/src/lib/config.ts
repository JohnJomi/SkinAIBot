/**
 * Runtime configuration. Every value comes from Vite env vars so nothing is
 * hard-coded per environment. No secrets belong in this file — everything here
 * ships to the browser.
 */

const env = import.meta.env

/** Application backend (FastAPI). Matches the existing default in lib/api.ts. */
export const API_BASE_URL: string = env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/**
 * Optional override for the SERVER-SIDE analysis URL only.
 *
 * Not required: the backend serves uploads itself and returns both
 * `analysis_image_url` and `display_image_url` on every upload response. This
 * exists for a deployment that serves stored images from somewhere else (a CDN,
 * an object store) and needs the AI service pointed there instead.
 *
 * Two things to know before setting it:
 *  - It replaces `analysis_image_url`, which the AI service fetches
 *    server-side. It does NOT affect `display_image_url`, so the value must be
 *    reachable from the server, not merely from a browser.
 *  - `{id}` substitutes the upload row id and `{filename}` the original
 *    filename. Neither matches the backend's own route, which is keyed on the
 *    stored filename, so this cannot be pointed at `/api/v1/uploads/...`.
 *
 * Leave unset unless both apply. See src/lib/imageSource.ts.
 */
export const UPLOAD_URL_TEMPLATE: string | null = env.VITE_UPLOAD_URL_TEMPLATE ?? null

/**
 * Demo mode. When true, the analysis and chat screens render from
 * src/lib/mockData.ts instead of calling the backend, so the UI can be reviewed
 * without a running stack. Mock data is never mixed into real responses.
 */
export const USE_MOCK_DATA: boolean = env.VITE_USE_MOCK_DATA === 'true'

/** localStorage key already used by the existing App.tsx. Kept as-is. */
export const TOKEN_STORAGE_KEY = 'skinaibot_token'

/**
 * Prefix for the client-side analysis index (see src/lib/analysisStore.ts).
 *
 * A prefix rather than a key: the index is stored per user id, because two
 * accounts used from the same browser must not see each other's history. The
 * old unscoped `skinaibot_analyses` key is deliberately never read again.
 */
export const ANALYSIS_STORE_KEY_PREFIX = 'skinaibot_analyses'

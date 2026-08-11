/**
 * Runtime configuration. Every value comes from Vite env vars so nothing is
 * hard-coded per environment. No secrets belong in this file — everything here
 * ships to the browser.
 */

const env = import.meta.env

/** Application backend (FastAPI). Matches the existing default in lib/api.ts. */
export const API_BASE_URL: string = env.VITE_API_BASE_URL ?? 'http://localhost:8000'

/**
 * Template used to turn an upload id into an absolute, publicly fetchable image
 * URL for `POST /api/v1/analyze` (whose `image_url` field is a required
 * `HttpUrl`).
 *
 * BLOCKER: the backend currently has no endpoint that serves or signs an
 * uploaded file — `POST /api/v1/uploads` returns only
 * `{id, original_filename, content_type, size_bytes, created_at}`. Until such an
 * endpoint exists this must be supplied manually, e.g.
 *   VITE_UPLOAD_URL_TEMPLATE=http://localhost:8000/static/uploads/{id}
 * See src/lib/imageSource.ts and ASSUMPTIONS_AND_BLOCKERS.md.
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

/** Client-side analysis index key (see src/lib/analysisStore.ts). */
export const ANALYSIS_STORE_KEY = 'skinaibot_analyses'

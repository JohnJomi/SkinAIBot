/**
 * Image display. One component for every photograph in the app so that swapping
 * placeholders for real assets — or for a served upload URL — is a single change
 * here.
 *
 * `src` must be browser-loadable: a local object URL, an AnalysisRecord's
 * `display_image_url`, or null for the placeholder. Never `analysis_image_url` —
 * that address is resolvable only from the server network and renders broken.
 */

import { formatBytes } from '../lib/format'
import { PhotoPlaceholder } from './StateViews'

export function ImagePreview({
  src,
  alt = 'The skin photograph you uploaded',
  filename,
  sizeBytes,
  contentType,
  heightClass = 'h-72',
  placeholderLabel,
}: {
  src?: string | null
  alt?: string
  filename?: string
  sizeBytes?: number
  contentType?: string
  heightClass?: string
  placeholderLabel?: string
}) {
  return (
    <figure>
      {src ? (
        <img
          src={src}
          alt={alt}
          className={`grayscale-photo w-full object-cover ${heightClass} bg-surface`}
        />
      ) : (
        <PhotoPlaceholder
          label={placeholderLabel ?? 'Lesion photograph'}
          className={`w-full ${heightClass}`}
        />
      )}
      {filename ? (
        <figcaption className="mt-1 text-[11px] text-muted">
          {filename}
          {contentType ? ` · ${contentType}` : ''}
          {typeof sizeBytes === 'number' ? ` · ${formatBytes(sizeBytes)}` : ''}
        </figcaption>
      ) : null}
    </figure>
  )
}

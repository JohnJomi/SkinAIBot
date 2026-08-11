/**
 * Image input. Click, keyboard, or drag-and-drop; validates against the same
 * rules the backend enforces (Settings.allowed_upload_content_types and
 * max_upload_size_bytes in backend/app/config/settings.py) so an obvious
 * rejection happens before a request is spent on it.
 */

import { useId, useRef, useState, type DragEvent } from 'react'
import { InlineError } from './StateViews'

export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type as (typeof ACCEPTED_TYPES)[number])) {
    return 'That file type is not supported. Use a JPEG, PNG or WebP photograph.'
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return 'That photograph is larger than 10 MB. Reduce the resolution and try again.'
  }
  return null
}

export function ImageDropzone({
  onFile,
  disabled = false,
}: {
  onFile: (file: File) => void
  disabled?: boolean
}) {
  const inputId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const accept = (file: File | undefined) => {
    if (!file) return
    const problem = validateImageFile(file)
    setError(problem)
    if (!problem) onFile(file)
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    accept(event.dataTransfer.files?.[0])
  }

  return (
    <div>
      <div
        onDragOver={(event) => {
          event.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`border-2 border-dashed p-8 text-center transition-colors ${
          dragging ? 'border-accent bg-accent-100' : 'border-rule bg-surface'
        } ${disabled ? 'opacity-45' : ''}`}
      >
        <p className="text-base font-extrabold">Drop a photograph here</p>
        <p className="mx-auto mt-2 max-w-sm text-xs text-muted">
          JPEG, PNG or WebP, up to 10 MB. Daylight, no flash, lens about 15 cm from the skin, with a
          coin or ruler in frame for scale.
        </p>
        <label
          htmlFor={inputId}
          className={`btn btn-primary mt-5 ${disabled ? 'pointer-events-none opacity-45' : ''}`}
        >
          Choose a photograph
        </label>
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          disabled={disabled}
          className="sr-only"
          onChange={(event) => {
            accept(event.target.files?.[0])
            event.target.value = ''
          }}
        />
      </div>
      {error ? <div className="mt-3">{<InlineError message={error} />}</div> : null}
    </div>
  )
}

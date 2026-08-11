import { useEffect, useState } from 'react'

/**
 * Object URL for a chosen File, revoked when the file changes or the component
 * unmounts. Keeps image handling modular: the preview never cares whether the
 * source is a local File or a served URL.
 */
export function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setUrl(null)
      return
    }
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])

  return url
}

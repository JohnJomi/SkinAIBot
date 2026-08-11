/**
 * Upload → analyze orchestration, and the four states the UI needs from it.
 *
 * Two backend calls, in order:
 *   1. POST /api/v1/uploads   (multipart, Bearer)         → UploadRecord
 *   2. POST /api/v1/analyze   ({analysis_id, image_url})  → AnalyzeResponse
 *
 * Between them the upload must be turned into an absolute URL — see
 * lib/imageSource.ts for why that is currently a configuration step.
 */

import { useCallback, useState } from 'react'
import { analyzeImage, ApiError, uploadImage, type AnalyzeResponse } from '../lib/api'
import { localAnalysisStore, newAnalysisId, type AnalysisRecord } from '../lib/analysisStore'
import { MissingImageUrlError, resolveUploadImageUrl } from '../lib/imageSource'
import { USE_MOCK_DATA } from '../lib/config'
import { mockAnalysis } from '../lib/mockData'

export type RunPhase = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error'

export interface RunState {
  phase: RunPhase
  record: AnalysisRecord | null
  result: AnalyzeResponse | null
  error: string | null
  /** True when the failure is a missing VITE_UPLOAD_URL_TEMPLATE, not a real error. */
  isConfigError: boolean
  run: (file: File, bodySite: string) => Promise<string | null>
  reset: () => void
}

export function useAnalysisRun(token: string | null): RunState {
  const [phase, setPhase] = useState<RunPhase>('idle')
  const [record, setRecord] = useState<AnalysisRecord | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isConfigError, setIsConfigError] = useState(false)

  const reset = useCallback(() => {
    setPhase('idle')
    setRecord(null)
    setResult(null)
    setError(null)
    setIsConfigError(false)
  }, [])

  const run = useCallback(
    async (file: File, bodySite: string): Promise<string | null> => {
      setError(null)
      setIsConfigError(false)
      setResult(null)

      const analysisId = newAnalysisId()

      if (USE_MOCK_DATA) {
        setPhase('analyzing')
        await new Promise((resolve) => setTimeout(resolve, 900))
        const demo: AnalyzeResponse = { ...mockAnalysis, analysis_id: analysisId }
        const demoRecord: AnalysisRecord = {
          analysis_id: analysisId,
          created_at: new Date().toISOString(),
          body_site: bodySite || null,
          upload: {
            id: `u-${analysisId}`,
            original_filename: file.name,
            content_type: file.type,
            size_bytes: file.size,
          },
          image_url: null,
          result: demo,
          error: null,
        }
        localAnalysisStore.save(demoRecord)
        setRecord(demoRecord)
        setResult(demo)
        setPhase('done')
        return analysisId
      }

      if (!token) {
        setPhase('error')
        setError('You need to be logged in to upload a photograph.')
        return null
      }

      try {
        setPhase('uploading')
        const upload = await uploadImage(token, file)

        setPhase('analyzing')
        const imageUrl = resolveUploadImageUrl(upload)
        const analysis = await analyzeImage(
          { analysis_id: analysisId, image_url: imageUrl },
          token,
        )

        const saved: AnalysisRecord = {
          analysis_id: analysisId,
          created_at: upload.created_at,
          body_site: bodySite || null,
          upload: {
            id: upload.id,
            original_filename: upload.original_filename,
            content_type: upload.content_type,
            size_bytes: upload.size_bytes,
          },
          image_url: imageUrl,
          result: analysis.status === 'completed' ? analysis : null,
          error: analysis.status === 'failed' ? 'The model could not analyse this photograph.' : null,
        }
        localAnalysisStore.save(saved)
        setRecord(saved)
        setResult(analysis)
        setPhase('done')
        return analysisId
      } catch (err) {
        setPhase('error')
        if (err instanceof MissingImageUrlError) {
          setIsConfigError(true)
          setError(err.message)
        } else if (err instanceof ApiError) {
          setError(err.message)
        } else {
          setError(err instanceof Error ? err.message : 'The analysis could not be completed.')
        }
        return null
      }
    },
    [token],
  )

  return { phase, record, result, error, isConfigError, run, reset }
}

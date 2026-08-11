/**
 * Upload & analyze. Four states in one screen: empty (dropzone), file chosen
 * (preview + site + consent), analyzing (progress), error (real error or the
 * documented configuration gap).
 */

import { useState } from 'react'
import { RailLayout } from '../components/AppShell'
import { ImageDropzone } from '../components/ImageDropzone'
import { ImagePreview } from '../components/ImagePreview'
import { ErrorState, LoadingState, PhotoPlaceholder } from '../components/StateViews'
import { Button, Field, Rule, SectionLabel } from '../components/ui'
import { useAnalysisRun } from '../hooks/useAnalysisRun'
import type { AnalysisStore } from '../lib/analysisStore'
import { useObjectUrl } from '../hooks/useObjectUrl'
import type { Route } from '../hooks/useHashRoute'
import { hasImageUrlSupport } from '../lib/imageSource'
import { USE_MOCK_DATA } from '../lib/config'

const SHOOTING_RULES = [
  'Daylight, no flash. Flash flattens the texture the model reads.',
  'Fill the frame — lens about 15 cm from the skin, in focus.',
  'Include a coin or ruler so size can be judged.',
  'Clean skin: no makeup, cream or marker on the spot.',
]

export function UploadScreen({
  token,
  store,
  navigate,
  onAnalysisSaved,
}: {
  token: string | null
  store: AnalysisStore
  navigate: (route: Route) => void
  onAnalysisSaved: () => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [bodySite, setBodySite] = useState('')
  const [consented, setConsented] = useState(false)
  const previewUrl = useObjectUrl(file)
  const run = useAnalysisRun(token, store)

  const busy = run.phase === 'uploading' || run.phase === 'analyzing'
  const configMissing = !USE_MOCK_DATA && !hasImageUrlSupport()

  const submit = async () => {
    if (!file) return
    const analysisId = await run.run(file, bodySite)
    if (analysisId) {
      onAnalysisSaved()
      navigate({ name: 'result', analysisId })
    }
  }

  return (
    <RailLayout
      rail={
        <div>
          <SectionLabel>Before you shoot</SectionLabel>
          <ol className="m-0 list-decimal pl-5 text-[13px] leading-relaxed">
            {SHOOTING_RULES.map((rule) => (
              <li key={rule} className="mb-2.5 last:mb-0">
                {rule}
              </li>
            ))}
          </ol>
          <Rule className="my-5" />
          <PhotoPlaceholder label="Good vs poor framing — example pair" className="h-40 w-full" />
          <p className="mt-2 text-[11px] text-muted">
            Replace with real reference photographs when they are available.
          </p>
        </div>
      }
    >
      <h1 className="text-3xl">New analysis</h1>
      <p className="mt-2 text-sm text-muted">
        The photograph is uploaded to your account, then scored by the model. You will see possible
        conditions and a confidence figure — never a diagnosis.
      </p>
      <Rule className="my-6" />

      {configMissing ? (
        <div className="mb-6">
          <ErrorState
            variant="config"
            title="Analysis is not wired up in this environment"
            message="This environment cannot reach the analysis service for uploaded photographs. Uploading still works, but no result will be produced until it is configured."
          />
        </div>
      ) : null}

      {busy ? (
        <LoadingState
          title={run.phase === 'uploading' ? 'Uploading the photograph…' : 'Running the model…'}
          steps={
            run.phase === 'uploading'
              ? ['Validating file type and size', 'Sending to the application backend']
              : ['Uploaded and validated', 'Requesting an AI-assisted analysis', 'Usually under 10 seconds']
          }
        />
      ) : run.phase === 'error' ? (
        <ErrorState
          title={run.isConfigError ? 'Configuration incomplete' : 'The analysis could not be completed'}
          message={run.error ?? 'Unknown error.'}
          variant={run.isConfigError ? 'config' : 'error'}
          onRetry={run.reset}
        />
      ) : !file ? (
        <ImageDropzone onFile={setFile} />
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="border-2 border-rule p-4">
            <ImagePreview
              src={previewUrl}
              heightClass="h-56"
              filename={file.name}
              sizeBytes={file.size}
              contentType={file.type}
            />
            <div className="mt-3 flex gap-2">
              <Button className="text-[13px]" onClick={() => setFile(null)}>
                Replace
              </Button>
              <Button
                variant="ghost"
                className="text-[13px]"
                onClick={() => {
                  setFile(null)
                  run.reset()
                }}
              >
                Remove
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-3.5">
            <Field
              id="body-site"
              label="Body site (optional)"
              placeholder="e.g. Left forearm"
              value={bodySite}
              onChange={(event) => setBodySite(event.target.value)}
              hint="Recorded locally so photographs of the same spot can be compared."
            />
            <label className="flex items-start gap-2.5 text-[13px] leading-snug">
              <input
                type="checkbox"
                checked={consented}
                onChange={(event) => setConsented(event.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-[var(--color-accent)]"
              />
              I understand this is an AI-assisted analysis, not a medical diagnosis, and that I should
              consult a qualified dermatologist about anything that concerns me.
            </label>
            <Button
              variant="primary"
              className="justify-start"
              disabled={!consented || (configMissing && !USE_MOCK_DATA)}
              onClick={submit}
            >
              Analyse photograph
            </Button>
          </div>
        </div>
      )}
    </RailLayout>
  )
}

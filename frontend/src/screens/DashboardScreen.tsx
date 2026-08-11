import { Page, PageHeading } from '../components/AppShell'
import { EmptyState, PhotoPlaceholder } from '../components/StateViews'
import { Button, ConfidenceTag, Rule, SectionLabel } from '../components/ui'
import { TopPredictionLine } from '../components/PredictionList'
import type { AnalysisRecord } from '../lib/analysisStore'
import { formatDate } from '../lib/format'
import type { Route } from '../hooks/useHashRoute'
import { routeToHash } from '../hooks/useHashRoute'

function Stat({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="border-rule px-0 py-5 sm:px-6 sm:first:pl-0 sm:last:pr-0 sm:[&:not(:last-child)]:border-r-2">
      <SectionLabel>{label}</SectionLabel>
      <div className="text-3xl font-extrabold">{value}</div>
      <p className="mt-2 text-[13px] text-muted">{note}</p>
    </div>
  )
}

export function DashboardScreen({
  records,
  navigate,
}: {
  records: AnalysisRecord[]
  navigate: (route: Route) => void
}) {
  const latest = records[0] ?? null
  const withResult = records.filter((record) => record.result)

  return (
    <Page>
      <PageHeading
        title="Your skin record"
        lede={
          records.length === 0
            ? 'No analyses yet. Upload a photograph to get an AI-assisted read.'
            : `${records.length} ${records.length === 1 ? 'analysis' : 'analyses'} on file.`
        }
        action={
          <Button variant="primary" onClick={() => navigate({ name: 'upload' })}>
            Start a new analysis
          </Button>
        }
      />
      <Rule className="mt-6" />

      {records.length === 0 ? (
        <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
          <EmptyState
            title="Nothing analysed yet"
            body="Photograph the spot in daylight without flash, about 15 cm away, with a coin or ruler in frame for scale. The first analysis takes under a minute."
            action={
              <Button variant="primary" onClick={() => navigate({ name: 'upload' })}>
                Upload a photograph
              </Button>
            }
          />
          <div className="border-2 border-rule p-4">
            <SectionLabel>Reference framing</SectionLabel>
            <PhotoPlaceholder label="Example photograph" className="h-40 w-full" />
          </div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 border-b-2 border-rule sm:grid-cols-3">
            <Stat
              label="Analyses on file"
              value={String(records.length)}
              note={`Most recent ${latest ? formatDate(latest.created_at) : '—'}.`}
            />
            <Stat
              label="Latest read"
              value={latest?.result ? `${latest.result.confidence_status} confidence` : 'No read'}
              note={
                latest?.result
                  ? 'Model prediction only — a clinician should confirm.'
                  : latest?.error ?? 'The last analysis did not complete.'
              }
            />
            <Stat
              label="Usable reads"
              value={`${withResult.length} of ${records.length}`}
              note="A failed read usually means the photograph was blurred or too far away."
            />
          </div>

          <div className="mt-7">
            <SectionLabel>Recent analyses</SectionLabel>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Site</th>
                  <th>Most likely (model prediction)</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {records.slice(0, 5).map((record) => (
                  <tr key={record.analysis_id}>
                    <td>{formatDate(record.created_at)}</td>
                    <td>{record.body_site ?? '—'}</td>
                    <td>
                      <a
                        href={routeToHash({ name: 'result', analysisId: record.analysis_id })}
                        className="text-ink no-underline hover:text-accent"
                      >
                        {record.result ? (
                          <TopPredictionLine predictions={record.result.predictions} />
                        ) : (
                          'Analysis failed'
                        )}
                      </a>
                    </td>
                    <td>
                      {record.result ? (
                        <ConfidenceTag status={record.result.confidence_status} />
                      ) : (
                        <span className="tag tag-outline">No read</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {records.length > 5 ? (
              <button
                type="button"
                className="btn btn-ghost mt-2.5"
                onClick={() => navigate({ name: 'results' })}
              >
                See all {records.length}
              </button>
            ) : null}
          </div>
        </>
      )}
    </Page>
  )
}

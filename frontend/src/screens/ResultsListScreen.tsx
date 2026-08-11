/** Results — the history list. A row opens the case sheet. */

import { useMemo, useState } from 'react'
import { Page, PageHeading } from '../components/AppShell'
import { EmptyState, PhotoPlaceholder } from '../components/StateViews'
import { Button, ConfidenceTag, Rule } from '../components/ui'
import { TopPredictionLine } from '../components/PredictionList'
import type { AnalysisRecord } from '../lib/analysisStore'
import { formatDate } from '../lib/format'
import type { Route } from '../hooks/useHashRoute'
import { routeToHash } from '../hooks/useHashRoute'

type Filter = 'all' | 'reads' | 'failed'

export function ResultsListScreen({
  records,
  navigate,
}: {
  records: AnalysisRecord[]
  navigate: (route: Route) => void
}) {
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return records
      .filter((record) =>
        filter === 'all' ? true : filter === 'reads' ? Boolean(record.result) : !record.result,
      )
      .filter((record) => {
        if (!needle) return true
        const haystack = [
          record.body_site ?? '',
          record.upload.original_filename,
          ...(record.result?.predictions.map((p) => p.label) ?? []),
        ]
          .join(' ')
          .toLowerCase()
        return haystack.includes(needle)
      })
  }, [filter, query, records])

  return (
    <Page>
      <PageHeading
        title="Results"
        lede="Every analysis run from this browser. Scores are comparable only between photographs of the same spot, taken the same way."
        action={
          <Button variant="primary" onClick={() => navigate({ name: 'upload' })}>
            New analysis
          </Button>
        }
      />
      <Rule className="my-6" />

      {records.length === 0 ? (
        <EmptyState
          title="No analyses yet"
          body="Upload a photograph and the result will be listed here."
          action={
            <Button variant="primary" onClick={() => navigate({ name: 'upload' })}>
              Upload a photograph
            </Button>
          }
        />
      ) : (
        <>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex border border-rule">
              {(
                [
                  ['all', `All ${records.length}`],
                  ['reads', 'With a read'],
                  ['failed', 'No read'],
                ] as [Filter, string][]
              ).map(([value, label], index) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={filter === value}
                  onClick={() => setFilter(value)}
                  className={`px-3 py-1.5 text-[13px] ${index > 0 ? 'border-l border-rule' : ''} ${
                    filter === value ? 'bg-accent text-ground' : 'text-ink hover:bg-neutral-200'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="w-full max-w-72">
              <label className="sr-only" htmlFor="results-filter">
                Filter results
              </label>
              <input
                id="results-filter"
                className="input"
                placeholder="Filter by site, file or condition"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
          </div>

          {visible.length === 0 ? (
            <EmptyState title="Nothing matches that filter" body="Clear the filter to see every analysis." />
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table min-w-[720px]">
                <thead>
                  <tr>
                    <th className="w-16" />
                    <th>Date</th>
                    <th>Site</th>
                    <th>Most likely (model prediction)</th>
                    <th>Confidence</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((record) => (
                    <tr key={record.analysis_id}>
                      <td>
                        <PhotoPlaceholder label="" className="h-10 w-10" />
                      </td>
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
                      <td className="text-muted">{record.result?.status ?? record.error ?? 'failed'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </Page>
  )
}

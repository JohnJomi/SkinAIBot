/**
 * Settings / account.
 *
 * Deliberately thin: the backend exposes only GET /api/v1/auth/me. There is no
 * password-change, email-change, account-deletion, upload-listing or
 * upload-deletion endpoint, so those controls are rendered disabled with the
 * reason stated rather than wired to invented routes. Local controls (the
 * client-side analysis index) are real and do work.
 */

import type { ReactNode } from 'react'
import { Page, PageHeading } from '../components/AppShell'
import { Button, Rule, SectionLabel } from '../components/ui'
import type { User } from '../lib/api'
import type { AnalysisRecord } from '../lib/analysisStore'
import { API_BASE_URL, USE_MOCK_DATA } from '../lib/config'
import { hasImageUrlSupport, hasUploadUrlTemplate } from '../lib/imageSource'

function Row({ term, value }: { term: string; value: string }) {
  return (
    <tr>
      <td className="text-muted">{term}</td>
      <td className="text-right">{value}</td>
    </tr>
  )
}

// Call sites pass JSX (<code> elements), not plain strings.
function Pending({ children }: { children: ReactNode }) {
  return (
    <p className="mt-2 text-[11px] leading-relaxed text-muted">
      Needs a backend endpoint that does not exist yet: {children}
    </p>
  )
}

export function SettingsScreen({
  user,
  records,
  onClearLocalIndex,
}: {
  user: User | null
  records: AnalysisRecord[]
  onClearLocalIndex: () => void
}) {
  const latestModel = records.find((record) => record.result)?.result?.model_version ?? null

  return (
    <Page>
      <PageHeading title="Settings" />
      <Rule className="my-6" />

      <div className="grid gap-10 lg:grid-cols-2">
        <div>
          <SectionLabel>Account</SectionLabel>
          <table className="data-table">
            <tbody>
              <Row term="Email" value={user?.email ?? '—'} />
              <Row term="Account id" value={user?.id ?? '—'} />
              <Row term="Status" value={user?.is_active ? 'Active' : 'Inactive'} />
            </tbody>
          </table>
          <div className="mt-4 flex flex-wrap gap-2.5">
            <Button disabled>Change password</Button>
            <Button disabled>Change email</Button>
          </div>
          <Pending>
            no <code>PATCH /api/v1/auth/me</code> or password-reset route (Sprint 2 in
            docs/ROADMAP.md).
          </Pending>

          <Rule className="my-6" />
          <SectionLabel>Your photographs</SectionLabel>
          <table className="data-table">
            <tbody>
              <Row term="Analyses indexed in this browser" value={String(records.length)} />
              <Row term="Used for model training" value="No" />
              <Row
                term="Served back to the app"
                value={hasImageUrlSupport() ? 'Yes' : 'Not available yet'}
              />
            </tbody>
          </table>
          <div className="mt-4 flex flex-wrap gap-2.5">
            <Button onClick={onClearLocalIndex}>Clear local analysis index</Button>
            <Button disabled>Delete uploads from the server</Button>
          </div>
          <Pending>
            no <code>GET</code> or <code>DELETE /api/v1/uploads</code> route — only{' '}
            <code>POST</code> exists.
          </Pending>
        </div>

        <div>
          <SectionLabel>Model</SectionLabel>
          <table className="data-table">
            <tbody>
              <Row term="Version reported by the last analysis" value={latestModel ?? '—'} />
              <Row term="Contract" value="contracts/ai-api/openapi.yaml" />
            </tbody>
          </table>
          <p className="mt-3.5 text-[11px] leading-relaxed text-muted">
            The model scores a photograph against the classes it was trained on. A low top score
            usually means the spot is not something it recognises, not that the spot is harmless.
          </p>

          <Rule className="my-6" />
          <SectionLabel>Environment</SectionLabel>
          <table className="data-table">
            <tbody>
              <Row term="API base URL" value={API_BASE_URL} />
              <Row term="Demo mode" value={USE_MOCK_DATA ? 'On' : 'Off'} />
              <Row
                term="Upload URL template"
                value={hasUploadUrlTemplate() ? 'Configured' : 'Unset (backend serves uploads)'}
              />
            </tbody>
          </table>

          <Rule className="my-6" />
          <Button disabled variant="ghost">
            Delete my account
          </Button>
          <Pending>
            no account-deletion route; GDPR erasure is unimplemented on the backend.
          </Pending>
        </div>
      </div>
    </Page>
  )
}

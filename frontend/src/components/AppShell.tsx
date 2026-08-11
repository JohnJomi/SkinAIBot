import type { ReactNode } from 'react'
import type { Route } from '../hooks/useHashRoute'
import { routeToHash } from '../hooks/useHashRoute'
import type { User } from '../lib/api'
import { USE_MOCK_DATA } from '../lib/config'

const TABS: { label: string; route: Route; match: Route['name'][] }[] = [
  { label: 'Dashboard', route: { name: 'dashboard' }, match: ['dashboard'] },
  { label: 'Upload', route: { name: 'upload' }, match: ['upload'] },
  { label: 'Results', route: { name: 'results' }, match: ['results', 'result'] },
  { label: 'Chat', route: { name: 'chat', analysisId: null }, match: ['chat'] },
]

/**
 * The persistent safety band. Present on every authenticated screen: the AI
 * contract requires a disclaimer with every analysis and every chat reply, and
 * the product is decision support, not diagnosis.
 */
export function DisclaimerBanner({ text }: { text?: string }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 border-b-2 border-rule bg-accent-100 px-4 py-2.5 sm:px-6">
      <span className="kicker shrink-0 text-accent-800">AI-assisted · not a diagnosis</span>
      <span className="text-xs leading-relaxed text-accent-900">
        {text ??
          'SkinAIBot is a clinical decision-support prototype. It suggests possible conditions and how confident the model is — it cannot diagnose. Please consult a qualified dermatologist about anything that concerns you.'}
      </span>
    </div>
  )
}

export function AppShell({
  user,
  route,
  navigate,
  onLogout,
  disclaimer,
  children,
}: {
  user: User | null
  route: Route
  navigate: (route: Route) => void
  onLogout: () => void
  disclaimer?: string
  children: ReactNode
}) {
  return (
    <div className="min-h-screen bg-ground">
      <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b-2 border-rule px-4 py-3 sm:px-6">
        <a
          href={routeToHash({ name: 'dashboard' })}
          className="mr-auto text-lg font-extrabold text-ink no-underline"
        >
          SkinAIBot
        </a>
        <nav aria-label="Main" className="flex flex-wrap items-center gap-4">
          {TABS.map((tab) => {
            const current = tab.match.includes(route.name)
            return (
              <a
                key={tab.label}
                href={routeToHash(tab.route)}
                aria-current={current ? 'page' : undefined}
                className={`text-sm no-underline ${current ? 'text-accent' : 'text-ink hover:text-accent'}`}
              >
                {tab.label}
              </a>
            )
          })}
        </nav>
        <div className="flex items-center gap-4">
          {user ? <span className="hidden text-xs text-muted sm:inline">{user.email}</span> : null}
          <a
            href={routeToHash({ name: 'settings' })}
            className="text-sm text-ink no-underline hover:text-accent"
          >
            Settings
          </a>
          <button
            type="button"
            onClick={() => {
              onLogout()
              navigate({ name: 'login' })
            }}
            className="btn btn-ghost text-sm"
          >
            Log out
          </button>
        </div>
      </header>
      <DisclaimerBanner text={disclaimer} />
      {USE_MOCK_DATA ? (
        <div className="border-b border-rule-soft bg-surface px-4 py-2 text-xs text-muted sm:px-6">
          Demo mode — screens are rendering fixture data from <code>src/lib/mockData.ts</code>. Unset{' '}
          <code>VITE_USE_MOCK_DATA</code> to talk to the backend.
        </div>
      ) : null}
      <main className="animate-enter">{children}</main>
    </div>
  )
}

/** Two-column page: a fixed fact rail and the content column. Stacks under lg. */
export function RailLayout({ rail, children }: { rail: ReactNode; children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)]">
      <aside className="border-b-2 border-rule p-6 lg:border-b-0 lg:border-r-2">{rail}</aside>
      <div className="min-w-0 p-6 sm:p-8">{children}</div>
    </div>
  )
}

export function Page({ children }: { children: ReactNode }) {
  return <div className="mx-auto max-w-[1180px] p-6 sm:p-8">{children}</div>
}

export function PageHeading({
  title,
  lede,
  action,
}: {
  title: string
  lede?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-3xl sm:text-4xl">{title}</h1>
        {lede ? <p className="mt-2 max-w-2xl text-sm">{lede}</p> : null}
      </div>
      {action}
    </div>
  )
}

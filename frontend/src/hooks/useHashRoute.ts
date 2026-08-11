/**
 * Minimal hash router — deliberately not a dependency.
 *
 * The repository has no routing library and no routes today (App.tsx switches on
 * a `activeTab` string). Hash routing keeps deep links working ("/#/result/A-4192")
 * with no server rewrite rules and no new package. Swap for react-router later by
 * replacing this hook and the `navigate` calls.
 */

import { useCallback, useEffect, useState } from 'react'

export type Route =
  | { name: 'login' }
  | { name: 'dashboard' }
  | { name: 'upload' }
  | { name: 'results' }
  | { name: 'result'; analysisId: string }
  | { name: 'chat'; analysisId: string | null }
  | { name: 'settings' }

export function routeToHash(route: Route): string {
  switch (route.name) {
    case 'result':
      return `#/result/${encodeURIComponent(route.analysisId)}`
    case 'chat':
      return route.analysisId ? `#/chat/${encodeURIComponent(route.analysisId)}` : '#/chat'
    default:
      return `#/${route.name}`
  }
}

function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean)
  const [head, tail] = parts
  switch (head) {
    case 'dashboard':
      return { name: 'dashboard' }
    case 'upload':
      return { name: 'upload' }
    case 'results':
      return { name: 'results' }
    case 'result':
      return tail ? { name: 'result', analysisId: decodeURIComponent(tail) } : { name: 'results' }
    case 'chat':
      return { name: 'chat', analysisId: tail ? decodeURIComponent(tail) : null }
    case 'settings':
      return { name: 'settings' }
    case 'login':
      return { name: 'login' }
    default:
      return { name: 'dashboard' }
  }
}

export function useHashRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash))

  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  const navigate = useCallback((next: Route) => {
    const hash = routeToHash(next)
    if (window.location.hash === hash) setRoute(next)
    else window.location.hash = hash
  }, [])

  return [route, navigate]
}

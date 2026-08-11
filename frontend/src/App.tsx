/**
 * Application root: auth gate, routing, and the analysis index the screens read.
 *
 * Replaces the scaffold App.tsx. The auth logic is the same as before, moved into
 * hooks/useAuth.ts; nothing about the backend contract changed.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { AppShell } from './components/AppShell'
import { LoginScreen } from './screens/LoginScreen'
import { DashboardScreen } from './screens/DashboardScreen'
import { UploadScreen } from './screens/UploadScreen'
import { ResultScreen } from './screens/ResultScreen'
import { ResultsListScreen } from './screens/ResultsListScreen'
import { ChatScreen } from './screens/ChatScreen'
import { SettingsScreen } from './screens/SettingsScreen'
import { useAuth } from './hooks/useAuth'
import { useHashRoute } from './hooks/useHashRoute'
import { localAnalysisStore, type AnalysisRecord } from './lib/analysisStore'
import { USE_MOCK_DATA } from './lib/config'
import { mockAnalysisHistory } from './lib/mockData'

export default function App() {
  const auth = useAuth()
  const [route, navigate] = useHashRoute()
  const [indexVersion, setIndexVersion] = useState(0)

  const records: AnalysisRecord[] = useMemo(
    () => (USE_MOCK_DATA ? mockAnalysisHistory : localAnalysisStore.list()),
    // indexVersion is the invalidation signal for the localStorage-backed list.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [indexVersion],
  )

  const refreshIndex = useCallback(() => setIndexVersion((version) => version + 1), [])

  // Send unauthenticated visitors to the login screen, and away from it once in.
  useEffect(() => {
    if (USE_MOCK_DATA) return
    if (!auth.token && route.name !== 'login') navigate({ name: 'login' })
    if (auth.token && route.name === 'login') navigate({ name: 'dashboard' })
  }, [auth.token, navigate, route.name])

  const recordFor = (analysisId: string | null): AnalysisRecord | null => {
    if (!analysisId) return null
    return records.find((record) => record.analysis_id === analysisId) ?? null
  }

  if (!USE_MOCK_DATA && !auth.token) {
    return <LoginScreen onLoggedIn={auth.logIn} />
  }

  return (
    <AppShell user={auth.user} route={route} navigate={navigate} onLogout={auth.logOut}>
      {route.name === 'dashboard' ? (
        <DashboardScreen records={records} navigate={navigate} />
      ) : route.name === 'upload' ? (
        <UploadScreen token={auth.token} navigate={navigate} onAnalysisSaved={refreshIndex} />
      ) : route.name === 'results' ? (
        <ResultsListScreen records={records} navigate={navigate} />
      ) : route.name === 'result' ? (
        <ResultScreen record={recordFor(route.analysisId)} token={auth.token} navigate={navigate} />
      ) : route.name === 'chat' ? (
        <ChatScreen
          record={recordFor(route.analysisId)}
          records={records}
          token={auth.token}
          navigate={navigate}
        />
      ) : route.name === 'settings' ? (
        <SettingsScreen
          user={auth.user}
          records={records}
          onClearLocalIndex={() => {
            localAnalysisStore.clear()
            refreshIndex()
          }}
        />
      ) : (
        <DashboardScreen records={records} navigate={navigate} />
      )}
    </AppShell>
  )
}

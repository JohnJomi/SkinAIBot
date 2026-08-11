/**
 * Auth state. This is the token/user effect from the repository's existing
 * App.tsx, moved into a hook unchanged in behaviour — including the `active`
 * guard that stops a stale 401 from logging a freshly re-authenticated user out.
 */

import { useCallback, useEffect, useState } from 'react'
import { getCurrentUser, type User } from '../lib/api'
import { TOKEN_STORAGE_KEY } from '../lib/config'

export interface AuthState {
  token: string | null
  user: User | null
  /** True while /auth/me is in flight for the current token. */
  loading: boolean
  logIn: (token: string) => void
  logOut: () => void
}

export function useAuth(): AuthState {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_STORAGE_KEY))
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!token) {
      setUser(null)
      return
    }
    let active = true
    setLoading(true)
    getCurrentUser(token)
      .then((currentUser) => {
        if (!active) return
        setUser(currentUser)
      })
      .catch(() => {
        if (!active) return
        localStorage.removeItem(TOKEN_STORAGE_KEY)
        setToken(null)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [token])

  const logIn = useCallback((newToken: string) => {
    localStorage.setItem(TOKEN_STORAGE_KEY, newToken)
    setToken(newToken)
  }, [])

  const logOut = useCallback(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    setToken(null)
    setUser(null)
  }, [])

  return { token, user, loading, logIn, logOut }
}

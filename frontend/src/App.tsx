import { useEffect, useState, type FormEvent } from 'react'
import { getCurrentUser, login as apiLogin, register as apiRegister, uploadImage, type UploadRecord, type User } from './lib/api'

const TOKEN_STORAGE_KEY = 'skinaibot_token'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_STORAGE_KEY))
  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {
    if (!token) {
      setUser(null)
      return
    }
    // The token can change while this request is in flight (logout then login,
    // or a fast re-login). Only the newest effect instance may touch auth state,
    // otherwise a stale 401 from the previous token would log the user back out.
    let active = true
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
    return () => {
      active = false
    }
  }, [token])

  const handleLoggedIn = async (newToken: string) => {
    localStorage.setItem(TOKEN_STORAGE_KEY, newToken)
    setToken(newToken)
    setActiveTab('upload')
  }

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_STORAGE_KEY)
    setToken(null)
    setUser(null)
    setActiveTab('login')
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      <nav className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-blue-600">SkinAIBot Platform</h1>
        <div className="space-x-4">
          <button type="button" aria-current={activeTab === 'dashboard' ? 'page' : undefined} onClick={() => setActiveTab('dashboard')} className={activeTab === 'dashboard' ? 'font-bold' : ''}>Dashboard</button>
          <button type="button" aria-current={activeTab === 'upload' ? 'page' : undefined} onClick={() => setActiveTab('upload')} className={activeTab === 'upload' ? 'font-bold' : ''}>Upload</button>
          <button type="button" aria-current={activeTab === 'chat' ? 'page' : undefined} onClick={() => setActiveTab('chat')} className={activeTab === 'chat' ? 'font-bold' : ''}>Chat</button>
          {token ? (
            <>
              {user && <span className="text-sm text-gray-500">{user.email}</span>}
              <button type="button" onClick={handleLogout}>Logout</button>
            </>
          ) : (
            <button type="button" aria-current={activeTab === 'login' ? 'page' : undefined} onClick={() => setActiveTab('login')} className={activeTab === 'login' ? 'font-bold' : ''}>Login</button>
          )}
        </div>
      </nav>
      <main className="p-8 max-w-4xl mx-auto">
        {activeTab === 'dashboard' && <div><h2 className="text-2xl font-semibold mb-4">Dashboard Placeholder</h2><p>Overview of recent analyses and system status.</p></div>}
        {activeTab === 'upload' && <UploadPanel token={token} />}
        {activeTab === 'chat' && <div><h2 className="text-2xl font-semibold mb-4">Chat Placeholder</h2><p>Interactive AI chat regarding diagnosis will be placed here.</p></div>}
        {activeTab === 'login' && <LoginPanel onLoggedIn={handleLoggedIn} />}
      </main>
    </div>
  )
}

function LoginPanel({ onLoggedIn }: { onLoggedIn: (token: string) => void | Promise<void> }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'register') {
        await apiRegister(email, password)
      }
      const token = await apiLogin(email, password)
      await onLoggedIn(token)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="max-w-sm mx-auto text-left">
      <h2 className="text-2xl font-semibold mb-4 text-center">{mode === 'login' ? 'Login' : 'Create account'}</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <label className="block text-sm mb-1" htmlFor="email">Email</label>
          <input id="email" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="w-full border rounded px-3 py-2" />
        </div>
        <div>
          <label className="block text-sm mb-1" htmlFor="password">Password</label>
          <input id="password" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} className="w-full border rounded px-3 py-2" />
        </div>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <button type="submit" disabled={submitting} className="w-full bg-blue-600 text-white rounded px-3 py-2 disabled:opacity-50">
          {submitting ? 'Please wait...' : mode === 'login' ? 'Login' : 'Register'}
        </button>
      </form>
      <button
        type="button"
        className="mt-3 text-sm text-blue-600 underline"
        onClick={() => setMode(mode === 'login' ? 'register' : 'login')}
      >
        {mode === 'login' ? "Don't have an account? Register" : 'Already have an account? Login'}
      </button>
    </div>
  )
}

function UploadPanel({ token }: { token: string | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<UploadRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  if (!token) {
    return (
      <div>
        <h2 className="text-2xl font-semibold mb-4">Image Upload</h2>
        <p>Please log in to upload an image.</p>
      </div>
    )
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!file) return
    setError(null)
    setResult(null)
    setSubmitting(true)
    try {
      setResult(await uploadImage(token, file))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div>
      <h2 className="text-2xl font-semibold mb-4">Image Upload</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={!file || submitting} className="block bg-blue-600 text-white rounded px-3 py-2 disabled:opacity-50">
          {submitting ? 'Uploading...' : 'Upload'}
        </button>
      </form>
      {error && <p className="text-red-600 text-sm mt-3">{error}</p>}
      {result && (
        <div className="mt-4 border rounded p-4">
          <p><strong>Uploaded:</strong> {result.original_filename}</p>
          <p><strong>Type:</strong> {result.content_type}</p>
          <p><strong>Size:</strong> {result.size_bytes} bytes</p>
          <p><strong>Uploaded at:</strong> {new Date(result.created_at).toLocaleString()}</p>
        </div>
      )}
    </div>
  )
}

export default App

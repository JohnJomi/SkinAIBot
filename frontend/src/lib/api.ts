const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface User {
  id: string
  email: string
  is_active: boolean
}

export interface UploadRecord {
  id: string
  original_filename: string
  content_type: string
  size_bytes: number
  created_at: string
}

class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    return body?.error?.message ?? body?.detail ?? response.statusText
  } catch {
    return response.statusText
  }
}

export async function register(email: string, password: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function login(email: string, password: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username: email, password }),
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  const data = await response.json()
  return data.access_token
}

export async function getCurrentUser(token: string): Promise<User> {
  const response = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

export async function uploadImage(token: string, file: File): Promise<UploadRecord> {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}/api/v1/uploads`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  })
  if (!response.ok) throw new ApiError(await parseErrorMessage(response), response.status)
  return response.json()
}

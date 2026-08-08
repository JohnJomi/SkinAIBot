import { useState } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      <nav className="bg-white shadow-sm p-4 flex justify-between items-center">
        <h1 className="text-xl font-bold text-blue-600">SkinAIBot Platform</h1>
        <div className="space-x-4">
          <button onClick={() => setActiveTab('dashboard')} className={activeTab === 'dashboard' ? 'font-bold' : ''}>Dashboard</button>
          <button onClick={() => setActiveTab('upload')} className={activeTab === 'upload' ? 'font-bold' : ''}>Upload</button>
          <button onClick={() => setActiveTab('chat')} className={activeTab === 'chat' ? 'font-bold' : ''}>Chat</button>
          <button onClick={() => setActiveTab('login')} className={activeTab === 'login' ? 'font-bold' : ''}>Login</button>
        </div>
      </nav>
      <main className="p-8 max-w-4xl mx-auto">
        {activeTab === 'dashboard' && <div><h2 className="text-2xl font-semibold mb-4">Dashboard Placeholder</h2><p>Overview of recent analyses and system status.</p></div>}
        {activeTab === 'upload' && <div><h2 className="text-2xl font-semibold mb-4">Image Upload Placeholder</h2><p>Upload UI and result viewing will go here.</p></div>}
        {activeTab === 'chat' && <div><h2 className="text-2xl font-semibold mb-4">Chat Placeholder</h2><p>Interactive AI chat regarding diagnosis will be placed here.</p></div>}
        {activeTab === 'login' && <div><h2 className="text-2xl font-semibold mb-4">Login Placeholder</h2><p>User authentication view.</p></div>}
      </main>
    </div>
  )
}

export default App

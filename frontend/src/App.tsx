import { useEffect, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { api } from './api';
import Chat from './pages/Chat';
import Dashboard from './pages/Dashboard';
import Documents from './pages/Documents';
import History from './pages/History';
import Insights from './pages/Insights';
import Verification from './pages/Verification';

const NAV = [
  ['/', 'Chat'],
  ['/documents', 'Documents'],
  ['/insights', 'Retrieval Insights'],
  ['/verification', 'Evidence Verification'],
  ['/dashboard', 'Research Dashboard'],
  ['/history', 'History'],
] as const;

export default function App() {
  const [health, setHealth] = useState<{ llm_configured: boolean; nli_models: Record<string, string> } | 'down' | null>(null);
  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth('down'));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 pb-12">
      <header className="flex flex-wrap items-center justify-between gap-2 py-4">
        <h1 className="text-lg font-bold">
          Adaptive RAG <span className="font-normal text-slate-500">· query-adaptive retrieval + evidence verification</span>
        </h1>
        <div className="text-xs text-slate-500">
          {health === 'down' && <span className="text-red-600">backend unreachable (is it running on :8000?)</span>}
          {health && health !== 'down' && (
            <>
              {health.llm_configured ? 'LLM ready' : <span className="text-amber-600">OPENAI_API_KEY not set</span>}
              {Object.entries(health.nli_models).map(([name, status]) => (
                <span key={name}> · NLI {status}</span>
              ))}
            </>
          )}
        </div>
      </header>
      <nav className="mb-4 flex flex-wrap gap-1 border-b border-slate-200 pb-2 text-sm dark:border-slate-800">
        {NAV.map(([to, label]) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `rounded-lg px-3 py-1.5 ${isActive ? 'bg-sky-600 text-white' : 'hover:bg-slate-200 dark:hover:bg-slate-800'}`
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/insights/:id" element={<Insights />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/verification/:id" element={<Verification />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}

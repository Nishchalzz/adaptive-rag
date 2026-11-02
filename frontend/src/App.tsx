import { useEffect, useState } from 'react';
import { NavLink, Route, Routes } from 'react-router-dom';
import { api } from './api';
import Chat from './pages/Chat';
import Documents from './pages/Documents';
import History from './pages/History';
import How from './pages/How';
import Insights from './pages/Insights';
import Verification from './pages/Verification';

const NAV = [
  ['/', 'Chat'],
  ['/documents', 'Documents'],
  ['/insights', 'Retrieval Insights'],
  ['/verification', 'Evidence Verification'],
  ['/how-it-works', 'How it works'],
  ['/history', 'History'],
] as const;

type Health = { llm_configured: boolean; nli_models: Record<string, string> } | 'down' | null;

function HealthPill({ health }: { health: Health }) {
  if (health === null) return null;
  if (health === 'down') {
    return <span className="rounded-full bg-red-50 px-3 py-1 text-xs text-red-800 ring-1 ring-inset ring-red-200">Backend unreachable (is it running on :8000?)</span>;
  }
  const nli = Object.values(health.nli_models);
  const loading = nli.some((s) => s === 'loading');
  return (
    <span className="flex items-center gap-2 text-xs text-slate-600">
      <span className={`h-2 w-2 rounded-full ${health.llm_configured ? 'bg-emerald-500' : 'bg-amber-500'}`} />
      {health.llm_configured ? 'LLM ready' : 'OPENAI_API_KEY not set'}
      {loading && <span className="text-amber-700">· verification model loading…</span>}
    </span>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health>(null);
  useEffect(() => {
    const load = () => api.health().then(setHealth).catch(() => setHealth('down'));
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <div className="text-lg font-semibold tracking-tight text-slate-900">Adaptive RAG</div>
            <div className="text-xs text-slate-500">Query-adaptive retrieval and evidence verification</div>
          </div>
          <HealthPill health={health} />
        </div>
        <nav className="mx-auto flex max-w-6xl flex-wrap gap-1 px-6 pb-3 text-sm" aria-label="Main">
          {NAV.map(([to, label]) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `rounded-lg px-3 py-1.5 font-medium ${isActive ? 'bg-indigo-50 text-indigo-700' : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'}`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/insights" element={<Insights />} />
          <Route path="/insights/:id" element={<Insights />} />
          <Route path="/verification" element={<Verification />} />
          <Route path="/verification/:id" element={<Verification />} />
          <Route path="/how-it-works" element={<How />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </div>
  );
}

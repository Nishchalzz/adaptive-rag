import { useEffect, useState, type ReactNode } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api';
import type { QueryLog, QueryResponse } from '../types';
import { Empty, ErrorBox } from './ui';

/** Loads a stored query trace (from the URL id, else the latest) and renders it with ``render``. */
export function TraceView({ base, render }: { base: 'insights' | 'verification'; render: (r: QueryResponse) => ReactNode }) {
  const { id } = useParams();
  const navigate = useNavigate();
  const [logs, setLogs] = useState<QueryLog[] | null>(null);
  const [trace, setTrace] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.queries(50).then(setLogs).catch((e: Error) => setError(e.message));
  }, []);

  const selected = id ?? logs?.[0]?.id;
  useEffect(() => {
    if (!selected) return;
    setTrace(null);
    api.queryById(selected).then(setTrace).catch((e: Error) => setError(e.message));
  }, [selected]);

  if (error) return <ErrorBox message={error} />;
  if (logs && logs.length === 0 && !id) return <Empty>No queries yet. Ask a question in Chat first.</Empty>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <label className="text-slate-500">Query</label>
        <select
          value={selected ?? ''}
          onChange={(e) => navigate(`/${base}/${e.target.value}`)}
          className="max-w-full rounded-lg border border-slate-300 bg-white p-1.5 dark:border-slate-700 dark:bg-slate-900"
        >
          {(logs ?? []).map((l) => (
            <option key={l.id} value={l.id}>
              [{l.pipeline}] {l.question.slice(0, 80)}
            </option>
          ))}
        </select>
      </div>
      {trace ? render(trace) : <p className="animate-pulse text-sm text-slate-500">loading…</p>}
    </div>
  );
}

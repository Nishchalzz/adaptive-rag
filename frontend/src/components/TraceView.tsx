import { useEffect, useState, type ReactNode } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api';
import type { QueryLog, QueryResponse } from '../types';
import { ReplayNotice } from './Replay';
import { Empty, ErrorBox, PageHeader, SkeletonCard } from './ui';

/** Loads a stored query trace (from the URL id, else the latest) and renders it with ``render``. */
export function TraceView({
  base,
  title,
  subtitle,
  render,
}: {
  base: 'insights' | 'verification';
  title: string;
  subtitle: string;
  render: (r: QueryResponse) => ReactNode;
}) {
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

  const header = <PageHeader title={title} subtitle={subtitle} />;
  if (error) return <>{header}<ErrorBox message={error} /></>;
  if (logs && logs.length === 0 && !id) {
    return (
      <>
        {header}
        <Empty>
          No questions have been asked yet. <Link className="font-medium text-indigo-700 hover:underline" to="/">Ask one in Chat</Link> and its trace will appear here.
        </Empty>
      </>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title={title}
        subtitle={subtitle}
        right={
          <label className="flex items-center gap-2 text-xs text-slate-600">
            Question
            <select
              value={selected ?? ''}
              onChange={(e) => navigate(`/${base}/${e.target.value}`)}
              className="max-w-xs rounded-lg border border-slate-300 bg-white p-1.5 text-sm"
            >
              {(logs ?? []).map((l) => (
                <option key={l.id} value={l.id}>
                  [{l.pipeline}] {l.question.slice(0, 70)}
                </option>
              ))}
            </select>
          </label>
        }
      />
      {trace ? <>{<ReplayNotice response={trace} />}{render(trace)}</> : <div className="space-y-4"><SkeletonCard lines={3} /><SkeletonCard lines={5} /></div>}
    </div>
  );
}

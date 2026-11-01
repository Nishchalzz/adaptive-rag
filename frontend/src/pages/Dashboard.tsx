import { useEffect, useMemo, useState } from 'react';
import { api } from '../api';
import { Bar, Card, Empty, ErrorBox, Stat, ms, usd } from '../components/ui';
import type { ExperimentListItem, QueryResponse } from '../types';

type Json = Record<string, unknown>;

/** Flattens nested objects into dotted numeric metrics; arrays and non-numbers are skipped. */
function flatten(obj: unknown, prefix = ''): Record<string, number> {
  const out: Record<string, number> = {};
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return out;
  for (const [k, v] of Object.entries(obj as Json)) {
    if (typeof v === 'number' && Number.isFinite(v)) out[prefix + k] = v;
    else if (v && typeof v === 'object' && !Array.isArray(v)) Object.assign(out, flatten(v, `${prefix}${k}.`));
  }
  return out;
}

const COLORS = ['bg-sky-500', 'bg-violet-500', 'bg-emerald-500', 'bg-amber-500', 'bg-rose-500', 'bg-teal-500'];

function figureNames(summary: Json): string[] {
  const figs = summary.figures;
  const items: unknown[] = Array.isArray(figs) ? figs : figs && typeof figs === 'object' ? Object.values(figs as Json) : [];
  return items
    .map((f) => (typeof f === 'string' ? f : f && typeof f === 'object' ? String((f as Json).name ?? (f as Json).file ?? (f as Json).path ?? '') : ''))
    .map((f) => f.split(/[\\/]/).pop() ?? '')
    .filter((f) => f.endsWith('.png'));
}

function SuiteResults({ name }: { name: string }) {
  const [summary, setSummary] = useState<Json | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metric, setMetric] = useState('');

  useEffect(() => {
    setSummary(null);
    setReport(null);
    setError(null);
    api.experiment(name).then(setSummary).catch((e: Error) => setError(e.message));
    api.report(name).then(setReport).catch(() => undefined);
  }, [name]);

  const variants = useMemo(() => {
    const raw = summary?.variants;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {} as Record<string, Record<string, number>>;
    return Object.fromEntries(Object.entries(raw as Json).map(([k, v]) => [k, flatten(v)]));
  }, [summary]);

  const names = Object.keys(variants);
  const metrics = useMemo(() => [...new Set(names.flatMap((n) => Object.keys(variants[n])))].sort(), [names, variants]);
  useEffect(() => {
    if (!metric || !metrics.includes(metric)) setMetric(metrics[0] ?? '');
  }, [metrics, metric]);

  if (error) return <ErrorBox message={error} />;
  if (!summary) return <p className="animate-pulse text-sm text-slate-500">loading…</p>;

  const values = names.map((n) => variants[n][metric]).filter((v): v is number => v !== undefined);
  const max = Math.max(...values.map(Math.abs), 1e-9);
  const dataset = flatten(summary.dataset);
  const figures = figureNames(summary);

  return (
    <div className="space-y-4">
      {Object.keys(dataset).length > 0 && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {Object.entries(dataset).map(([k, v]) => (
            <Stat key={k} label={k} value={v} />
          ))}
        </div>
      )}
      {names.length === 0 ? (
        <Empty>This summary has no per-variant numeric metrics to show.</Empty>
      ) : (
        <>
          <Card
            title="Compare variants"
            right={
              <select value={metric} onChange={(e) => setMetric(e.target.value)} className="max-w-[50%] rounded-lg border border-slate-300 bg-white p-1 text-xs dark:border-slate-700 dark:bg-slate-900">
                {metrics.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            }
          >
            <div className="space-y-2">
              {names.map((n, i) =>
                variants[n][metric] === undefined ? null : (
                  <div key={n} className="flex items-center gap-2 text-xs">
                    <span className="w-36 shrink-0 truncate" title={n}>
                      {n}
                    </span>
                    <div className="h-3 flex-1 rounded-full bg-slate-200 dark:bg-slate-800">
                      <div className={`h-full rounded-full ${COLORS[i % COLORS.length]}`} style={{ width: `${(Math.abs(variants[n][metric]) / max) * 100}%` }} />
                    </div>
                    <span className="w-20 text-right tabular-nums">{variants[n][metric].toPrecision(4)}</span>
                  </div>
                ),
              )}
            </div>
          </Card>
          <Card title="All metrics">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="text-slate-500">
                    <th className="py-1 pr-3">metric</th>
                    {names.map((n) => (
                      <th key={n} className="pr-3 text-right">
                        {n}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m) => (
                    <tr key={m} className="border-t border-slate-200 dark:border-slate-800">
                      <td className="py-1 pr-3 font-medium">{m}</td>
                      {names.map((n) => (
                        <td key={n} className="pr-3 text-right tabular-nums">
                          {variants[n][m] === undefined ? '–' : Number(variants[n][m].toPrecision(4))}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
      {figures.length > 0 && (
        <Card title="Figures">
          <div className="grid gap-4 md:grid-cols-2">
            {figures.map((f) => (
              <img key={f} src={`/api/experiments/${encodeURIComponent(name)}/figures/${encodeURIComponent(f)}`} alt={f} className="w-full rounded-lg bg-white" />
            ))}
          </div>
        </Card>
      )}
      {report && (
        <Card title="Report">
          <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap text-xs">{report}</pre>
        </Card>
      )}
    </div>
  );
}

function CompareLive() {
  const [question, setQuestion] = useState('');
  const [results, setResults] = useState<{ baseline?: QueryResponse; proposed?: QueryResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    if (question.trim().length < 3) return;
    setBusy(true);
    setError(null);
    setResults(null);
    try {
      const [baseline, proposed] = await Promise.all([
        api.query({ question, pipeline: 'baseline' }),
        api.query({ question, pipeline: 'proposed' }),
      ]);
      setResults({ baseline, proposed });
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  };

  const rows: [string, (r: QueryResponse) => string][] = [
    ['Retrieval depth k', (r) => `${r.retrieval.k_requested}${r.retrieval.k_final != null && r.retrieval.k_final !== r.retrieval.k_requested ? `→${r.retrieval.k_final}` : ''}`],
    ['Chunks given to LLM', (r) => String(r.retrieval.num_retrieved)],
    ['Complexity', (r) => (r.complexity ? `${r.complexity.label} (${r.complexity.score.toFixed(2)})` : '–')],
    ['Retrieval latency', (r) => ms(r.retrieval.latency_ms)],
    ['Total latency', (r) => ms(r.total_latency_ms)],
    ['Input tokens', (r) => String(r.usage.input_tokens)],
    ['Output tokens', (r) => String(r.usage.output_tokens)],
    ['Estimated cost', (r) => usd(r.usage.cost_usd)],
    ['Generation attempts', (r) => String(r.attempts?.length || 1)],
    ['Claim support score', (r) => (r.verification ? r.verification.support_score.toFixed(2) : 'not verified')],
    ['Citation accuracy', (r) => (r.verification?.citation_accuracy != null ? r.verification.citation_accuracy.toFixed(2) : '–')],
    ['Unsupported claims', (r) => (r.verification ? String(r.verification.num_unsupported) : '–')],
  ];

  return (
    <Card title="Compare live: baseline vs proposed on one question">
      <div className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run()}
          placeholder="Question to run through both pipelines"
          className="flex-1 rounded-lg border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-950"
        />
        <button onClick={run} disabled={busy} className="rounded-lg bg-sky-600 px-4 text-sm font-medium text-white disabled:opacity-50">
          {busy ? 'Running…' : 'Compare'}
        </button>
      </div>
      {error && (
        <div className="mt-3">
          <ErrorBox message={error} />
        </div>
      )}
      {results?.baseline && results.proposed && (
        <table className="mt-3 w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase text-slate-500">
              <th className="py-1" />
              <th>Baseline</th>
              <th>Proposed</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, f]) => (
              <tr key={label} className="border-t border-slate-200 dark:border-slate-800">
                <td className="py-1 text-slate-500">{label}</td>
                <td>{f(results.baseline!)}</td>
                <td>{f(results.proposed!)}</td>
              </tr>
            ))}
            <tr className="border-t border-slate-200 align-top dark:border-slate-800">
              <td className="py-1 text-slate-500">Answer</td>
              <td className="pr-3 text-xs">{results.baseline.answer}</td>
              <td className="text-xs">{results.proposed.answer}</td>
            </tr>
          </tbody>
        </table>
      )}
      <p className="mt-2 text-xs text-slate-500">A single query is an anecdote. Use the experiment results above for conclusions.</p>
    </Card>
  );
}

export default function Dashboard() {
  const [suites, setSuites] = useState<ExperimentListItem[] | null>(null);
  const [suite, setSuite] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .experiments()
      .then((s) => {
        setSuites(s);
        if (s.length) setSuite(s[s.length - 1].name);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="space-y-4">
      <Card
        title="Experiment results"
        right={
          suites && suites.length > 0 && (
            <select value={suite} onChange={(e) => setSuite(e.target.value)} className="rounded-lg border border-slate-300 bg-white p-1 text-xs dark:border-slate-700 dark:bg-slate-900">
              {suites.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                  {s.num_questions ? ` (${s.num_questions} questions)` : ''}
                </option>
              ))}
            </select>
          )
        }
      >
        {error && <ErrorBox message={error} />}
        {suites && suites.length === 0 && (
          <Empty>
            No experiment results yet. Run <code>python -m scripts.run_experiments --help</code> in <code>backend/</code>; results appear here when
            <code> experiments/results/&lt;suite&gt;/summary.json</code> exists.
          </Empty>
        )}
        {suite && <SuiteResults name={suite} />}
      </Card>
      <CompareLive />
    </div>
  );
}

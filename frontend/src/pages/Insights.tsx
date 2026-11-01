import { TraceView } from '../components/TraceView';
import { Badge, Bar, Card, Stat, ms, sourceLabel, usd } from '../components/ui';
import type { QueryResponse } from '../types';

const BAND = { simple: 'bg-emerald-500', moderate: 'bg-amber-500', complex: 'bg-red-500' } as const;

function Insights({ r }: { r: QueryResponse }) {
  const c = r.complexity;
  const rt = r.retrieval;
  const conf = rt.confidence;
  const timings = Object.entries(r.timings_ms).filter(([k]) => k !== 'total');
  const maxTiming = Math.max(...timings.map(([, v]) => v), 1);

  return (
    <div className="space-y-4">
      <Card title="Question">
        <p className="font-medium">{r.question}</p>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          <Badge tone={r.pipeline === 'proposed' ? 'violet' : 'neutral'}>{r.pipeline}</Badge>
          <Badge>{rt.strategy}</Badge>
          {Object.entries(r.variant).map(([k, v]) => (
            <Badge key={k} tone={v ? 'info' : 'neutral'}>
              {k}: {v ? 'on' : 'off'}
            </Badge>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Complexity" value={c ? c.label : 'n/a'} hint={c ? `score ${c.score.toFixed(3)}` : 'fixed-k baseline'} />
        <Stat label="Selected k" value={rt.k_requested} hint={rt.k_final != null && rt.k_final !== rt.k_requested ? `expanded to ${rt.k_final}` : undefined} />
        <Stat label="Chunks given to LLM" value={rt.num_retrieved} hint={`${rt.expansions} expansion(s)`} />
        <Stat label="Retrieval latency" value={ms(rt.latency_ms)} hint={`embed ${ms(rt.embedding_latency_ms)} · search ${ms(rt.search_latency_ms)}`} />
        <Stat label="Total latency" value={ms(r.total_latency_ms)} hint={`${r.usage.input_tokens + r.usage.output_tokens} tokens · ${usd(r.usage.cost_usd)}`} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Query complexity assessment">
          {c ? (
            <div className="space-y-3 text-sm">
              <div className="relative h-3 rounded-full bg-gradient-to-r from-emerald-500 via-amber-500 to-red-500">
                <div
                  className="absolute -top-1 h-5 w-1.5 rounded bg-slate-900 dark:bg-white"
                  style={{ left: `calc(${c.score * 100}% - 3px)` }}
                />
              </div>
              <div className="flex justify-between text-xs text-slate-500">
                <span>simple</span>
                <span>moderate</span>
                <span>complex</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge tone="violet">{c.reasoning_type}</Badge>
                <Badge>{c.requires_multiple_sources ? 'multiple sources' : 'single source'}</Badge>
                {c.llm_label && <Badge>LLM label: {c.llm_label}</Badge>}
                {c.fallback && <Badge tone="partial">lexical fallback (LLM analysis failed)</Badge>}
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold uppercase text-slate-500">Score components (0–1)</div>
                <div className="space-y-1">
                  {Object.entries(c.components).map(([k, v]) => (
                    <Bar key={k} label={k} value={v} color={BAND[c.label]} />
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold uppercase text-slate-500">Information needs</div>
                <ol className="list-decimal space-y-0.5 pl-5">
                  {c.information_needs.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ol>
              </div>
              <p className="text-slate-500">{c.rationale}</p>
              <p className="text-xs text-slate-500">
                Lexical cues:{' '}
                {Object.entries(c.lexical)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(', ')}
              </p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">This run used fixed-k retrieval, so no complexity analysis was performed.</p>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Retrieval confidence">
            {conf ? (
              <div className="space-y-1">
                <Bar label="confidence" value={conf.score} color={conf.sufficient ? 'bg-emerald-500' : 'bg-amber-500'} />
                <Bar label="best score (norm.)" value={conf.top1} />
                <Bar label="mean score (norm.)" value={conf.mean} />
                <Bar label="query coverage" value={conf.coverage} />
                <p className="pt-1 text-xs text-slate-500">
                  {conf.sufficient ? 'Sufficient' : 'Below threshold'}
                  {conf.uncovered_queries.length > 0 && ` · uncovered: ${conf.uncovered_queries.join(' | ')}`}
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">Not computed for this pipeline.</p>
            )}
          </Card>
          <Card title="Stage latency">
            <div className="space-y-1">
              {timings.map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 text-xs">
                  <span className="w-32 shrink-0 text-slate-500">{k}</span>
                  <div className="h-2 flex-1 rounded-full bg-slate-200 dark:bg-slate-800">
                    <div className="h-full rounded-full bg-sky-500" style={{ width: `${(v / maxTiming) * 100}%` }} />
                  </div>
                  <span className="w-16 text-right tabular-nums">{ms(v)}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Token usage and cost by stage">
            <table className="w-full text-xs">
              <tbody>
                {r.usage.stages.map((s, i) => (
                  <tr key={i} className="border-t border-slate-200 dark:border-slate-800">
                    <td className="py-1">{s.stage}</td>
                    <td className="text-slate-500">{s.model}</td>
                    <td className="text-right tabular-nums">
                      {s.input_tokens}/{s.output_tokens}
                      {s.estimated && '*'}
                    </td>
                    <td className="text-right tabular-nums">{usd(s.cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1 text-xs text-slate-500">* estimated (embeddings do not report usage)</p>
          </Card>
        </div>
      </div>

      <Card title={`Search queries (${rt.queries.length})`}>
        <ul className="list-disc space-y-0.5 pl-5 text-sm">
          {rt.queries.map((q, i) => (
            <li key={i}>
              <span className="text-slate-500">{i === 0 ? 'original' : `sub-query ${i}`}:</span> {q}
            </li>
          ))}
        </ul>
      </Card>

      <Card title={`Retrieved chunks (${rt.chunks.length})`}>
        <div className="space-y-2">
          {rt.chunks.map((ch) => (
            <details key={ch.chunk_id} className="rounded-lg border border-slate-200 p-2 text-sm dark:border-slate-800">
              <summary className="cursor-pointer">
                <strong>[{ch.citation_number}]</strong> {sourceLabel(ch.filename, ch.page)}{' '}
                <Badge tone="info">{ch.score.toFixed(3)}</Badge> <Badge>{ch.retrieved_by}</Badge>
              </summary>
              <p className="mt-2 whitespace-pre-wrap text-slate-700 dark:text-slate-300">{ch.text}</p>
            </details>
          ))}
        </div>
      </Card>
    </div>
  );
}

export default function InsightsPage() {
  return <TraceView base="insights" render={(r) => <Insights r={r} />} />;
}

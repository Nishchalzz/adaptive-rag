import { Link } from 'react-router-dom';
import type { RunState } from '../useRun';
import type { PipelineName } from '../types';
import { AnswerView } from './AnswerView';
import { Badge, Card, ErrorBox, ms, usd } from './ui';

const NODE_LABELS: Record<string, string> = {
  analyze_query: 'Analyse query complexity',
  select_depth: 'Select retrieval depth',
  retrieve: 'Retrieve evidence',
  assess_confidence: 'Assess retrieval confidence',
  expand_retrieval: 'Expand retrieval',
  generate: 'Generate cited answer',
  extract_claims: 'Extract claims',
  verify: 'Verify claims',
  refine: 'Refine (targeted retrieval)',
  finalize: 'Finalise answer',
};

function summarise(summary: Record<string, unknown>): string {
  return Object.entries(summary)
    .filter(([, v]) => v !== null && v !== undefined && !Array.isArray(v))
    .map(([k, v]) => `${k}=${typeof v === 'number' ? Number(v.toFixed(3)) : String(v)}`)
    .join('  ');
}

export function RunPanel({ title, pipeline, state }: { title: string; pipeline: PipelineName; state: RunState }) {
  const { response, steps, error, running } = state;
  return (
    <Card
      title={title}
      right={
        response && (
          <span className="flex gap-3 text-xs">
            <Link className="text-sky-700 hover:underline dark:text-sky-300" to={`/insights/${response.query_id}`}>
              retrieval insights
            </Link>
            {response.verification && (
              <Link className="text-sky-700 hover:underline dark:text-sky-300" to={`/verification/${response.query_id}`}>
                verification
              </Link>
            )}
          </span>
        )
      }
    >
      {pipeline === 'proposed' && steps.length > 0 && (
        <ol className="mb-3 space-y-1 border-l-2 border-slate-200 pl-3 text-xs dark:border-slate-700">
          {steps.map((s, i) => (
            <li key={i}>
              <span className="font-medium">{NODE_LABELS[s.node] ?? s.node}</span>{' '}
              <span className="text-slate-500">{summarise(s.summary)}</span>
            </li>
          ))}
          {running && <li className="animate-pulse text-slate-500">working…</li>}
        </ol>
      )}
      {running && !response && pipeline === 'baseline' && <p className="animate-pulse text-sm text-slate-500">working…</p>}
      {error && <ErrorBox message={error} />}
      {response && (
        <>
          <div className="mb-3 flex flex-wrap gap-2 text-xs">
            {response.complexity && (
              <Badge tone="violet">
                {response.complexity.label} · score {response.complexity.score.toFixed(2)}
              </Badge>
            )}
            <Badge tone="info">
              k={response.retrieval.k_requested}
              {response.retrieval.k_final != null && response.retrieval.k_final !== response.retrieval.k_requested
                ? `→${response.retrieval.k_final}`
                : ''}{' '}
              · {response.retrieval.num_retrieved} chunks
            </Badge>
            <Badge>{ms(response.total_latency_ms)}</Badge>
            <Badge>
              {response.usage.input_tokens + response.usage.output_tokens} tokens · {usd(response.usage.cost_usd)}
            </Badge>
            {response.verification && (
              <Badge tone={response.verification.passed ? 'supported' : 'partial'}>
                support {response.verification.support_score.toFixed(2)}
              </Badge>
            )}
            {response.attempts.length > 1 && <Badge tone="partial">{response.attempts.length} attempts</Badge>}
          </div>
          <AnswerView response={response} />
        </>
      )}
    </Card>
  );
}

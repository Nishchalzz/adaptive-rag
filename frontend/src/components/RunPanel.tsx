import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import type { RunState } from '../useRun';
import type { PipelineName, QueryResponse } from '../types';
import { AnswerView } from './AnswerView';
import { ReplayNotice } from './Replay';
import { Badge, Card, ErrorBox, SkeletonCard, ms } from './ui';

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

export type Tone = 'plain' | 'gray' | 'indigo';
const TITLE_STYLE: Record<Tone, string> = {
  plain: '',
  gray: 'rounded-md bg-slate-600 px-2 py-0.5 text-white',
  indigo: 'rounded-md bg-indigo-600 px-2 py-0.5 text-white',
};
const CARD_STYLE: Record<Tone, string> = { plain: '', gray: '!border-slate-300', indigo: '!border-indigo-300' };

export function RunPanel({
  title,
  pipeline,
  state,
  tone = 'plain',
  answerResponse,
  note,
  supportLabel = 'support',
  onRunLive,
}: {
  title: string;
  pipeline: PipelineName;
  state: RunState;
  tone?: Tone;
  /** Response used for the answer text and its colouring (the audited baseline in compare mode). */
  answerResponse?: QueryResponse;
  /** Short explanation shown above the answer. */
  note?: ReactNode;
  supportLabel?: string;
  /** Re-run live instead of replaying a saved real run. */
  onRunLive?: () => void;
}) {
  const { response, steps, error, running } = state;
  const shown = answerResponse ?? response;
  return (
    <Card
      className={CARD_STYLE[tone]}
      title={<span className={TITLE_STYLE[tone]}>{title}</span>}
      right={
        response && (
          <span className="flex gap-3 text-xs">
            <Link className="text-indigo-700 hover:underline" to={`/insights/${response.query_id}`}>
              retrieval insights
            </Link>
            {response.verification && (
              <Link className="text-indigo-700 hover:underline" to={`/verification/${response.query_id}`}>
                verification
              </Link>
            )}
          </span>
        )
      }
    >
      {pipeline === 'proposed' && steps.length > 0 && (
        <ol className="mb-4 space-y-1.5 border-l-2 border-indigo-200 pl-3 text-xs">
          {steps.map((s, i) => (
            <li key={i}>
              <span className="font-medium">{NODE_LABELS[s.node] ?? s.node}</span>{' '}
              <span className="text-slate-500">{summarise(s.summary)}</span>
            </li>
          ))}
          {running && <li className="animate-pulse text-indigo-600">working…</li>}
        </ol>
      )}
      {running && !response && (pipeline === 'baseline' || steps.length === 0) && <SkeletonCard lines={3} />}
      {error && <ErrorBox message={error} />}
      {response && shown && (
        <>
          <ReplayNotice response={response} onRunLive={onRunLive} disabled={running} />
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
              {response.usage.input_tokens + response.usage.output_tokens} tokens
            </Badge>
            {shown.verification && (
              <Badge tone={shown.verification.passed ? 'supported' : 'partial'}>
                {supportLabel} {shown.verification.support_score.toFixed(2)}
              </Badge>
            )}
            {response.attempts.length > 1 && <Badge tone="partial">{response.attempts.length} attempts</Badge>}
          </div>
          {note && <div className="mb-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">{note}</div>}
          <AnswerView response={shown} />
        </>
      )}
    </Card>
  );
}

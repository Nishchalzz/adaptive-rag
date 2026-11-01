import { TraceView } from '../components/TraceView';
import { Badge, Bar, Card, Empty, Stat, ms, sourceLabel, statusTone } from '../components/ui';
import type { Attempt, ClaimVerification, QueryResponse } from '../types';

function ClaimCard({ claim }: { claim: ClaimVerification }) {
  const ev = claim.evidence;
  return (
    <div
      className={`rounded-lg border p-3 text-sm ${
        claim.status === 'unsupported'
          ? 'border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950/30'
          : 'border-slate-200 dark:border-slate-800'
      }`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={statusTone(claim.status)}>{claim.status}</Badge>
        <Badge>{claim.method}</Badge>
        <Badge tone={claim.citation_status === 'correct' ? 'supported' : claim.citation_status === 'misattributed' ? 'partial' : 'unsupported'}>
          citation {claim.citation_status}
        </Badge>
        {claim.failure && <Badge tone="unsupported">{claim.failure.replace('_', ' ')}</Badge>}
        <span className="text-xs text-slate-500">
          cites {claim.citations.length ? claim.citations.map((n) => `[${n}]`).join('') : 'nothing'}
          {claim.suggested_citation != null && ` · better: [${claim.suggested_citation}]`}
        </span>
      </div>
      <p className="mt-2 font-medium">{claim.text}</p>
      {ev && (
        <div className="mt-2 rounded-md bg-slate-100 p-2 dark:bg-slate-800/60">
          <div className="mb-1 text-xs text-slate-500">
            Evidence: [{ev.citation_number}] {sourceLabel(ev.filename, ev.page)} · similarity {ev.similarity.toFixed(3)}
          </div>
          <p className="text-xs italic">“{ev.text}”</p>
          <div className="mt-2 space-y-1">
            <Bar label="entailment" value={ev.entailment} color="bg-emerald-500" />
            <Bar label="neutral" value={ev.neutral} color="bg-slate-400" />
            <Bar label="contradiction" value={ev.contradiction} color="bg-red-500" />
          </div>
        </div>
      )}
      {!ev && claim.status === 'unsupported' && <p className="mt-2 text-xs text-red-700 dark:text-red-300">No supporting evidence found in the retrieved passages.</p>}
      {claim.judge_rationale && (
        <p className="mt-2 text-xs text-slate-600 dark:text-slate-400">
          Judge: {claim.judge_rationale}
          {claim.judge_quote && (
            <>
              {' '}
              Quote: “{claim.judge_quote}”{' '}
              {claim.judge_quote_verified === false ? <Badge tone="unsupported">quote not found in passage</Badge> : claim.judge_quote_verified ? <Badge tone="supported">quote verified</Badge> : null}
            </>
          )}
        </p>
      )}
    </div>
  );
}

function AttemptCard({ attempt, selected, refinement }: { attempt: Attempt; selected: boolean; refinement?: QueryResponse['refinements'][number] }) {
  const v = attempt.verification;
  return (
    <Card
      title={`Attempt ${attempt.attempt + 1}`}
      right={
        <span className="flex gap-2">
          {selected && <Badge tone="supported">returned</Badge>}
          {v && <Badge tone={v.passed ? 'supported' : 'unsupported'}>{v.passed ? 'passed' : 'failed'}</Badge>}
        </span>
      }
    >
      {refinement && (
        <div className="mb-3 rounded-lg border border-violet-300 bg-violet-50 p-2 text-xs dark:border-violet-900 dark:bg-violet-950/30">
          <strong>Refinement before this attempt:</strong> targeted retrieval for {refinement.targeted_claims.length} claim(s), added{' '}
          {refinement.added_chunk_ids.length} new chunk(s) ({ms(refinement.latency_ms)}).
          {refinement.contradicted_claims.length > 0 && <> Contradicted: {refinement.contradicted_claims.join(' | ')}.</>}
        </div>
      )}
      <p className="mb-3 whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-300">{attempt.answer}</p>
      {v ? (
        <>
          <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-5">
            <Stat label="Support score" value={v.support_score.toFixed(2)} />
            <Stat label="Citation accuracy" value={v.citation_accuracy == null ? 'n/a' : v.citation_accuracy.toFixed(2)} />
            <Stat label="Supported / partial / unsupported" value={`${v.num_supported} / ${v.num_partial} / ${v.num_unsupported}`} />
            <Stat label="Verification time" value={ms(v.latency_ms)} />
            <Stat label="Cumulative" value={ms(attempt.cumulative_latency_ms)} hint={`${attempt.cumulative_input_tokens + attempt.cumulative_output_tokens} tokens`} />
          </div>
          {v.abstained && <Empty>The answer made no factual claims (it abstained), so there was nothing to verify.</Empty>}
          {v.extraction_fallback && <p className="mb-2 text-xs text-amber-700">Claim extraction fell back to one claim per sentence.</p>}
          <div className="space-y-2">
            {v.claims.map((c) => (
              <ClaimCard key={c.claim_id} claim={c} />
            ))}
          </div>
        </>
      ) : (
        <p className="text-sm text-slate-500">Not verified.</p>
      )}
    </Card>
  );
}

function Verification({ r }: { r: QueryResponse }) {
  if (!r.verification) {
    return <Empty>This run used a pipeline without evidence verification (baseline or adaptive-only variant).</Empty>;
  }
  return (
    <div className="space-y-4">
      <Card title="Question and final answer">
        <p className="font-medium">{r.question}</p>
        <p className="mt-2 whitespace-pre-wrap text-sm">{r.answer}</p>
        {r.uncertainty?.flagged && (
          <p className="mt-2 rounded-lg border border-amber-400 bg-amber-50 p-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
            {r.uncertainty.notice}
          </p>
        )}
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <Badge>NLI model: {r.verification.nli_model}</Badge>
          <Badge>{r.attempts.length} attempt(s)</Badge>
          <Badge>{r.refinements.length} refinement(s)</Badge>
        </div>
      </Card>
      {r.attempts.map((a) => (
        <AttemptCard
          key={a.attempt}
          attempt={a}
          selected={a.attempt === r.selected_attempt}
          refinement={r.refinements.find((x) => x.attempt === a.attempt)}
        />
      ))}
    </div>
  );
}

export default function VerificationPage() {
  return <TraceView base="verification" render={(r) => <Verification r={r} />} />;
}

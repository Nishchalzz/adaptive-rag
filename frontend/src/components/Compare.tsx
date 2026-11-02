import { useState } from 'react';
import type { RunState } from '../useRun';
import {
  auditedBaseline,
  diffAnswers,
  proposedActions,
  scorecard,
  tally,
  verdictLine,
  type AuditState,
  type RowVerdict,
  type ScoreRow,
} from '../compareLogic';
import type { QueryResponse } from '../types';
import { Card, Empty, ErrorBox, InfoTip, SkeletonCard } from './ui';
import { RunPanel } from './RunPanel';

const VERDICT_STYLE: Record<RowVerdict, { text: string; cls: string }> = {
  proposed: { text: 'proposed better', cls: 'bg-indigo-50 text-indigo-800 ring-1 ring-inset ring-indigo-200' },
  baseline: { text: 'baseline better', cls: 'bg-slate-200 text-slate-800 ring-1 ring-inset ring-slate-300' },
  same: { text: 'same', cls: 'bg-white text-slate-600 ring-1 ring-inset ring-slate-200' },
  info: { text: 'info only', cls: 'bg-white text-slate-400 ring-1 ring-inset ring-slate-100' },
  pending: { text: 'waiting…', cls: 'bg-white text-slate-400 ring-1 ring-inset ring-slate-100' },
};

function VerdictBadge({ verdict }: { verdict: RowVerdict }) {
  const v = VERDICT_STYLE[verdict];
  return <span className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${v.cls}`}>{v.text}</span>;
}

function Scorecard({ rows, audit }: { rows: ScoreRow[]; audit: AuditState }) {
  const t = tally(rows);
  const unavailable = audit.status === 'error';
  const show = (v: string) => (unavailable && v === '…' ? 'n/a' : v);
  return (
    <Card title="Difference summary" right={<span className="text-xs text-slate-500">proposed − baseline</span>}>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full min-w-[560px] text-sm">
          <thead className="bg-slate-50 text-xs font-medium text-slate-600">
            <tr>
              <th className="px-3 py-2 text-left">Measure</th>
              <th className="px-3 py-2 text-right"><span className="rounded bg-slate-600 px-1.5 py-0.5 text-white">Baseline</span></th>
              <th className="px-3 py-2 text-right"><span className="rounded bg-indigo-600 px-1.5 py-0.5 text-white">Proposed</span></th>
              <th className="px-3 py-2 text-right">Difference</th>
              <th className="px-3 py-2 text-right">Result</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((r) => (
              <tr key={r.label}>
                <td className="px-3 py-2 text-slate-800">
                  {r.label}
                  {r.hint && <InfoTip>{r.hint}</InfoTip>}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{show(r.baseline)}</td>
                <td className="px-3 py-2 text-right font-medium tabular-nums text-slate-900">{show(r.proposed)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-500">{show(r.delta)}</td>
                <td className="px-3 py-2 text-right">
                  <VerdictBadge verdict={unavailable && r.verdict === 'pending' ? 'info' : r.verdict} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-600">
        Across the rows that have a better direction: proposed better on <strong>{t.proposed}</strong>, baseline better on <strong>{t.baseline}</strong>,
        same on <strong>{t.same}</strong>. “Same” means within a small tolerance. This is a single question, so treat it as an example, not evidence.
      </p>
      {audit.status === 'running' && (
        <p className="mt-1 animate-pulse text-xs text-indigo-700">Auditing the baseline answer with the same claim checker so reliability can be compared fairly…</p>
      )}
      {audit.status === 'error' && <div className="mt-2"><ErrorBox message={`Baseline audit failed: ${audit.message}. Claim rows are unavailable.`} /></div>}
    </Card>
  );
}

function Differences({ lines }: { lines: string[] }) {
  return (
    <Card title="What the proposed system did differently">
      <ul className="space-y-2 text-sm leading-relaxed text-slate-700">
        {lines.map((l, i) => (
          <li key={i} className="flex gap-2">
            <span aria-hidden className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-400" />
            <span>{l}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function AnswerDiff({ b, p, verdict }: { b: QueryResponse; p: QueryResponse; verdict: string }) {
  const d = diffAnswers(b, p);
  return (
    <Card title="Answer comparison">
      <p className="mb-3 rounded-lg bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800">{verdict}</p>
      <div className="grid gap-3 md:grid-cols-3">
        <div>
          <div className="mb-1 text-xs font-medium text-slate-600">Shared by both ({d.shared.length})</div>
          <ul className="space-y-1.5 text-sm">
            {d.shared.length === 0 && <li className="text-xs text-slate-400">none</li>}
            {d.shared.map((s, i) => <li key={i} className="rounded-md border border-slate-200 bg-white px-2 py-1.5 text-slate-700">{s}</li>)}
          </ul>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-slate-700"><span className="rounded bg-slate-600 px-1.5 py-0.5 text-white">Only in baseline</span> ({d.onlyBaseline.length})</div>
          <ul className="space-y-1.5 text-sm">
            {d.onlyBaseline.length === 0 && <li className="text-xs text-slate-400">none</li>}
            {d.onlyBaseline.map((s, i) => <li key={i} className="rounded-md border border-slate-300 bg-slate-100 px-2 py-1.5 text-slate-800">{s}</li>)}
          </ul>
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-indigo-800"><span className="rounded bg-indigo-600 px-1.5 py-0.5 text-white">Only in proposed</span> ({d.onlyProposed.length})</div>
          <ul className="space-y-1.5 text-sm">
            {d.onlyProposed.length === 0 && <li className="text-xs text-slate-400">none</li>}
            {d.onlyProposed.map((s, i) => <li key={i} className="rounded-md border border-indigo-200 bg-indigo-50 px-2 py-1.5 text-indigo-900">{s}</li>)}
          </ul>
        </div>
      </div>
    </Card>
  );
}

export interface HistoryItem {
  id: string;
  question: string;
  verdict: string;
}

export function CompareHistory({ items }: { items: HistoryItem[] }) {
  const [open, setOpen] = useState(true);
  if (items.length === 0) return null;
  return (
    <Card
      title={`Recent comparisons (${items.length})`}
      right={<button className="text-xs text-slate-500 hover:underline" onClick={() => setOpen(!open)}>{open ? 'hide' : 'show'}</button>}
    >
      {open && (
        <ul className="divide-y divide-slate-100 text-sm">
          {items.map((h) => (
            <li key={h.id} className="py-2">
              <div className="font-medium text-slate-800">{h.question}</div>
              <div className="text-xs text-slate-600">{h.verdict}</div>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-xs text-slate-400">Kept in memory for this session only.</p>
    </Card>
  );
}

/** Side-by-side layout: scorecard on top, two same-shaped answer columns, then explanations. */
export function SideBySide({
  baselineState,
  proposedState,
  audit,
  onRunLive,
  busy,
}: {
  baselineState: RunState;
  proposedState: RunState;
  audit: AuditState;
  /** Re-run baseline, proposed and the audit live instead of replaying saved real runs. */
  onRunLive?: () => void;
  busy?: boolean;
}) {
  const b = baselineState.response;
  const p = proposedState.response;
  const k = b?.retrieval.k_requested ?? 5;
  const audited = b ? auditedBaseline(b, audit) : null;

  const baselineNote =
    audit.status === 'running' ? (
      <span className="animate-pulse text-indigo-700">Auditing this answer with the same claim checker used by the proposed system…</span>
    ) : audit.status === 'done' ? (
      <>
        Sentences are coloured by an audit with the same claim checker (not part of the baseline).
        {audited && audited.total > 0 && audited.matched < audited.total && (
          <span className="text-amber-700"> {audited.total - audited.matched} of {audited.total} sentences could not be matched to the audit run because wording differed, so they are not coloured.</span>
        )}
      </>
    ) : audit.status === 'error' ? (
      'The audit could not be run, so this answer is shown without verification colours.'
    ) : undefined;

  const bothDone = !!b && !!p;
  const replayed = [b, p, audit.status === 'done' ? audit.response : null].filter((r) => r?.cached).length;
  const rows = bothDone ? scorecard(b, p, audit) : null;
  const verdict = bothDone ? verdictLine(b, p, audit) : '';

  return (
    <div className="space-y-6">
      {replayed > 0 && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span
            title="Saved sample: real runs recorded earlier on this question and document set. Latency and tokens in the scorecard are from those original runs."
            className="rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-600 ring-1 ring-inset ring-slate-200"
          >
            Saved sample
          </span>
          {onRunLive && (
            <button type="button" onClick={onRunLive} disabled={busy} className="text-indigo-700 hover:underline disabled:opacity-50">
              Run live
            </button>
          )}
        </div>
      )}
      {rows ? <Scorecard rows={rows} audit={audit} /> : (baselineState.running || proposedState.running) && <SkeletonCard lines={5} />}

      <div className="grid gap-4 lg:grid-cols-2">
        <RunPanel
          title={`Baseline: fixed top-${k} RAG`}
          pipeline="baseline"
          state={baselineState}
          tone="gray"
          answerResponse={audited?.response}
          onRunLive={onRunLive}
          note={baselineNote}
          supportLabel="audited support"
        />
        <RunPanel
          title="Proposed: adaptive + verified"
          pipeline="proposed"
          state={proposedState}
          tone="indigo"
          onRunLive={onRunLive}
          note={p ? 'Verified by the system’s own claim checker; unsupported sentences are removed from the answer shown here.' : undefined}
        />
      </div>

      {bothDone && <Differences lines={proposedActions(b, p)} />}
      {bothDone && <AnswerDiff b={b} p={p} verdict={verdict} />}
      {!bothDone && !baselineState.running && !proposedState.running && !baselineState.error && !proposedState.error && (
        <Empty>Ask a question to compare the two systems.</Empty>
      )}
    </div>
  );
}

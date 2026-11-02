/**
 * Pure comparison logic for the side-by-side Chat mode. Everything here is computed from the two
 * responses (and the baseline audit); no number or verdict is hardcoded.
 */
import type { Attempt, QueryResponse, Sentence, VerificationReport } from './types';

export type AuditState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'done'; response: QueryResponse }
  | { status: 'error'; message: string };

/** Same retrieval (fixed k, same prompt) as the baseline, verified once, no retries: attempts[0] is the baseline-equivalent answer. */
export const AUDIT_OVERRIDES = {
  proposed: { adaptive: false, verify: true },
  verification: { max_retries: 0 },
};

/* ------------------------------------------------------------------ text helpers */

const CITATION = /\[\d+(?:\s*[,;]\s*\d+)*\]/g;
const norm = (t: string) =>
  t.replace(CITATION, ' ').toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
const wordSet = (t: string) => new Set(norm(t).split(' ').filter(Boolean));

/** Two sentences count as the same statement when their wording overlaps almost completely. */
export function similar(a: string, b: string): boolean {
  const x = wordSet(a);
  const y = wordSet(b);
  if (x.size === 0 || y.size === 0) return false;
  let inter = 0;
  x.forEach((w) => y.has(w) && inter++);
  return inter / (x.size + y.size - inter) >= 0.8;
}

const sentencesOf = (r: QueryResponse): Sentence[] =>
  r.sentences.length ? r.sentences : r.answer.trim() ? [{ index: 0, text: r.answer, citation_numbers: [], status: null }] : [];

/* ------------------------------------------------------------------ claim counts */

export interface Counts {
  supported: number;
  partial: number;
  unsupported: number;
  total: number;
  score: number | null;
}

function countClaims(report: VerificationReport, keepSentence?: Set<number>): Counts {
  const claims = keepSentence ? report.claims.filter((c) => keepSentence.has(c.sentence_index)) : report.claims;
  const supported = claims.filter((c) => c.status === 'supported').length;
  const partial = claims.filter((c) => c.status === 'partial').length;
  const unsupported = claims.filter((c) => c.status === 'unsupported').length;
  const total = claims.length;
  return { supported, partial, unsupported, total, score: total ? (supported + 0.5 * partial) / total : null };
}

/** Claims of the answer the user actually receives (unsupported sentences removed by the proposed system are excluded). */
export function proposedCounts(p: QueryResponse): Counts | null {
  return p.verification ? countClaims(p.verification, new Set(p.sentences.map((s) => s.index))) : null;
}

export function auditAttempt(audit: AuditState): Attempt | null {
  return audit.status === 'done' ? audit.response.attempts[0] ?? null : null;
}

export function baselineCounts(audit: AuditState): Counts | null {
  const report = auditAttempt(audit)?.verification;
  return report ? countClaims(report) : null;
}

/** Baseline response with its sentences coloured by the audit, matched by wording. */
export function auditedBaseline(b: QueryResponse, audit: AuditState): { response: QueryResponse; matched: number; total: number } {
  const attempt = auditAttempt(audit);
  const sentences = sentencesOf(b);
  if (!attempt || !attempt.verification) return { response: b, matched: 0, total: sentences.length };
  let matched = 0;
  const coloured = sentences.map((s) => {
    const hit = attempt.sentences.find((a) => similar(a.text, s.text));
    if (hit) matched++;
    return { ...s, status: hit?.status ?? null };
  });
  return { response: { ...b, sentences: coloured, verification: attempt.verification, uncertainty: null }, matched, total: sentences.length };
}

/* ------------------------------------------------------------------ scorecard */

export type RowVerdict = 'proposed' | 'baseline' | 'same' | 'info' | 'pending';
export interface ScoreRow {
  label: string;
  hint?: string;
  baseline: string;
  proposed: string;
  delta: string;
  verdict: RowVerdict;
}

type Better = 'lower' | 'higher' | 'none';
interface Tolerance {
  abs: number;
  rel: number;
}

function judge(bv: number | null, pv: number | null, better: Better, tol: Tolerance): { verdict: RowVerdict; delta: number | null } {
  if (bv == null || pv == null) return { verdict: 'pending', delta: null };
  const delta = pv - bv;
  const limit = Math.max(tol.abs, tol.rel * Math.max(Math.abs(bv), Math.abs(pv)));
  if (better === 'none') return { verdict: 'info', delta };
  if (Math.abs(delta) <= limit) return { verdict: 'same', delta };
  const proposedWins = better === 'lower' ? delta < 0 : delta > 0;
  return { verdict: proposedWins ? 'proposed' : 'baseline', delta };
}

const sign = (d: number) => (d > 0 ? '+' : d < 0 ? '−' : '');
const fmtInt = (v: number) => String(Math.round(v));
const fmtMs = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`);
const fmt2 = (v: number) => v.toFixed(2);

function row(
  label: string,
  bv: number | null,
  pv: number | null,
  better: Better,
  tol: Tolerance,
  fmt: (v: number) => string,
  opts: { hint?: string; pendingWhenMissing?: boolean } = {},
): ScoreRow {
  const j = judge(bv, pv, better, tol);
  const missing = bv == null || pv == null;
  return {
    label,
    hint: opts.hint,
    baseline: bv == null ? '…' : fmt(bv),
    proposed: pv == null ? '…' : fmt(pv),
    delta: j.delta == null ? '' : j.delta === 0 || Math.abs(j.delta) < 1e-12 ? '0' : `${sign(j.delta)}${fmt(Math.abs(j.delta))}`,
    // A side that can never be computed (e.g. no price for the model) is shown as info, not as "pending".
    verdict: missing && opts.pendingWhenMissing === false ? 'info' : j.verdict,
  };
}

export function scorecard(b: QueryResponse, p: QueryResponse, audit: AuditState): ScoreRow[] {
  const bc = baselineCounts(audit);
  const pc = proposedCounts(p);
  const auditNote = 'Baseline figures come from auditing its answer with the same claim checker';
  const claimRow = (label: string, pick: (c: Counts) => number, better: Better) =>
    row(label, bc ? pick(bc) : null, pc ? pick(pc) : null, better, { abs: 0.5, rel: 0 }, fmtInt, { hint: auditNote });
  return [
    row('Chunks retrieved', b.retrieval.num_retrieved, p.retrieval.num_retrieved, 'lower', { abs: 0.5, rel: 0 }, fmtInt, {
      hint: 'Chunks given to the language model. Fewer means a shorter prompt.',
    }),
    row('Retrieval latency', b.retrieval.latency_ms, p.retrieval.latency_ms, 'lower', { abs: 20, rel: 0.05 }, fmtMs),
    row('Total latency', b.total_latency_ms, p.total_latency_ms, 'lower', { abs: 100, rel: 0.05 }, fmtMs, {
      hint: 'The proposed system also analyses the question and verifies the answer, which takes time.',
    }),
    row('Tokens (input + output)', b.usage.input_tokens + b.usage.output_tokens, p.usage.input_tokens + p.usage.output_tokens, 'lower', { abs: 5, rel: 0.02 }, fmtInt),
    claimRow('Supported claims', (c) => c.supported, 'higher'),
    claimRow('Partially supported claims', (c) => c.partial, 'none'),
    claimRow('Unsupported claims', (c) => c.unsupported, 'lower'),
    row('Support score', bc?.score ?? null, pc?.score ?? null, 'higher', { abs: 0.01, rel: 0 }, fmt2, {
      hint: '(supported + ½ × partial) ÷ claims in the delivered answer. Blank when the answer makes no factual claims.',
      pendingWhenMissing: audit.status === 'running' ? true : false,
    }),
    row('Valid citations', b.citations.length, p.citations.length, 'none', { abs: 0.5, rel: 0 }, fmtInt, {
      hint: 'Distinct passages cited that exist in the retrieved context.',
    }),
    row('Invalid citations', b.invalid_citations.length, p.invalid_citations.length, 'lower', { abs: 0.5, rel: 0 }, fmtInt, {
      hint: 'Citation numbers in the answer that match no retrieved passage.',
    }),
    row('Answer attempts', 1, p.attempts.length || 1, 'none', { abs: 0.5, rel: 0 }, fmtInt, {
      hint: 'More than one attempt means verification failed and the answer was regenerated.',
    }),
  ];
}

export function tally(rows: ScoreRow[]) {
  const count = (v: RowVerdict) => rows.filter((r) => r.verdict === v).length;
  return { proposed: count('proposed'), baseline: count('baseline'), same: count('same') };
}

/* ------------------------------------------------------------------ answer diff */

export interface AnswerDiff {
  shared: string[];
  onlyBaseline: string[];
  onlyProposed: string[];
}

export function diffAnswers(b: QueryResponse, p: QueryResponse): AnswerDiff {
  const bs = sentencesOf(b);
  const ps = sentencesOf(p);
  const usedP = new Set<number>();
  const shared: string[] = [];
  const onlyBaseline: string[] = [];
  for (const s of bs) {
    const i = ps.findIndex((q, idx) => !usedP.has(idx) && similar(q.text, s.text));
    if (i >= 0) {
      usedP.add(i);
      shared.push(s.text);
    } else {
      onlyBaseline.push(s.text);
    }
  }
  const onlyProposed = ps.filter((_, idx) => !usedP.has(idx)).map((s) => s.text);
  return { shared, onlyBaseline, onlyProposed };
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

/** One line summarising how the two runs differ; used above the diff and in the comparison history. */
export function verdictLine(b: QueryResponse, p: QueryResponse, audit: AuditState): string {
  const d = diffAnswers(b, p);
  const sameAnswer = d.onlyBaseline.length === 0 && d.onlyProposed.length === 0;
  const parts: string[] = [];
  parts.push(
    sameAnswer
      ? 'Same answer'
      : `Answers differ (${plural(d.shared.length, 'shared sentence')}, ${d.onlyBaseline.length} only in baseline, ${d.onlyProposed.length} only in proposed)`,
  );
  const chunkDiff = b.retrieval.num_retrieved - p.retrieval.num_retrieved;
  if (chunkDiff > 0) parts.push(`proposed used ${plural(chunkDiff, 'fewer chunk')}`);
  else if (chunkDiff < 0) parts.push(`proposed used ${plural(-chunkDiff, 'more chunk')}`);
  const removed = p.uncertainty?.removed_sentences.length ?? 0;
  if (removed > 0) parts.push(`proposed removed ${plural(removed, 'unsupported statement')}`);
  const bc = baselineCounts(audit);
  const pc = proposedCounts(p);
  if (bc && pc && bc.unsupported !== pc.unsupported) {
    parts.push(`unsupported claims in the delivered answer: baseline ${bc.unsupported}, proposed ${pc.unsupported}`);
  }
  if (p.attempts.length > 1) parts.push(`proposed regenerated the answer ${plural(p.attempts.length - 1, 'time')}`);
  const meaningful = parts.length > 1 || !sameAnswer;
  const line = meaningful ? parts.join('; ') : 'No meaningful difference';
  return line.charAt(0).toUpperCase() + line.slice(1);
}

/* ------------------------------------------------------------------ what the proposed system did differently */

export function proposedActions(b: QueryResponse, p: QueryResponse): string[] {
  const out: string[] = [];
  const c = p.complexity;
  if (c) {
    const needs = c.information_needs.slice(0, 4).map((n) => `“${n}”`).join(', ');
    out.push(
      `Rated the question ${c.label} (score ${c.score.toFixed(2)}, ${c.reasoning_type}${c.requires_multiple_sources ? ', needs several sources' : ''}) ` +
        (c.fallback
          ? 'using lexical cues only, because the language-model analysis was unavailable.'
          : `because it identified ${plural(c.information_needs.length, 'information need')}: ${needs}${c.information_needs.length > 4 ? ', …' : ''}.`),
    );
  } else {
    out.push('No complexity analysis was run for this answer.');
  }

  const rt = p.retrieval;
  const baselineK = b.retrieval.k_requested;
  let depth =
    rt.k_requested === baselineK
      ? `Chose a retrieval depth of k=${rt.k_requested}, the same as the baseline's fixed ${baselineK}`
      : `Chose k=${rt.k_requested} instead of the baseline's fixed ${baselineK}`;
  if (rt.expansions > 0 && rt.k_final != null) {
    depth += `, then expanded retrieval to k=${rt.k_final} because confidence was low${rt.confidence ? ` (${rt.confidence.score.toFixed(2)})` : ''}`;
  } else if (rt.confidence) {
    depth += `; no expansion was needed (retrieval confidence ${rt.confidence.score.toFixed(2)})`;
  }
  if (rt.queries.length > 1) depth += `; searched ${plural(rt.queries.length - 1, 'sub-question')} separately and merged the results`;
  out.push(`${depth}.`);

  const first = p.attempts[0]?.verification;
  if (first) {
    const flagged = first.claims.filter((x) => x.status === 'unsupported');
    out.push(
      `Checked ${plural(first.claims.length, 'claim')} against their cited evidence: ${first.num_supported} supported, ${first.num_partial} partial, ${first.num_unsupported} unsupported.` +
        (flagged.length ? ` Flagged as unsupported: ${flagged.slice(0, 3).map((x) => `“${x.text}”`).join('; ')}${flagged.length > 3 ? '; …' : ''}.` : ''),
    );
  }

  if (p.attempts.length > 1) {
    const added = p.refinements.reduce((n, r) => n + r.added_chunk_ids.length, 0);
    out.push(
      `Regenerated the answer ${plural(p.attempts.length - 1, 'time')}${added ? ` after retrieving ${plural(added, 'extra chunk')} for the weak claims` : ''}; ` +
        `attempt ${(p.selected_attempt ?? p.attempts.length - 1) + 1} was returned.`,
    );
  } else if (first) {
    out.push('Did not need to regenerate the answer.');
  }

  const removed = p.uncertainty?.removed_sentences ?? [];
  if (removed.length) out.push(`Removed ${plural(removed.length, 'unsupported sentence')} from the final answer: ${removed.slice(0, 2).map((s) => `“${s}”`).join('; ')}.`);
  if (p.uncertainty?.flagged && p.uncertainty.notice) out.push(`Added an uncertainty notice: “${p.uncertainty.notice}”`);
  if (first && !removed.length && !p.uncertainty?.flagged) out.push('Nothing had to be removed and no uncertainty notice was needed.');

  const d = diffAnswers(b, p);
  out.push(
    d.onlyBaseline.length === 0 && d.onlyProposed.length === 0
      ? 'The final answers say the same thing.'
      : `The final answers differ: ${plural(d.shared.length, 'sentence')} shared, ${d.onlyBaseline.length} only in the baseline, ${d.onlyProposed.length} only in the proposed answer.`,
  );
  return out;
}

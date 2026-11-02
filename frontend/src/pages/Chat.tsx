import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { CompareHistory, SideBySide, type HistoryItem } from '../components/Compare';
import { RunPanel } from '../components/RunPanel';
import { Card, Empty, PageHeader, btnPrimary, inputBase } from '../components/ui';
import { AUDIT_OVERRIDES, verdictLine, type AuditState } from '../compareLogic';
import { useRun } from '../useRun';

type Mode = 'proposed' | 'baseline' | 'both';

export default function Chat() {
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<Mode>('proposed');
  const [demo, setDemo] = useState<{ category: string; question: string; featured?: boolean }[]>([]);
  const [demoNote, setDemoNote] = useState<string | undefined>();
  const proposed = useRun();
  const baseline = useRun();
  const [audit, setAudit] = useState<AuditState>({ status: 'idle' });
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const recorded = useRef(new Set<string>());
  // When true, every request of the current comparison bypasses the replay cache (the audit included).
  const forceLive = useRef(false);
  const [lastQuestion, setLastQuestion] = useState('');

  useEffect(() => {
    api.demoQuestions().then((d) => { setDemo(d.questions); setDemoNote(d.selection_note); }).catch(() => undefined);
  }, []);

  // Fair reliability comparison: the baseline has no verification of its own, so once its answer is in,
  // audit it with the same claim checker (fixed-k retrieval, same prompt, no retries).
  const baselineResponse = baseline.state.response;
  useEffect(() => {
    if (mode !== 'both' || !baselineResponse) {
      setAudit({ status: 'idle' });
      return;
    }
    let cancelled = false;
    setAudit({ status: 'running' });
    api
      .query({
        question: baselineResponse.question,
        pipeline: 'proposed',
        overrides: AUDIT_OVERRIDES,
        ...(forceLive.current ? { use_cache: false } : {}),
      })
      .then((response) => !cancelled && setAudit({ status: 'done', response }))
      .catch((e: Error) => !cancelled && setAudit({ status: 'error', message: e.message }));
    return () => {
      cancelled = true;
    };
  }, [baselineResponse?.query_id, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Remember the verdict line of each finished comparison (in memory only).
  const proposedResponse = proposed.state.response;
  useEffect(() => {
    if (mode !== 'both' || !baselineResponse || !proposedResponse) return;
    if (audit.status !== 'done' && audit.status !== 'error') return;
    if (recorded.current.has(baselineResponse.query_id)) return;
    recorded.current.add(baselineResponse.query_id);
    const item = { id: baselineResponse.query_id, question: baselineResponse.question, verdict: verdictLine(baselineResponse, proposedResponse, audit) };
    setHistory((h) => [item, ...h].slice(0, 8));
  }, [mode, baselineResponse, proposedResponse, audit]);

  const busy = proposed.state.running || baseline.state.running || audit.status === 'running';

  const ask = (q: string, live = false) => {
    const text = q.trim();
    if (text.length < 3 || busy) return;
    forceLive.current = live;
    setLastQuestion(text);
    if (mode !== 'baseline') proposed.run(text, 'proposed', undefined, !live);
    if (mode !== 'proposed') baseline.run(text, 'baseline', undefined, !live);
  };
  const runLive = () => ask(lastQuestion, true);

  const showProposed = mode !== 'baseline' && (proposed.state.running || proposed.state.response || proposed.state.error);
  const showBaseline = mode !== 'proposed' && (baseline.state.running || baseline.state.response || baseline.state.error);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Chat"
        subtitle="Ask a question about the indexed documents. Choose Side by side to see the conventional baseline and the proposed system answer the same question."
      />
      <Card title="Ask a question">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(question);
          }}
          className="space-y-3"
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                ask(question);
              }
            }}
            rows={2}
            placeholder="Ask something about the indexed documents…"
            className={inputBase}
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex overflow-hidden rounded-lg border border-slate-300 bg-white text-sm" role="group" aria-label="Pipeline">
              {(
                [
                  ['proposed', 'Proposed'],
                  ['baseline', 'Baseline'],
                  ['both', 'Side by side'],
                ] as [Mode, string][]
              ).map(([value, label]) => (
                <button
                  type="button"
                  key={value}
                  onClick={() => setMode(value)}
                  className={`px-3 py-2 font-medium ${mode === value ? 'bg-indigo-600 text-white' : 'text-slate-600 hover:bg-slate-50'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={busy || question.trim().length < 3}
              className={btnPrimary}
            >
              {busy ? 'Working…' : 'Ask'}
            </button>
          </div>
        </form>
        {demo.length > 0 && (
          <div className="mt-3 space-y-2">
            {[
              { label: 'Main examples', items: demo.filter((d) => d.featured) },
              { label: 'More examples', items: demo.filter((d) => !d.featured) },
            ]
              .filter((row) => row.items.length > 0)
              .map((row) => (
                <div key={row.label} className="flex flex-wrap items-center gap-2">
                  <span className="w-28 shrink-0 text-xs font-medium text-slate-500">{row.label}</span>
                  {row.items.map((d) => (
                    <button
                      key={d.question}
                      onClick={() => {
                        setQuestion(d.question);
                        ask(d.question);
                      }}
                      title={d.question}
                      className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-700 hover:border-indigo-300 hover:bg-indigo-50"
                    >
                      <span className="mr-1 text-slate-400">{d.category}</span>
                      {d.question.length > 60 ? `${d.question.slice(0, 60)}…` : d.question}
                    </button>
                  ))}
                </div>
              ))}
            {demoNote && <p className="text-xs text-slate-500">Sample questions: {demoNote}</p>}
          </div>
        )}
      </Card>

      {!showProposed && !showBaseline && <Empty>Pick an example question above or type your own. Answers appear here with clickable citations and a colour-coded verification status.</Empty>}

      {mode === 'both' ? (
        (showProposed || showBaseline) && <SideBySide baselineState={baseline.state} proposedState={proposed.state} audit={audit} onRunLive={runLive} busy={busy} />
      ) : (
        <div className="grid gap-4">
          {showProposed && <RunPanel title="Proposed: adaptive + verified" pipeline="proposed" state={proposed.state} tone="indigo" onRunLive={runLive} />}
          {showBaseline && <RunPanel title="Baseline: fixed top-k RAG" pipeline="baseline" state={baseline.state} tone="gray" onRunLive={runLive} />}
        </div>
      )}
      {mode === 'both' && <CompareHistory items={history} />}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { api } from '../api';
import { RunPanel } from '../components/RunPanel';
import { Card } from '../components/ui';
import { useRun } from '../useRun';

type Mode = 'proposed' | 'baseline' | 'both';

export default function Chat() {
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<Mode>('proposed');
  const [demo, setDemo] = useState<{ category: string; question: string }[]>([]);
  const proposed = useRun();
  const baseline = useRun();

  useEffect(() => {
    api.demoQuestions().then((d) => setDemo(d.questions)).catch(() => undefined);
  }, []);

  const busy = proposed.state.running || baseline.state.running;

  const ask = (q: string) => {
    const text = q.trim();
    if (text.length < 3 || busy) return;
    if (mode !== 'baseline') proposed.run(text, 'proposed');
    if (mode !== 'proposed') baseline.run(text, 'baseline');
  };

  const showProposed = mode !== 'baseline' && (proposed.state.running || proposed.state.response || proposed.state.error);
  const showBaseline = mode !== 'proposed' && (baseline.state.running || baseline.state.response || baseline.state.error);

  return (
    <div className="space-y-4">
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
            className="w-full rounded-lg border border-slate-300 bg-white p-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex overflow-hidden rounded-lg border border-slate-300 text-sm dark:border-slate-700">
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
                  className={`px-3 py-1.5 ${mode === value ? 'bg-sky-600 text-white' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={busy || question.trim().length < 3}
              className="rounded-lg bg-sky-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {busy ? 'Working…' : 'Ask'}
            </button>
          </div>
        </form>
        {demo.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {demo.map((d) => (
              <button
                key={d.question}
                onClick={() => {
                  setQuestion(d.question);
                  ask(d.question);
                }}
                title={d.category}
                className="rounded-full border border-slate-300 px-3 py-1 text-xs hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
              >
                <span className="mr-1 text-slate-400">{d.category}</span>
                {d.question.length > 70 ? `${d.question.slice(0, 70)}…` : d.question}
              </button>
            ))}
          </div>
        )}
      </Card>

      <div className={`grid gap-4 ${showProposed && showBaseline ? 'lg:grid-cols-2' : ''}`}>
        {showProposed && <RunPanel title="Proposed: adaptive + verified" pipeline="proposed" state={proposed.state} />}
        {showBaseline && <RunPanel title="Baseline: fixed top-k" pipeline="baseline" state={baseline.state} />}
      </div>
    </div>
  );
}

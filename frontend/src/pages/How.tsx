import { Link } from 'react-router-dom';
import { Card, PageHeader } from '../components/ui';

interface Step {
  title: string;
  body: string;
  tone: 'plain' | 'adaptive' | 'verify';
}

const STEPS: Step[] = [
  { title: '1. Your question', body: 'A question is typed in the chat box.', tone: 'plain' },
  {
    title: '2. Complexity assessment',
    body: 'An LLM lists the facts needed (sub-questions) and the reasoning type; lexical cues are added. Together they give a score from 0 to 1 and a label: simple, moderate or complex.',
    tone: 'adaptive',
  },
  {
    title: '3. Adaptive depth k',
    body: 'The score picks how many chunks to retrieve, inside a band for its label (simple 1–3, moderate 4–7, complex 8–12 by default).',
    tone: 'adaptive',
  },
  {
    title: '4. Retrieve',
    body: 'The question and each sub-question are searched separately in the vector database, then merged with Reciprocal Rank Fusion. Low retrieval confidence triggers one bounded expansion.',
    tone: 'adaptive',
  },
  { title: '5. Generate with citations', body: 'The LLM answers using only the numbered passages and cites [n] after each factual sentence.', tone: 'plain' },
  { title: '6. Extract claims', body: 'The answer is split into single, self-contained factual claims. Each inherits the citations of its sentence.', tone: 'verify' },
  {
    title: '7. Verify each claim',
    body: 'An NLI model checks whether the cited text entails the claim. Ambiguous cases go to an LLM judge that must quote its evidence. Result: supported, partial or unsupported.',
    tone: 'verify',
  },
  {
    title: '8. Refine or finalise',
    body: 'If claims fail, targeted retrieval adds evidence and the answer is regenerated (up to a retry limit). The best attempt is returned; unsupported sentences are removed and uncertainty is stated.',
    tone: 'verify',
  },
];

const TONES = {
  plain: 'border-slate-200 bg-white',
  adaptive: 'border-indigo-200 bg-indigo-50',
  verify: 'border-emerald-200 bg-emerald-50',
} as const;

export default function How() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="How the system works"
        subtitle="A conventional RAG system always retrieves the same number of chunks and trusts the answer. This prototype changes both ends of that pipeline and measures whether the changes help."
      />

      <div className="flex flex-wrap gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-slate-300 bg-white" /> Standard RAG step</span>
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-indigo-300 bg-indigo-100" /> Contribution 1: query-adaptive retrieval</span>
        <span className="flex items-center gap-2"><i className="h-3 w-3 rounded border border-emerald-300 bg-emerald-100" /> Contribution 2: evidence verification</span>
      </div>

      <ol className="grid gap-3 md:grid-cols-2">
        {STEPS.map((s, i) => (
          <li key={s.title} className={`relative rounded-xl border p-4 ${TONES[s.tone]}`}>
            <h3 className="text-sm font-semibold text-slate-900">{s.title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-slate-700">{s.body}</p>
            {i < STEPS.length - 1 && i % 2 === 0 && (
              <span aria-hidden className="absolute -right-3 top-1/2 hidden -translate-y-1/2 text-slate-300 md:block">→</span>
            )}
          </li>
        ))}
      </ol>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="The baseline it is compared with">
          <p className="text-sm leading-relaxed text-slate-700">
            Fixed top-k retrieval (default k=5) feeding the same prompt, model and documents. No complexity analysis, no verification. Because
            everything else is identical, any difference in the results comes from the two contributions above.
          </p>
        </Card>
        <Card title="The four variants in the ablation study">
          <ul className="space-y-1 text-sm text-slate-700">
            <li><strong>B</strong> Baseline: fixed k, no verification</li>
            <li><strong>A</strong> Adaptive retrieval only</li>
            <li><strong>V</strong> Verification only (fixed k)</li>
            <li><strong>A+V</strong> Adaptive retrieval and verification: the full system</li>
          </ul>
          <p className="mt-2 text-xs text-slate-500">A− (adaptive depth without sub-question decomposition) separates the effect of depth from decomposition.</p>
        </Card>
      </div>

      <Card title="Where to look next">
        <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
          <li><Link className="text-indigo-700 hover:underline" to="/">Chat</Link>: ask a question and watch each step stream in.</li>
          <li><Link className="text-indigo-700 hover:underline" to="/insights">Retrieval Insights</Link>: the complexity score, chosen k and retrieved chunks.</li>
          <li><Link className="text-indigo-700 hover:underline" to="/verification">Evidence Verification</Link>: every claim, its evidence and its verdict.</li>
        </ul>
      </Card>
    </div>
  );
}

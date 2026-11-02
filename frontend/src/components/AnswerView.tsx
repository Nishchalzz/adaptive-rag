import { useState } from 'react';
import type { QueryResponse, RetrievedChunk, Sentence } from '../types';
import { Badge, Card, sourceLabel, statusStyles } from './ui';

const CITATION = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;

function SentenceText({ sentence, onCite }: { sentence: Sentence; onCite: (n: number) => void }) {
  const parts: (string | number[])[] = [];
  let last = 0;
  for (const m of sentence.text.matchAll(CITATION)) {
    parts.push(sentence.text.slice(last, m.index));
    parts.push(m[1].split(/[,;]/).map((s) => Number(s.trim())));
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(sentence.text.slice(last));
  const style = sentence.status ? statusStyles[sentence.status] : '';
  return (
    <span className={`rounded px-0.5 ${style}`} title={sentence.status ? `Verification: ${sentence.status}` : undefined}>
      {parts.map((p, i) =>
        typeof p === 'string' ? (
          <span key={i}>{p}</span>
        ) : (
          p.map((n) => (
            <button
              key={`${i}-${n}`}
              onClick={() => onCite(n)}
              className="mx-0.5 rounded bg-slate-900/10 px-1 text-xs font-semibold text-indigo-700 hover:bg-indigo-200"
            >
              [{n}]
            </button>
          ))
        ),
      )}{' '}
    </span>
  );
}

export function ChunkPopover({ chunk, onClose }: { chunk: RetrievedChunk | undefined; onClose: () => void }) {
  if (!chunk) return null;
  return (
    <div className="mt-3 rounded-lg border border-indigo-300 bg-indigo-50 p-3 text-sm">
      <div className="mb-1 flex items-center justify-between">
        <strong>
          [{chunk.citation_number}] {sourceLabel(chunk.filename, chunk.page)}
          <span className="ml-2 font-normal text-slate-500">similarity {chunk.score.toFixed(3)}</span>
        </strong>
        <button onClick={onClose} className="text-xs text-slate-500 hover:underline">
          close
        </button>
      </div>
      <p className="whitespace-pre-wrap">{chunk.text}</p>
    </div>
  );
}

export function AnswerView({ response }: { response: QueryResponse }) {
  const [open, setOpen] = useState<number | null>(null);
  const sentences: Sentence[] = response.sentences.length
    ? response.sentences
    : [{ index: 0, text: response.answer, citation_numbers: [], status: null }];
  const chunk = response.retrieval.chunks.find((c) => c.citation_number === open);
  const verified = response.verification !== null;

  return (
    <div>
      {response.uncertainty?.flagged && (
        <div className="mb-3 rounded-lg border border-amber-400 bg-amber-50 p-3 text-sm text-amber-900">
          <strong>Uncertainty: </strong>
          {response.uncertainty.notice}
          {response.uncertainty.unsupported_claims.length > 0 && (
            <ul className="mt-1 list-disc pl-5">
              {response.uncertainty.unsupported_claims.map((c) => (
                <li key={c}>Unverified: {c}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      <p className="leading-relaxed">
        {sentences.map((s) => (
          <SentenceText key={s.index} sentence={s} onCite={setOpen} />
        ))}
      </p>
      <ChunkPopover chunk={chunk} onClose={() => setOpen(null)} />
      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        {verified ? (
          <>
            <Badge tone="supported">supported</Badge>
            <Badge tone="partial">partial</Badge>
            <Badge tone="unsupported">unsupported</Badge>
          </>
        ) : (
          <Badge>not verified (no evidence verification in this pipeline)</Badge>
        )}
        {response.invalid_citations.length > 0 && (
          <Badge tone="unsupported">invalid citations: {response.invalid_citations.join(', ')}</Badge>
        )}
      </div>
      {response.citations.length > 0 && (
        <Card title="Sources" className="mt-4 !p-3">
          <ul className="space-y-1 text-sm">
            {response.citations.map((c) => (
              <li key={c.number}>
                <button className="text-indigo-700 hover:underline" onClick={() => setOpen(c.number)}>
                  [{c.number}]
                </button>{' '}
                {sourceLabel(c.filename, c.page)}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

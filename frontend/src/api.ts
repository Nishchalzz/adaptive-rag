import type { DocumentRecord, ExperimentListItem, PipelineName, QueryLog, QueryResponse, StreamEvent } from './types';

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === 'string') return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) throw new Error(await errorMessage(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface QueryBody {
  question: string;
  pipeline: PipelineName;
  corpus?: string;
  overrides?: Record<string, unknown>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<{ status: string; llm_configured: boolean; nli_models: Record<string, string> }>('/api/health'),
  demoQuestions: () =>
    request<{ questions: { category: string; question: string }[] }>('/api/demo/questions'),
  query: (body: QueryBody) => request<QueryResponse>('/api/query', json(body)),
  queries: (limit = 50) => request<QueryLog[]>(`/api/queries?limit=${limit}`),
  queryById: (id: string) => request<QueryResponse>(`/api/queries/${encodeURIComponent(id)}`),
  documents: () => request<{ documents: DocumentRecord[]; total: number }>('/api/documents'),
  deleteDocument: (id: string) => request<void>(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  uploadDocument: async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/documents?wait=true', { method: 'POST', body: form });
    // 422 still carries the failed record with its error message
    const body = await res.json().catch(() => null);
    if (!res.ok && !(body && body.status === 'failed')) throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
    return body as DocumentRecord;
  },
  experiments: () => request<ExperimentListItem[]>('/api/experiments'),
  experiment: (suite: string) => request<Record<string, unknown>>(`/api/experiments/${encodeURIComponent(suite)}`),
  report: async (suite: string) => {
    const res = await fetch(`/api/experiments/${encodeURIComponent(suite)}/report`);
    return res.ok ? await res.text() : null;
  },
};

/** POST /api/query/stream and call ``onEvent`` for every server-sent event. */
export async function streamQuery(body: QueryBody, onEvent: (e: StreamEvent) => void): Promise<void> {
  const res = await fetch('/api/query/stream', json(body));
  if (!res.ok) throw new Error(await errorMessage(res));
  if (!res.body) throw new Error('Streaming is not supported by this browser');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end: number;
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      }
    }
  }
}

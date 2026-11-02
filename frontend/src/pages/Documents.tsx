import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { Badge, Card, Empty, ErrorBox, PageHeader, SkeletonCard } from '../components/ui';
import type { DocumentRecord } from '../types';

const kb = (n: number) => (n > 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

export default function Documents() {
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    api.documents().then((d) => { setDocs(d.documents); setLoaded(true); }).catch((e: Error) => { setError(e.message); setLoaded(true); });
  }, []);
  useEffect(refresh, [refresh]);

  const upload = async (files: FileList | null) => {
    if (!files) return;
    setError(null);
    for (const file of Array.from(files)) {
      setBusy(file.name);
      try {
        const record = await api.uploadDocument(file);
        if (record.status === 'failed') setError(`${file.name}: ${record.error}`);
      } catch (e) {
        setError(`${file.name}: ${(e as Error).message}`);
      }
    }
    setBusy(null);
    if (input.current) input.current.value = '';
    refresh();
  };

  const remove = async (doc: DocumentRecord) => {
    if (!confirm(`Delete ${doc.filename} and its ${doc.num_chunks} chunks?`)) return;
    try {
      await api.deleteDocument(doc.id);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Documents" subtitle="The knowledge base the system answers from. Upload PDF, DOCX, TXT or Markdown files; they are cleaned, split into chunks, embedded and indexed. A few synthetic demo documents are pre-loaded." />
      <Card title="Upload documents">
        <label
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            upload(e.dataTransfer.files);
          }}
          className="flex cursor-pointer flex-col items-center gap-1 rounded-lg border-2 border-dashed border-slate-300 p-8 text-sm text-slate-500 hover:border-indigo-500 hover:bg-indigo-50/40"
        >
          <span>{busy ? `Indexing ${busy}…` : 'Drop PDF, DOCX, TXT or Markdown files here, or click to choose'}</span>
          <input
            ref={input}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.markdown"
            className="hidden"
            disabled={busy !== null}
            onChange={(e) => upload(e.target.files)}
          />
        </label>
        {error && (
          <div className="mt-3">
            <ErrorBox message={error} />
          </div>
        )}
      </Card>

      <Card title={`Indexed documents (${docs.length})`}>
        {!loaded ? (
          <SkeletonCard lines={3} />
        ) : docs.length === 0 ? (
          <Empty>No documents yet. Drop a file above to index it.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 text-xs font-medium text-slate-600">
                <tr>
                  <th className="py-2">File</th>
                  <th>Type</th>
                  <th>Size</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>Chunking</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="border-t border-slate-200">
                    <td className="py-2 font-medium">{d.filename}</td>
                    <td>{d.file_type}</td>
                    <td>{kb(d.file_size)}</td>
                    <td>{d.num_pages ?? '–'}</td>
                    <td>{d.num_chunks}</td>
                    <td className="text-xs text-slate-500">
                      {d.chunk_size}/{d.chunk_overlap}
                    </td>
                    <td>
                      <Badge tone={d.status === 'ready' ? 'supported' : d.status === 'failed' ? 'unsupported' : 'partial'}>
                        {d.status}
                      </Badge>
                      {d.error && <div className="max-w-xs text-xs text-red-600">{d.error}</div>}
                    </td>
                    <td className="text-right">
                      <button onClick={() => remove(d)} className="text-xs text-red-600 hover:underline">
                        delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

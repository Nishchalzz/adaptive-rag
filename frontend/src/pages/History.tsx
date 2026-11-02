import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { Badge, Card, Empty, ErrorBox, PageHeader, SkeletonCard, ms } from '../components/ui';
import type { QueryLog } from '../types';

export default function History() {
  const [logs, setLogs] = useState<QueryLog[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.queries(200).then(setLogs).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  return (
    <div className="space-y-6">
    <PageHeader title="History" subtitle="Every question answered so far, with the complexity label, retrieval depth, and latency that were logged for it." />
    <Card>
      {logs === null && <SkeletonCard lines={4} />}
      {logs && logs.length === 0 && <Empty>No queries yet. Ask a question in Chat and it will be logged here.</Empty>}
      {logs && logs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs font-medium text-slate-600">
              <tr>
                <th className="py-2">Question</th>
                <th>Pipeline</th>
                <th>Complexity</th>
                <th>k</th>
                <th>Chunks</th>
                <th>Retrieval</th>
                <th>Total</th>
                <th>Tokens</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {logs.map((l) => (
                <tr key={l.id} className="border-t border-slate-200">
                  <td className="max-w-xs truncate py-2" title={l.question}>
                    {l.question}
                  </td>
                  <td>
                    <Badge tone={l.pipeline === 'proposed' ? 'violet' : 'neutral'}>{l.pipeline}</Badge>
                  </td>
                  <td>{l.complexity_label ? `${l.complexity_label} (${l.complexity_score?.toFixed(2)})` : '–'}</td>
                  <td>{l.k_selected}</td>
                  <td>{l.num_retrieved}</td>
                  <td>{ms(l.retrieval_latency_ms)}</td>
                  <td>{ms(l.total_latency_ms)}</td>
                  <td>{l.input_tokens + l.output_tokens}</td>
                  <td className="whitespace-nowrap text-xs">
                    <Link className="text-indigo-700 hover:underline" to={`/insights/${l.id}`}>
                      insights
                    </Link>{' '}
                    {l.pipeline === 'proposed' && (
                      <Link className="text-indigo-700 hover:underline" to={`/verification/${l.id}`}>
                        verification
                      </Link>
                    )}
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

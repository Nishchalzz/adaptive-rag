import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { Badge, Card, Empty, ErrorBox, ms, usd } from '../components/ui';
import type { QueryLog } from '../types';

export default function History() {
  const [logs, setLogs] = useState<QueryLog[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.queries(200).then(setLogs).catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  return (
    <Card title="Query history">
      {logs && logs.length === 0 && <Empty>No queries yet.</Empty>}
      {logs && logs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="py-2">Question</th>
                <th>Pipeline</th>
                <th>Complexity</th>
                <th>k</th>
                <th>Chunks</th>
                <th>Retrieval</th>
                <th>Total</th>
                <th>Tokens</th>
                <th>Cost</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {logs.map((l) => (
                <tr key={l.id} className="border-t border-slate-200 dark:border-slate-800">
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
                  <td>{usd(l.cost_usd)}</td>
                  <td className="whitespace-nowrap text-xs">
                    <Link className="text-sky-700 hover:underline dark:text-sky-300" to={`/insights/${l.id}`}>
                      insights
                    </Link>{' '}
                    {l.pipeline === 'proposed' && (
                      <Link className="text-sky-700 hover:underline dark:text-sky-300" to={`/verification/${l.id}`}>
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
  );
}

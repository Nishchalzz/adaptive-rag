import type { QueryResponse } from '../types';
import { ms } from './ui';

/** Small, unobtrusive tag for a saved sample run. Details (and the original numbers) are in the tooltip. */
export function ReplayNotice({
  response,
  onRunLive,
  disabled,
}: {
  response: QueryResponse;
  onRunLive?: () => void;
  disabled?: boolean;
}) {
  if (!response.cached) return null;
  const recorded = response.cached_from ? new Date(response.cached_from).toLocaleString() : 'unknown time';
  const detail = `Saved sample: a real run recorded on ${recorded}. Latency and tokens shown are from that original run (${ms(response.total_latency_ms)}).`;
  return (
    <div className="mb-2 flex items-center gap-2 text-xs text-slate-500">
      <span title={detail} className="rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-600 ring-1 ring-inset ring-slate-200">
        Saved sample
      </span>
      {onRunLive && (
        <button type="button" onClick={onRunLive} disabled={disabled} className="text-indigo-700 hover:underline disabled:opacity-50">
          Run live
        </button>
      )}
    </div>
  );
}

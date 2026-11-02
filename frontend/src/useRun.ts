import { useCallback, useRef, useState } from 'react';
import { streamQuery } from './api';
import type { PipelineName, QueryResponse, StepEvent } from './types';

export interface RunState {
  steps: StepEvent[];
  response: QueryResponse | null;
  error: string | null;
  running: boolean;
}

const MIN_REPLAY_MS = 1200;

const IDLE: RunState = { steps: [], response: null, error: null, running: false };

/** Runs one question through a pipeline via the SSE endpoint and tracks live progress. */
export function useRun() {
  const [state, setState] = useState<RunState>(IDLE);
  const last = useRef<{ question: string; pipeline: PipelineName; overrides?: Record<string, unknown> } | null>(null);

  /** ``useCache=false`` forces a live run; by default a saved real run of a demo question may be replayed. */
  const run = useCallback(
    async (question: string, pipeline: PipelineName, overrides?: Record<string, unknown>, useCache = true) => {
      last.current = { question, pipeline, overrides };
      let pending: QueryResponse | null = null;
      setState({ steps: [], response: null, error: null, running: true });
      const started = Date.now();
      try {
        await streamQuery({ question, pipeline, overrides, ...(useCache ? {} : { use_cache: false }) }, (ev) => {
          if (ev.type === 'step') setState((s) => ({ ...s, steps: [...s.steps, ev] }));
          else if (ev.type === 'result') pending = ev.response;
          else if (ev.type === 'error') setState((s) => ({ ...s, error: ev.detail }));
        });
      } catch (e) {
        setState((s) => ({ ...s, error: (e as Error).message }));
      }
      const result = pending as QueryResponse | null; // assigned inside the stream callback
      if (result) {
        // A saved sample arrives instantly; keep the loading state for a moment so the page does not jump.
        const wait = result.cached ? Math.max(0, MIN_REPLAY_MS - (Date.now() - started)) : 0;
        if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
        setState((s) => ({ ...s, response: result }));
      }
      setState((s) => ({ ...s, running: false }));
    },
    [],
  );

  /** Repeat the last question without the replay cache. */
  const runLive = useCallback(() => {
    if (last.current) void run(last.current.question, last.current.pipeline, last.current.overrides, false);
  }, [run]);

  return { state, run, runLive };
}

import { useCallback, useState } from 'react';
import { streamQuery } from './api';
import type { PipelineName, QueryResponse, StepEvent } from './types';

export interface RunState {
  steps: StepEvent[];
  response: QueryResponse | null;
  error: string | null;
  running: boolean;
}

const IDLE: RunState = { steps: [], response: null, error: null, running: false };

/** Runs one question through a pipeline via the SSE endpoint and tracks live progress. */
export function useRun() {
  const [state, setState] = useState<RunState>(IDLE);

  const run = useCallback(async (question: string, pipeline: PipelineName, overrides?: Record<string, unknown>) => {
    setState({ steps: [], response: null, error: null, running: true });
    try {
      await streamQuery({ question, pipeline, overrides }, (ev) => {
        if (ev.type === 'step') setState((s) => ({ ...s, steps: [...s.steps, ev] }));
        else if (ev.type === 'result') setState((s) => ({ ...s, response: ev.response }));
        else if (ev.type === 'error') setState((s) => ({ ...s, error: ev.detail }));
      });
    } catch (e) {
      setState((s) => ({ ...s, error: (e as Error).message }));
    }
    setState((s) => ({ ...s, running: false }));
  }, []);

  return { state, run };
}

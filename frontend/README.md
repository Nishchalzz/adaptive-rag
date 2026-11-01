# Frontend

React 18 + Vite + Tailwind CSS v4 + TypeScript. No chart library (bars are plain CSS).

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api -> http://localhost:8000
npm run build    # type-check + production build into dist/
```

Start the backend first (`uvicorn app.main:app --port 8000` in `backend/`). Light and dark themes follow the OS setting.

| Route | Page |
|---|---|
| `/` | Chat: proposed / baseline / side-by-side, demo-question chips, live pipeline steps (SSE), clickable `[n]` citations, sentences coloured by verification status, uncertainty banner |
| `/documents` | Upload (PDF, DOCX, TXT, MD), list, delete |
| `/insights[/:id]` | Complexity score and components, information needs, k and expansions, retrieval confidence, latency and token/cost breakdown, retrieved chunks |
| `/verification[/:id]` | Attempts timeline, per-claim status, method, citation status, NLI bars, evidence text, judge quote, refinement steps |
| `/dashboard` | Experiment results from `/api/experiments` (variants table, metric bar chart, figures, report) and a live baseline-vs-proposed comparison |
| `/history` | Stored query log with links to each trace |

Insights and Verification show the latest query unless a query id is in the URL or chosen in the selector.

The dashboard reads `summary.json` defensively: every numeric field under `variants.<name>` (nested objects flattened to dotted names)
becomes a metric, and PNG names listed in `summary.figures` are loaded from `/api/experiments/<suite>/figures/<name>`.

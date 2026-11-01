# Query-Adaptive Retrieval and Evidence Verification for Reliable RAG

Research prototype (7th-semester engineering project). It compares a **conventional fixed top-k RAG
baseline** with a **proposed pipeline** that

1. sets the retrieval depth from an assessment of **query complexity** (query-adaptive retrieval), and
2. **verifies every claim** of the generated answer against its cited evidence (NLI + quote-checked LLM
   judge), retrying with targeted retrieval when claims are unsupported and stating uncertainty when
   evidence stays insufficient.

Both pipelines share the vector store, embedding model, chat model and generation prompt, so a comparison
isolates the effect of the proposed components. A LangGraph feature-flag setup yields the ablation variants
(baseline / adaptive only / verification only / both).

| Document | Contents |
|---|---|
| [docs/approach.md](docs/approach.md) | The concepts and algorithms, read this first |
| [docs/architecture.md](docs/architecture.md) | System design and module responsibilities |
| [docs/experimental_methodology.md](docs/experimental_methodology.md) | Datasets, metrics, experiments, integrity rules |

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Research and architecture | done |
| 2 | Backend: ingestion, vector store, baseline, complexity classifier, adaptive retrieval, answer generation, evidence verification, LangGraph workflow | done |
| 3 | Evaluation framework, benchmark, experiment runner | done (run it to produce results; see below) |
| 4 | React + Vite + Tailwind frontend | done |
| 5 | Tests, Docker, documentation | done |

No experiment numbers are quoted in this repository's documentation. Results exist only as files written by
the experiment runner under `experiments/results/`; read them there or in the dashboard.

## Quick start

### Local

Requires Python 3.11+ and Node 20+.

```bash
cp .env.example .env                    # set OPENAI_API_KEY
cd backend
python -m venv .venv
.venv/Scripts/activate                  # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000     # API docs: http://localhost:8000/docs

# second terminal
cd frontend
npm install
npm run dev                                   # UI: http://localhost:5173
```

The first start downloads the NLI model (about 0.7 GB) in the background; `/api/health` reports its status.

### Docker

```bash
cp .env.example .env                    # set OPENAI_API_KEY
docker compose up --build
```

UI at http://localhost:5173, API at http://localhost:8000. PostgreSQL stores documents and query traces;
Chroma data lives in `backend/data/`, NLI weights in a named volume, and `./experiments` is mounted so
results are shared with the host. Without an API key the stack starts, but uploads and queries return 503.

## Demo corpus (pre-loaded)

On first start the backend indexes three **synthetic** documents from `backend/seed/` into the `default`
corpus: `institute_overview.md`, `research_labs.md` (both describing the fictional Halvorsen Institute of
Technology) and a 3-page `graduate_regulations.pdf`. They are fictional on purpose, so answers cannot come
from a language model's prior knowledge. Example questions (simple, moderate, complex, unanswerable) are
served at `GET /api/demo/questions`. Seeding happens once per data directory; deleted demo documents are not
re-added. Set `SEED_DEMO_CORPUS=false` to disable it. Regenerate the PDF with `python -m scripts.build_seed_pdf`.

## Configuration

Environment (`.env`, see `.env.example`): `OPENAI_API_KEY`, `CHAT_MODEL`, `EMBEDDING_MODEL`, `DATABASE_URL`
(defaults to SQLite; PostgreSQL via `postgresql+asyncpg://...`), `DEFAULT_CORPUS`, `JUDGE_MODEL`,
`EVAL_NLI_MODEL`, `SEED_DEMO_CORPUS`, `PRELOAD_NLI`, `EXPERIMENTS_DIR`.

Research parameters (`PipelineConfig`, overridable per request, via YAML at `PIPELINE_CONFIG_PATH`, or in
experiment configs): chunk size/overlap, baseline `top_k`, complexity weights and thresholds, per-label k
bands, retrieval-confidence weights/threshold, maximum expansions, verification thresholds, `max_retries`,
price table, and the `proposed.adaptive` / `proposed.verify` flags. Unknown keys are rejected. Every response
stores the full configuration that produced it.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, LLM configured, NLI model status |
| GET | `/api/config` | Active models and research parameters (no secrets) |
| POST | `/api/documents` | Upload PDF/DOCX/TXT/MD (`corpus` form field; `?wait=true` ingests synchronously) |
| GET | `/api/documents`, `/api/documents/{id}` | List / inspect documents |
| DELETE | `/api/documents/{id}` | Delete a document and its chunks |
| POST | `/api/query` | Answer with `pipeline`: `baseline` or `proposed`; returns the full trace |
| POST | `/api/query/stream` | Same, as server-sent events: one `step` per graph node, then `result` |
| GET | `/api/queries`, `/api/queries/{id}` | Query log and stored traces |
| GET | `/api/demo/questions` | Example questions for the demo corpus |
| GET | `/api/experiments` | List experiment suites with results |
| GET | `/api/experiments/{suite}` | Summary JSON of a suite |
| GET | `/api/experiments/{suite}/report` | Markdown report |
| GET | `/api/experiments/{suite}/figures/{name}` | A generated figure (PNG) |

Ablation variants per request: `"overrides": {"proposed": {"adaptive": false}}` (verification only),
`{"proposed": {"verify": false}}` (adaptive only), or both false (equivalent to the baseline).

## Tests

```bash
cd backend
pytest              # offline unit + integration tests (LLM and NLI replaced by test doubles)
pytest -m live      # smoke test against the real OpenAI API (needs OPENAI_API_KEY)
```

## Experiments

Experiments are run from the command line so they are reproducible, and the dashboard only reads their
output:

```bash
cd backend
python -m scripts.run_experiments --help
```

The runner writes a configuration snapshot, per-question results, a `summary.json`, `report.md` and figures
to `experiments/results/<suite>/`. Dataset preparation, metrics, statistics and the limits of automatic
evaluation are described in [docs/experimental_methodology.md](docs/experimental_methodology.md). Running
against the real API costs money; check the pilot cost before running a full suite.

## Layout

```
backend/app/
  core/          settings and research config, LLM factory, structured output, token/cost tracking, wiring
  api/           FastAPI routers (documents, query, experiments, system)
  schemas/       Pydantic models
  ingestion/     loaders, cleaning, chunking, document lifecycle
  vectorstore/   VectorStore protocol + ChromaDB
  retrieval/     dense, fixed-k baseline, complexity classifier, adaptive retriever
  generation/    shared prompt, generator, citation parsing
  verification/  claim extraction, evidence matching, NLI, verifier, finalisation
  workflows/     baseline pipeline, LangGraph proposed pipeline, query service
  evaluation/    metrics and benchmark runner
  database/      SQLAlchemy models and repositories
backend/seed/    demo documents      backend/tests/   unit, integration, live
backend/scripts/ seed PDF builder and experiment CLI
frontend/        React + Vite + Tailwind UI
experiments/     datasets, configs, results, figures
docs/            approach, architecture, methodology
```

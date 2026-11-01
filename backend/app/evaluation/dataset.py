"""Benchmark construction from MuSiQue (Trivedi et al., 2022).

Groups (complexity proxy = number of reasoning hops in the human-written decomposition):
  simple    a single hop of a decomposition, with earlier answers substituted for ``#k`` references, so it
            is a standalone natural-language question; gold evidence = that hop's supporting paragraph
  moderate  2-hop questions; gold evidence = the 2 supporting paragraphs
  complex   3- and 4-hop questions; gold evidence = the 3-4 supporting paragraphs

All paragraphs (supporting + distractors) of every sampled source question form one shared corpus.
Gold labels come from MuSiQue's human annotation; nothing is labelled by a model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from app.schemas.documents import Chunk
from app.vectorstore.store import VectorStore

GROUPS = ("simple", "moderate", "complex")
CORPUS_NAME = "musique_eval"
_PLACEHOLDER = re.compile(r"#(\d+)")


def chunk_id_for(title: str, text: str) -> str:
    return "mq-" + hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:12]


def _resolve_step(steps: list[dict[str, Any]], index: int) -> str | None:
    """Standalone form of decomposition step ``index`` or None if it is a relation template."""
    question = steps[index]["question"]
    if ">>" in question:
        return None

    def replace(match: re.Match[str]) -> str:
        ref = int(match.group(1)) - 1
        return steps[ref]["answer"] if 0 <= ref < len(steps) else match.group(0)

    resolved = _PLACEHOLDER.sub(replace, question).strip()
    if "#" in resolved or len(resolved.split()) < 5 or steps[index]["answer"].lower() in resolved.lower():
        return None
    resolved = resolved[0].upper() + resolved[1:]
    return resolved if resolved.endswith("?") else resolved + "?"


def _record(row: dict[str, Any], *, group: str, question: str, answer: str, aliases: list[str], gold_idx: list[int], hops: int) -> dict[str, Any]:
    paragraphs = {p["idx"]: p for p in row["paragraphs"]}
    return {
        "group": group,
        "source_id": row["id"],
        "question": question,
        "answer": answer,
        "aliases": aliases,
        "hops": hops,
        "gold_chunk_ids": sorted({chunk_id_for(paragraphs[i]["title"], paragraphs[i]["paragraph_text"]) for i in gold_idx}),
        "gold_titles": sorted({paragraphs[i]["title"] for i in gold_idx}),
    }


def build_dataset(raw_path: Path, *, seed: int = 13, n_test: int = 12, n_dev: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (dataset, corpus_paragraphs). Deterministic for a given seed and raw file."""
    rng = random.Random(seed)
    with raw_path.open(encoding="utf-8") as fh:
        rows = sorted((r for r in map(json.loads, fh) if r["answerable"]), key=lambda r: r["id"])
    need = n_test + n_dev
    two_hop = [r for r in rows if r["id"].startswith("2hop")]
    deep = [r for r in rows if r["id"].startswith(("3hop", "4hop"))]
    moderate_rows = rng.sample(two_hop, need)
    complex_rows = rng.sample(deep, need)
    used = {r["id"] for r in moderate_rows + complex_rows}

    simple: list[tuple[dict[str, Any], int, str]] = []
    pool = [r for r in rows if r["id"] not in used]
    rng.shuffle(pool)
    for row in pool:
        steps = row["question_decomposition"]
        options = [(i, q) for i in range(len(steps)) if (q := _resolve_step(steps, i)) is not None]
        if options:
            i, q = rng.choice(options)
            simple.append((row, i, q))
        if len(simple) == need:
            break

    items: list[dict[str, Any]] = []
    for row, i, question in simple:
        step = row["question_decomposition"][i]
        items.append(_record(row, group="simple", question=question, answer=step["answer"], aliases=[], gold_idx=[step["paragraph_support_idx"]], hops=1))
        used.add(row["id"])
    for group, sample in (("moderate", moderate_rows), ("complex", complex_rows)):
        for row in sample:
            gold = [p["idx"] for p in row["paragraphs"] if p["is_supporting"]]
            items.append(_record(row, group=group, question=row["question"], answer=row["answer"], aliases=row["answer_aliases"],
                                 gold_idx=gold, hops=len(row["question_decomposition"])))

    for group in GROUPS:  # stratified dev/test split
        members = [it for it in items if it["group"] == group]
        rng.shuffle(members)
        for n, item in enumerate(members):
            item["split"] = "dev" if n < n_dev else "test"
    for n, item in enumerate(sorted(items, key=lambda it: (GROUPS.index(it["group"]), it["source_id"], it["question"]))):
        item["qid"] = f"{item['group'][:3]}{n:03d}"

    source_ids = {it["source_id"] for it in items}
    corpus: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["id"] in source_ids:
            for p in row["paragraphs"]:
                cid = chunk_id_for(p["title"], p["paragraph_text"])
                corpus.setdefault(cid, {"chunk_id": cid, "title": p["title"], "text": p["paragraph_text"]})

    questions = sorted(items, key=lambda it: it["qid"])
    dataset = {
        "name": "musique_eval",
        "source": "MuSiQue v1.0 dev (answerable), https://github.com/StonyBrookNLP/musique",
        "seed": seed,
        "n_test_per_group": n_test,
        "n_dev_per_group": n_dev,
        "corpus": CORPUS_NAME,
        "corpus_size": len(corpus),
        "counts": {f"{g}/{s}": sum(1 for it in questions if it["group"] == g and it["split"] == s) for g in GROUPS for s in ("dev", "test")},
        "hops": dict(Counter(f"{it['group']}:{it['hops']}" for it in questions)),
        "questions": questions,
    }
    return dataset, sorted(corpus.values(), key=lambda c: c["chunk_id"])


def load_dataset(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


async def index_corpus(corpus_path: Path, store: VectorStore, embeddings: Embeddings, corpus: str = CORPUS_NAME, batch: int = 128) -> int:
    """Embed and store the benchmark paragraphs (one chunk per paragraph). Returns chunks added."""
    paragraphs = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if store.count(corpus) >= len(paragraphs):
        return 0
    chunks = [
        Chunk(chunk_id=p["chunk_id"], doc_id=p["chunk_id"], corpus=corpus, filename=p["title"], page=None, chunk_index=0, text=p["text"])
        for p in paragraphs
    ]
    for start in range(0, len(chunks), batch):
        part: Sequence[Chunk] = chunks[start : start + batch]
        vectors = await embeddings.aembed_documents([c.text for c in part])
        store.add_chunks(part, vectors)
    return len(chunks)

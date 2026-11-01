"""Candidate evidence selection for claims.

Chunks are split into overlapping sentence windows. Embedding similarity ranks the windows for each
claim and only decides *which* windows are tested; whether a window supports a claim is decided by
NLI (and, when NLI is ambiguous, an LLM judge) in ``verifier.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.usage import UsageTracker
from app.generation.citations import split_sentences
from app.schemas.query import RetrievedChunk

def sentence_windows(text: str, size: int) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i : i + size]) for i in range(len(sentences) - size + 1)]


@dataclass(frozen=True)
class EvidenceWindow:
    chunk: RetrievedChunk
    text: str
    similarity: float


class EvidenceIndex:
    def __init__(self, windows: list[tuple[RetrievedChunk, str]], window_vectors: np.ndarray, claim_vectors: np.ndarray) -> None:
        self._windows = windows
        self._window_vectors = window_vectors
        self._claim_vectors = claim_vectors

    def candidates(self, claim_index: int, citation_numbers: Iterable[int] | None, top_m: int) -> list[EvidenceWindow]:
        """Best ``top_m`` windows for a claim, restricted to the given passages (None = all passages)."""
        allowed = None if citation_numbers is None else set(citation_numbers)
        if not self._windows:
            return []
        sims = self._window_vectors @ self._claim_vectors[claim_index]
        ranked = []
        for i in np.argsort(-sims):
            chunk, text = self._windows[int(i)]
            if allowed is not None and chunk.citation_number not in allowed:
                continue
            ranked.append(EvidenceWindow(chunk=chunk, text=text, similarity=float(sims[int(i)])))
            if len(ranked) >= top_m:
                break
        return ranked


def _normalised(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


class EvidenceMatcher:
    def __init__(self, embeddings: Embeddings, embedding_model: str) -> None:
        self._embeddings = embeddings
        self._embedding_model = embedding_model

    async def build_index(
        self,
        chunks: Sequence[RetrievedChunk],
        claims: Sequence[str],
        *,
        window_sentences: int,
        cache: dict[str, list[float]],
        usage: UsageTracker,
    ) -> EvidenceIndex:
        """Embed every window of every chunk plus the claims in one request; ``cache`` avoids
        re-embedding windows across verification attempts of the same query."""
        windows = [(chunk, window) for chunk in chunks for window in sentence_windows(chunk.text, window_sentences)]
        texts = [w for _, w in windows] + list(claims)
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            vectors = await self._embeddings.aembed_documents(missing)
            cache.update(zip(missing, vectors, strict=True))
            usage.record_embedding("verification_embedding", self._embedding_model, missing)
        return EvidenceIndex(
            windows,
            _normalised([cache[w] for _, w in windows]),
            _normalised([cache[c] for c in claims]),
        )

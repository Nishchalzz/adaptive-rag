"""Per-question metrics. All are automatic; none is human-verified.

Retrieval      gold = MuSiQue human-annotated supporting paragraphs (this part is human-labelled ground truth)
Answer         EM / token-F1 / containment against gold answer + aliases; correctness and relevance from the
               independent judge LLM (``judges.py``)
Evidence       measured with an NLI model that is NOT the pipeline's verifier (different checkpoint, different
               threshold, sentence-level claims split by rules instead of the pipeline's LLM extractor)
Efficiency     taken from the QueryResponse (tokens and cost from provider usage, latencies measured in-process)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Sequence

from app.generation.citations import citation_numbers, split_sentences, strip_citations
from app.schemas.query import QueryResponse
from app.verification.claim_extractor import is_abstention
from app.verification.nli import EntailmentModel

CUTOFFS = (1, 3, 5, 10)
EVAL_SUPPORT_THRESHOLD = 0.5  # evaluator entailment threshold (the pipeline's verifier uses 0.75)


# ------------------------------------------------------------------ retrieval
def retrieval_metrics(retrieved: Sequence[str], gold: set[str]) -> dict[str, float]:
    """P/R at the system's own depth and at fixed cutoffs, MRR and binary nDCG@10.

    Precision@c divides by c even when fewer than c chunks were retrieved (standard definition), so a
    system that retrieves few chunks is not rewarded for short lists at the fixed cutoffs.
    """
    hits = [1 if cid in gold else 0 for cid in retrieved]
    total = len(gold)
    out = {
        "p_at_k": sum(hits) / len(hits) if hits else 0.0,
        "r_at_k": sum(hits) / total if total else 0.0,
    }
    for c in CUTOFFS:
        out[f"p_at_{c}"] = sum(hits[:c]) / c
        out[f"r_at_{c}"] = sum(hits[:c]) / total if total else 0.0
    out["mrr"] = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(total, 10)))
    out["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return out


# ------------------------------------------------------------------ answer text
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def answer_text_metrics(answer: str, gold: str, aliases: Sequence[str]) -> dict[str, float]:
    prediction = strip_citations(answer)
    golds = [gold, *aliases]
    normalized = normalize_answer(prediction)
    return {
        "em": float(any(normalize_answer(g) == normalized for g in golds)),
        "f1": max(_f1(prediction, g) for g in golds),
        "contains": float(any(f" {normalize_answer(g)} " in f" {normalized} " for g in golds if normalize_answer(g))),
    }


# ------------------------------------------------------------------ evidence (independent NLI)
class EvidenceScorer:
    """Claim-level support and ALCE-style citation precision/recall with an independent NLI model."""

    def __init__(self, nli: EntailmentModel, threshold: float = EVAL_SUPPORT_THRESHOLD) -> None:
        self._nli = nli
        self._threshold = threshold
        self._cache: dict[tuple[str, str], float] = {}

    def _entails(self, premises: list[str], claim: str) -> list[float]:
        todo = [p for p in dict.fromkeys(premises) if (p, claim) not in self._cache]
        if todo:
            for premise, result in zip(todo, self._nli.score([(p, claim) for p in todo]), strict=True):
                self._cache[(premise, claim)] = result.entailment
        return [self._cache[(p, claim)] for p in premises]

    def score(self, response: QueryResponse) -> dict[str, float | None]:
        chunks = {c.citation_number: c.text for c in response.retrieval.chunks}
        claims = [s for s in split_sentences(response.answer) if strip_citations(s) and not is_abstention(s)]
        supported = recall_hits = rel_citations = n_citations = 0
        for sentence in claims:
            claim = strip_citations(sentence)
            numbers = citation_numbers(sentence)
            valid = [n for n in numbers if n in chunks]
            n_citations += len(numbers)
            cited_text = "\n".join(chunks[n] for n in valid)
            premises = [*chunks.values()] + ([cited_text] if len(valid) > 1 else [])
            context_ok = any(e >= self._threshold for e in self._entails(premises, claim)) if premises else False
            supported += context_ok

            cited_ok = bool(valid) and self._entails([cited_text], claim)[0] >= self._threshold
            recall_hits += cited_ok
            for n in valid:  # ALCE: a citation is relevant if it alone entails the claim, or is needed for the set to
                alone = self._entails([chunks[n]], claim)[0] >= self._threshold
                others = [chunks[m] for m in valid if m != n]
                without = bool(others) and self._entails(["\n".join(others)], claim)[0] >= self._threshold
                rel_citations += alone or (cited_ok and not without)
        n = len(claims)
        return {
            "num_claims": float(n),
            "claim_support_rate": supported / n if n else None,
            "unsupported_rate": 1 - supported / n if n else None,
            "citation_recall": recall_hits / n if n else None,
            "citation_precision": rel_citations / n_citations if n_citations else None,
        }


# ------------------------------------------------------------------ efficiency
def efficiency_metrics(response: QueryResponse) -> dict[str, float | None]:
    return {
        "num_chunks": float(response.retrieval.num_retrieved),
        "k_selected": float(response.retrieval.k_requested),
        "retrieval_latency_ms": response.retrieval.latency_ms,
        "total_latency_ms": response.total_latency_ms,
        "input_tokens": float(response.usage.input_tokens),
        "output_tokens": float(response.usage.output_tokens),
        "total_tokens": float(response.usage.input_tokens + response.usage.output_tokens),
        "cost_usd": response.usage.cost_usd,
        "num_attempts": float(max(len(response.attempts), 1)),
    }

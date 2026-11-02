"""Natural Language Inference scoring with Hugging Face sequence-classification models.

Label order differs between NLI checkpoints, so the entailment/neutral/contradiction indices are read
from each model's ``id2label`` instead of being assumed.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float


class EntailmentModel(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        """Score (premise, hypothesis) pairs."""
        ...


class HFEntailmentModel:
    def __init__(self, name: str, device: str = "cpu", batch_size: int = 8, max_length: int = 512) -> None:
        self.name = name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._lock = threading.Lock()
        self._loaded = False
        self.status: Literal["not_loaded", "loading", "ready", "error"] = "not_loaded"
        self.error: str | None = None

    def load(self) -> None:
        first_load = False
        with self._lock:
            if self._loaded:
                return
            self.status = "loading"
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._torch = torch
                self._tokenizer = AutoTokenizer.from_pretrained(self.name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.name).to(self._device).eval()
                labels = {label.lower(): index for index, label in self._model.config.id2label.items()}
                self._index = {key: labels[key] for key in ("entailment", "neutral", "contradiction")}
            except Exception as exc:
                self.status, self.error = "error", f"{type(exc).__name__}: {exc}"
                raise
            self._loaded = True
            self.status = "ready"
            logger.info("Loaded NLI model %s on %s", self.name, self._device)
            first_load = True
        if first_load:  # outside the lock: score() takes it
            self._warm_up()

    def _warm_up(self) -> None:
        """Run one tiny inference so lazy initialisation cost is paid at startup, not by the first query."""
        try:
            self.score([("The tower is in Paris.", "The tower is located in Paris.")])
        except Exception:  # noqa: BLE001 - warm-up is an optimisation only
            logger.warning("NLI warm-up failed", exc_info=True)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIResult]:
        if not pairs:
            return []
        self.load()
        # Padding makes every pair in a batch as expensive as the longest one, so pairs are processed in
        # order of length (short windows are not padded up to whole passages) and put back afterwards.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        results: list[NLIResult | None] = [None] * len(pairs)
        with self._lock, self._torch.inference_mode():  # one inference at a time; torch parallelises internally
            for start in range(0, len(order), self._batch_size):
                indices = order[start : start + self._batch_size]
                encoded = self._tokenizer(
                    [pairs[i][0] for i in indices],
                    [pairs[i][1] for i in indices],
                    truncation="only_first",
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                probs = self._model(**encoded).logits.softmax(dim=-1).tolist()
                for i, row in zip(indices, probs, strict=True):
                    results[i] = NLIResult(
                        entailment=row[self._index["entailment"]],
                        neutral=row[self._index["neutral"]],
                        contradiction=row[self._index["contradiction"]],
                    )
        return results  # type: ignore[return-value]


class NLIRegistry:
    """Loads each NLI checkpoint once per process; models are shared by all requests."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._models: dict[str, EntailmentModel] = {}
        self._lock = threading.Lock()

    def register(self, model: EntailmentModel) -> None:
        self._models[model.name] = model

    def get(self, name: str) -> EntailmentModel:
        with self._lock:
            if name not in self._models:
                self._models[name] = HFEntailmentModel(name, device=self._device)
            return self._models[name]

    def preload(self, name: str) -> threading.Thread:
        model = self.get(name)

        def _load() -> None:
            try:
                if isinstance(model, HFEntailmentModel):
                    model.load()
            except Exception:  # noqa: BLE001 - reported through status; the request path retries the load
                logger.exception("Background load of NLI model %s failed", name)

        thread = threading.Thread(target=_load, name=f"nli-preload-{name}", daemon=True)
        thread.start()
        return thread

    def status(self) -> dict[str, str]:
        return {name: getattr(m, "status", "ready") for name, m in self._models.items()}

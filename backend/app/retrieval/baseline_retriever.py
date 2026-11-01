from app.retrieval.dense import DenseRetriever, RetrievalResult


class FixedKRetriever:
    """Conventional RAG retrieval: the top-k chunks for the question, with k fixed for every query."""

    strategy = "fixed_top_k"

    def __init__(self, dense: DenseRetriever, k: int) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self._dense = dense
        self.k = k

    async def retrieve(self, corpus: str, question: str) -> RetrievalResult:
        return await self._dense.retrieve(corpus, question, self.k)

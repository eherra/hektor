from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Dict
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except Exception:
    _BM25_AVAILABLE = False


class HybridSearch:
    """
    Lightweight hybrid: BM25 + vector with min-max normalization + linear blend.
    If rank_bm25 isn't installed, this will raise at construction time.
    """

    def __init__(
        self,
        vector_index,
        embed_fn: Callable[[Sequence[str]], np.ndarray],
        tokenizer,
    ) -> None:
        if not _BM25_AVAILABLE:
            raise ImportError(
                "Hybrid search requires rank-bm25. Install with:\n  pip install rank-bm25"
            )
        self.vec = vector_index
        self.embed_fn = embed_fn
        self.tokenizer = tokenizer or (lambda s: s.lower().split())

        self._texts: List[str] = []
        self._ids: List[int] = []
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def add_documents(self, texts: Sequence[str], ids: Optional[Sequence[int]] = None) -> None:
        texts = list(texts)
        if not texts:
            return
        if ids is None:
            start = len(self._ids)
            ids = list(range(start, start + len(texts)))
        ids = list(map(int, ids))
        if len(ids) != len(texts):
            raise ValueError("ids length must match number of texts")

        toks = [self.tokenizer(t) for t in texts]
        self._texts.extend(texts)
        self._ids.extend(ids)
        self._corpus_tokens.extend(toks)
        self._bm25 = BM25Okapi(self._corpus_tokens)

    # ---------- search ----------
    def search_hybrid(
        self,
        query_text: str,
        k: int = 5,
        alpha: float = 0.5,
        vector_candidates: int = 50,
        bm25_candidates: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._bm25 is None or not self._ids:
            raise ValueError("No documents indexed yet.")

        # vector side
        q_emb = self.embed_fn([query_text]).astype("float32")
        v_scores, v_ids = self.vec.search_top_k(q_emb, k=min(vector_candidates, len(self._ids)))
        v_scores, v_ids = v_scores[0], v_ids[0]
        dense = {int(i): float(s) for i, s in zip(v_ids, v_scores) if int(i) != -1}

        # sparse side
        q_tok = self.tokenizer(query_text)
        bm25_scores = self._bm25.get_scores(q_tok)
        top_b_idx = np.argsort(bm25_scores)[::-1][:min(bm25_candidates, len(self._ids))]
        sparse = {self._ids[int(j)]: float(bm25_scores[int(j)]) for j in top_b_idx}

        def _minmax(d: Dict[int, float]) -> Dict[int, float]:
            if not d:
                return {}
            arr = np.array(list(d.values()), dtype="float32")
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo + 1e-12:
                return {k: 0.0 for k in d}
            return {k: (float(v) - lo) / (hi - lo) for k, v in d.items()}

        # convert to similarity if needed
        higher_is_better = (self.vec.metric in {"ip", "cosine"})
        if not higher_is_better:
            dense = {k: -v for k, v in dense.items()}

        dense_n = _minmax(dense)
        sparse_n = _minmax(sparse)

        all_ids = set(dense_n) | set(sparse_n)
        blended = {i: alpha * dense_n.get(i, 0.0) + (1 - alpha) * sparse_n.get(i, 0.0) for i in all_ids}
        if not blended:
            return np.asarray([[]], dtype="float32"), np.asarray([[]], dtype="int64")

        top_ids = sorted(blended, key=blended.get, reverse=True)[:k]
        top_scores = [blended[i] for i in top_ids]
        return np.asarray([top_scores], dtype="float32"), np.asarray([top_ids], dtype="int64")

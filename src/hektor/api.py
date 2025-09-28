from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union, Any, Dict
import json
import os

from .tokenizers import resolve_tokenizer
from .embeddings import create as create_embedder
from .index import HektorIndex
from .hybrid import HybridSearch

QueryLike = Union[str, Sequence[str]]

class HektorSearch:
    """
    Dead-simple vector (and optional hybrid) search with explicit model selection.

    Quickstart:
        s = HektorSearch(model="sentence-transformers/all-MiniLM-L6-v2", hybrid=True)
        s.from_texts(["hello world", "foo bar"])
        s.search("hello", k=3)
        s.save("my.index")

        t = HektorSearch.load("my.index")
        t.search(["foo", "world"], k=5)

    Parameters
    ----------
    model : str
        Embedding model identifier passed to your embedding factory.
    embedding : dict | None
        Extra kwargs for the embedder, merged with {"model": model}.
        e.g., {"device": "cpu", "batch_size": 64}
    dim : int | None
        Optional manual override of the embedding dimension.
        If None, inferred from the embedder (preferred).
    metric : {'cosine', 'ip', 'l2'}
    normalize : bool
    hybrid : bool
    alpha : float
    tokenizer : None | str | callable
    """

    def __init__(
        self,
        *,
        model: str = "intfloat/multilingual-e5-small",
        embedding: Optional[Dict[str, Any]] = None,
        dim: Optional[int] = None,
        metric: str = "cosine",
        normalize: bool = True,
        hybrid: bool = True,
        alpha: float = 0.5,
        tokenizer=None,
    ) -> None:
        # Core cfg
        self.model = model
        self._embedding_cfg: Dict[str, Any] = {"model": model}
        if embedding:
            self._embedding_cfg.update({k: v for k, v in embedding.items() if k != "model"})
        self._dim_override = int(dim) if dim is not None else None

        self.metric = metric
        self.normalize = normalize
        self.hybrid_enabled = bool(hybrid)
        self.alpha = float(alpha)

        self._embedder = None
        self._index: Optional[HektorIndex] = None
        self._hybrid: Optional[HybridSearch] = None
        self._texts: List[str] = []
        self._ids: List[int] = []

        if tokenizer is None or isinstance(tokenizer, str):
            self._tokenizer_name = tokenizer or "simple"
            self._tokenizer = resolve_tokenizer(self._tokenizer_name)
        else:
            self._tokenizer_name = getattr(tokenizer, "__name__", "custom")
            self._tokenizer = tokenizer

    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder, _ = create_embedder(dict(self._embedding_cfg))
        return self._embedder

    def _resolve_dim(self) -> int:
        if self._dim_override and self._dim_override > 0:
            return self._dim_override
        emb = self._ensure_embedder()
        d = int(getattr(emb, "dim", 0) or 0)
        if d <= 0:
            raise ValueError(
                "Cannot determine embedding dimension. Provide dim=... or ensure the embedder exposes `.dim`."
            )
        return d

    def _ensure_index(self, dim: Optional[int] = None) -> HektorIndex:
        if self._index is not None:
            return self._index
        use_dim = int(dim) if dim else self._resolve_dim()
        self._index = HektorIndex(dim=use_dim, metric=self.metric, config={
            "dim": use_dim, "metric": self.metric, "use_id_map": True, "normalize": self.normalize
        })
        return self._index

    def _maybe_init_hybrid(self):
        if not self.hybrid_enabled:
            return None
        if self._hybrid is None:
            idx = self._ensure_index()
            emb = self._ensure_embedder()
            self._hybrid = HybridSearch(
                vector_index=idx,
                embed_fn=emb.embed,
                tokenizer=self._tokenizer,
            )
            if self._texts:
                self._hybrid.add_documents(self._texts, ids=self._ids)
        return self._hybrid

    def from_texts(self, texts: Sequence[str], ids: Optional[Sequence[int]] = None) -> "HektorSearch":
        """Create a fresh index from texts. Overwrites any existing in-memory state."""
        self._index = None
        self._hybrid = None
        self._texts = []
        self._ids = []
        return self.add(texts, ids=ids)

    def add(self, texts: Sequence[str], ids: Optional[Sequence[int]] = None) -> "HektorSearch":
        """Append documents to the index."""
        texts = list(texts)
        if not texts:
            return self

        if ids is None:
            start = len(self._ids)
            ids = list(range(start, start + len(texts)))
        ids = list(map(int, ids))
        if len(ids) != len(texts):
            raise ValueError("len(ids) must match len(texts)")

        emb = self._ensure_embedder()
        vecs = emb.embed(texts).astype("float32")
        idx = self._ensure_index(dim=vecs.shape[1])
        idx.add_embeddings(vecs, ids=ids)

        self._texts.extend(texts)
        self._ids.extend(ids)

        if self.hybrid_enabled:
            self._maybe_init_hybrid()
            self._hybrid.add_documents(texts, ids=ids)

        return self

    def save(self, index_path: str) -> None:
        """Save the vector index and a meta file next to it."""
        if self._index is None:
            raise ValueError("Nothing to save. Did you call from_texts/add()?")

        self._index.save_index(index_path)
        meta = {
            "model": self.model,
            "embedding": self._embedding_cfg,   # full embedding cfg (without index-related params)
            "dim": self._index.dim,
            # vector index config
            "metric": self.metric,
            "normalize": self.normalize,
            # hybrid
            "hybrid_enabled": self.hybrid_enabled,
            "alpha": self.alpha,
            # tokenizer
            "tokenizer": self._tokenizer_name,
            # corpus (only for demo/small use; for big corpora you'd store externally)
            "ids": self._ids,
            "texts": self._texts,
        }
        with open(index_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_path: str) -> "HektorSearch":
        """Load from disk and return a ready-to-search instance."""
        meta_path = index_path + ".meta.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta file: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        inst = cls(
            model=meta.get("model", "intfloat/multilingual-e5-small"),
            embedding=meta.get("embedding", None),
            dim=meta.get("dim", None),  # safe to pass; _resolve_dim will fall back if needed
            metric=meta.get("metric", "cosine"),
            normalize=bool(meta.get("normalize", True)),
            hybrid=bool(meta.get("hybrid_enabled", True)),
            alpha=float(meta.get("alpha", 0.5)),
            tokenizer=meta.get("tokenizer", "simple"),
        )

        dim = int(meta.get("dim", 0) or 0) or None
        inst._index = HektorIndex.load_index(
            index_path,
            dim=dim,
            metric=inst.metric,
            config={"use_id_map": True, "normalize": inst.normalize, "metric": inst.metric, "dim": int(dim or 0)},
        )
        inst._texts = list(meta.get("texts", []))
        inst._ids = list(map(int, meta.get("ids", [])))

        if inst.hybrid_enabled and inst._texts:
            inst._maybe_init_hybrid()
        return inst

    # ----------------- search -----------------
    def search(self, queries: QueryLike, k: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Returns list of results per query: [[(id, score), ...], ...]
        Accepts a single string or a list of strings.
        """
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return [[]]

        idx = self._ensure_index()
        emb = self._ensure_embedder()

        results: List[List[Tuple[int, float]]] = []
        if self.hybrid_enabled and self._texts:
            hyb = self._maybe_init_hybrid()
            for q in queries:
                sc, ii = hyb.search_hybrid(q, k=k, alpha=self.alpha)
                results.append([(int(i), float(s)) for i, s in zip(ii[0], sc[0])])
        else:
            q_emb = emb.embed(list(queries)).astype("float32")
            sc, ii = idx.search_top_k(q_emb, k=k)
            for row_s, row_i in zip(sc, ii):
                results.append([(int(i), float(s)) for i, s in zip(row_i, row_s)])
        return results

    def close(self):
        if self._embedder:
            self._embedder.close()
            self._embedder = None

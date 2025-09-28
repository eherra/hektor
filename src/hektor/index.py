from __future__ import annotations
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import faiss


class HektorIndex:
    """
    Tiny FAISS wrapper.

    Args:
        dim: Embedding dimensionality (required unless provided via config["dim"]).
        metric: "l2", "ip", or "cosine".
        config: dict or JSON string with optional keys:
            - "dim": int
            - "metric": "l2" | "ip" | "cosine"
            - "use_id_map": bool (default True)
            - "normalize": bool (force L2-normalization even if not cosine)
    """

    def __init__(
        self,
        dim: Optional[int],
        metric: str = "l2",
        config: Optional[Dict[str, Any] | str] = None,
    ) -> None:
        # parse config
        if isinstance(config, str):
            config = json.loads(config)
        config = config or {}

        self.dim = int(config.get("dim", dim if dim is not None else -1))
        if self.dim <= 0:
            raise ValueError("`dim` must be provided (arg or config['dim']).")

        self.metric = str(config.get("metric", metric)).lower()
        if self.metric not in {"l2", "ip", "cosine"}:
            raise ValueError("metric must be one of {'l2','ip','cosine'}")

        use_id_map = bool(config.get("use_id_map", True))
        force_normalize = bool(config.get("normalize", False))

        # choose base index
        if self.metric == "l2":
            base = faiss.IndexFlatL2(self.dim)
        else:  # 'ip' or 'cosine'
            base = faiss.IndexFlatIP(self.dim)

        self._normalize = (self.metric == "cosine") or force_normalize
        self.index = faiss.IndexIDMap2(base) if use_id_map else base

    # ---------------- internal ----------------
    def _prepare_embeddings(self, x: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(x, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected shape (*, {self.dim}), got {arr.shape}")
        if self._normalize:
            faiss.normalize_L2(arr)
        return arr

    # ---------------- new, descriptive API ----------------
    def add_embeddings(self, embeddings, ids: Optional[Sequence[int]] = None) -> None:
        x = self._prepare_embeddings(embeddings)
        if ids is None:
            if isinstance(self.index, faiss.IndexIDMap2):
                start = int(self.index.ntotal)
                ids = np.arange(start, start + x.shape[0], dtype="int64")
                self.index.add_with_ids(x, ids)
            else:
                self.index.add(x)
        else:
            if not isinstance(self.index, faiss.IndexIDMap2):
                raise ValueError("Custom IDs require use_id_map=True.")
            ids = np.asarray(ids, dtype="int64").reshape(-1)
            if ids.shape[0] != x.shape[0]:
                raise ValueError("ids length must match number of embeddings.")
            self.index.add_with_ids(x, ids)

    def search_top_k(self, query_embeddings, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = self._prepare_embeddings(query_embeddings)
        if k <= 0:
            raise ValueError("k must be positive.")
        return self.index.search(q, int(k))

    def save_index(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load_index(
        cls,
        path: str,
        dim: Optional[int] = None,
        metric: str = "l2",
        config: Optional[Dict[str, Any] | str] = None,
    ) -> "HektorIndex":
        index = faiss.read_index(path)
        inferred = getattr(index, "d", None)
        if dim is None:
            if inferred is None:
                raise ValueError("Could not infer dimension; pass dim explicitly.")
            dim = int(inferred)
        obj = cls(dim=dim, metric=metric, config=config)
        obj.index = index
        return obj

    # convenience
    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def __len__(self) -> int:
        return self.size

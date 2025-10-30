from __future__ import annotations
from typing import Sequence
import gc
import numpy as np
from .base import Embedder
from . import register

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

@register("huggingface")
@register("sentence_transformers")
class HFEmbedder(Embedder):
    """
    Sentence-Transformers based embedder.

    Config (all optional):
      - model / model_name: str (default: intfloat/multilingual-e5-small)
      - device: str | None ("cpu", "cuda", "mps", "cuda:0", etc.)
      - normalize: bool (default: True) -> L2-normalize embeddings
      - batch_size: int (default: 32)
      - prefix: str (default: "") -> prepended to every text (useful for E5, e.g. "query: " / "passage: ")
    """

    def __init__(self, cfg: dict) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Hugging Face mode requires `sentence-transformers`.\n"
                "Install with: pip install -U sentence-transformers"
            ) from e

        # Backward compatible: support both "model" and "model_name"
        model_name = cfg.get("model") or cfg.get("model_name") or DEFAULT_MODEL
        device = cfg.get("device", None)
        self._normalize = bool(cfg.get("normalize", True))
        self._batch_size = int(cfg.get("batch_size", 32))
        self._prefix = str(cfg.get("prefix", ""))

        self.model = SentenceTransformer(model_name, device=device)
        # Prefer library API; fall back to probing a test vector if needed
        try:
            self.dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Last-resort: compute one vector to infer dim
            test_vec = self.model.encode(["test"], normalize_embeddings=self._normalize, convert_to_numpy=True)
            self.dim = int(test_vec.shape[1])

        self._device_str = str(device) if device is not None else getattr(self.model, "device", "cpu")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        # Optional prefix (handy for E5: set prefix="query: " or "passage: ")
        if self._prefix:
            texts = [self._prefix + t for t in texts]

        vecs = self.model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        """
        Release model resources. On CUDA, also try to free GPU memory.
        """
        try:
            import torch  # type: ignore
            is_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception:
            is_cuda = False

        try:
            del self.model
        except Exception:
            pass

        gc.collect()
        if is_cuda:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

from __future__ import annotations
from typing import Any, Callable, Dict, Tuple
from .base import Embedder

_REG: Dict[str, Callable[[dict], Embedder]] = {}

def register(name: str):
    def _wrap(factory: Callable[[dict], Embedder]):
        _REG[name.lower()] = factory
        return factory
    return _wrap

def available() -> list[str]:
    return sorted(_REG)

def create(cfg: Any | None = None):
    if cfg is None:
        cfg = {"type": "sentence_transformers"}
    elif isinstance(cfg, str):
        cfg = {"type": cfg}
    elif not isinstance(cfg, dict):
        raise ValueError("embedding config must be dict/str/None")

    t = cfg.get("type", "sentence_transformers").lower()

    if t not in _REG:
        # map type -> module name
        lazy = {
            "sentence_transformers": "huggingface",
            "huggingface": "huggingface",
            "hash": "hash_embedder",
        }
        mod = lazy.get(t)
        if mod:
            import importlib
            importlib.import_module(f"{__name__}.{mod}")

    if t not in _REG:
        raise ValueError(f"No embedder registered for type '{t}'. Available: {sorted(_REG)}")

    emb = _REG[t](cfg)
    return emb, emb.dim

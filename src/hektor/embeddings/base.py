from __future__ import annotations
from typing import Any, Sequence, Protocol

import numpy as np

class Embedder(Protocol):
    dim: int
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...
    def close(self) -> None: ...

from __future__ import annotations
from typing import Any, Callable, Dict

_TOK_REG: Dict[str, Callable[[str], list[str]]] = {}

def register(name: str):
    def _wrap(fn: Callable[[str], list[str]]):
        _TOK_REG[name.lower()] = fn
        return fn
    return _wrap

def _builtin_simple(s: str) -> list[str]:
    return s.lower().split()

def resolve_tokenizer(spec: Any) -> Callable[[str], list[str]]:
    default = _TOK_REG.get("simple", _builtin_simple)

    if spec is None:
        return default

    if isinstance(spec, str):
        return _TOK_REG.get(spec.lower(), default)

    # custom: {"module": "pkg.mod", "callable": "fn", "kwargs": {}}
    if isinstance(spec, dict) and "module" in spec and "callable" in spec:
        import importlib
        mod = importlib.import_module(spec["module"])
        fn = getattr(mod, spec["callable"])  # type: ignore[attr-defined]
        kwargs = spec.get("kwargs", {})
        if kwargs:
            return lambda s: fn(s, **kwargs)
        return fn

    return default

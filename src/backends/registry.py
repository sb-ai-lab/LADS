"""
Backend registry — maps keyword strings (as typed in user prompt) to backend classes.
To add a new backend: import it and add an entry to BACKEND_REGISTRY.
"""
from typing import Type, Dict
from .base import AbstractAutoMLBackend

# Lazy imports so missing optional deps don't break the whole app
def _load_autogluon():
    from .autogluon_backend import AutoGluonBackend
    return AutoGluonBackend

def _load_flaml():
    from .flaml_backend import FLAMLBackend
    return FLAMLBackend


# Maps lowercase keywords (as user might type them) → loader function
_LOADERS: Dict[str, callable] = {
    "autogluon": _load_autogluon,
    "flaml":     _load_flaml,
}

# Cache instantiated classes
_CACHE: Dict[str, Type[AbstractAutoMLBackend]] = {}


def get_backend(name: str) -> AbstractAutoMLBackend:
    """
    Instantiate a backend by name.
    Example: get_backend("autogluon") -> AutoGluonBackend()
    """
    key = name.lower().strip()
    if key not in _LOADERS:
        available = list(_LOADERS.keys())
        raise ValueError(
            f"Unknown AutoML backend '{name}'. Available: {available}. "
            f"To add a new backend, create a class in graph/backends/ and register it here."
        )
    if key not in _CACHE:
        _CACHE[key] = _LOADERS[key]()
    return _CACHE[key]()


# Public name → class mapping (for display and routing)
BACKEND_REGISTRY = {k: v for k, v in _LOADERS.items()}

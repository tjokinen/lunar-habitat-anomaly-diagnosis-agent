"""Module registry — maps module_type strings to AnomalyModule classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from selene.core.interfaces import AnomalyModule

_REGISTRY: dict[str, type] = {}


def register_module(module_type: str):
    """Class decorator that registers an AnomalyModule implementation."""
    def decorator(cls):
        _REGISTRY[module_type] = cls
        return cls
    return decorator


def get_module_class(module_type: str) -> type:
    if module_type not in _REGISTRY:
        raise KeyError(
            f"Unknown module_type {module_type!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[module_type]

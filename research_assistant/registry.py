"""Capability registry: a global dict of name → callable, populated via @register."""

from __future__ import annotations

from typing import Callable

registry: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Decorator that registers a capability function under the given name."""

    def decorator(fn: Callable) -> Callable:
        registry[name] = fn
        return fn

    return decorator

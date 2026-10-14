"""Load built-in chains.

A module in this package that defines a ``@register`` class is imported on
startup. The default chains live in ``synthesis.templates`` and are registered
here as well.
"""

from __future__ import annotations

import importlib
import pkgutil


def load() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")
    import synthesis.templates  # noqa: F401  registers read/prompt/parse templates


load()

"""Built-in pipelines.

Any module in this package is imported on startup. A class decorated with
``@register`` becomes available to the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil


def load() -> None:
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")


load()

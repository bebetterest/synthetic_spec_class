"""Default chain templates.

Each module here defines ``read``, ``prompt``, and ``parse``. Importing this
package registers them with the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil

from synthesis.template import register_template

_LOADED = False


def load() -> None:
    global _LOADED
    if _LOADED:
        return
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        register_template(module)
    _LOADED = True


load()

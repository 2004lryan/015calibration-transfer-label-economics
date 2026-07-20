"""pytest configuration.

The analysis scripts keep their original numbered file names (``61_...``,
``62_...``) so that the paths quoted in their docstrings, in the logs, and in
the manuscript stay traceable. Those names are not valid Python identifiers, so
they cannot be imported with a plain ``import`` statement. This module provides
a loader that mirrors what the scripts themselves do internally
(``importlib.util.spec_from_file_location``) and registers the result in
``sys.modules`` so repeated loads are cached.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(file_name: str, module_name: str) -> ModuleType:
    """Load a (possibly digit-prefixed) script from code/ as ``module_name``."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = os.path.join(CODE_DIR, file_name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None, f"cannot load {path}"
    assert spec.loader is not None, f"no loader for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the scripts' own `export_utils` lookup resolves.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# export_utils must be importable under its canonical name before any script
# that loads it by that name is executed.
load_script("01_export_utils.py", "export_utils")

"""Bootstrap helpers for dependency-free unit tests of the ``agents`` package.

The package ``agents/__init__.py`` transitively imports
``agents.client`` which requires ``google-genai``, ``openai`` and
``anthropic`` at module load time.  We do not want to force those installs
just to exercise ``prompt_buffer`` and the Reflexion agent state machine,
so we register a namespace-only ``agents`` package pointing at the real
directory and then load each submodule via ``importlib``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "agents"


def _install_namespace_package() -> None:
    if "agents" in sys.modules:
        return
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(AGENTS_DIR)]
    sys.modules["agents"] = pkg


def _load(submodule: str):
    full = f"agents.{submodule}"
    if full in sys.modules:
        return sys.modules[full]
    _install_namespace_package()
    path = AGENTS_DIR / f"{submodule}.py"
    spec = importlib.util.spec_from_file_location(full, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


prompt_buffer = _load("prompt_buffer")
base = _load("base")
reflexion = _load("reflexion")

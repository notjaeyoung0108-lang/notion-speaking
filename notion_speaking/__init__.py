"""Compatibility package for running the project with ``python -m``.

The repository directory is named ``notion-speaking`` for GitHub, while the
Python package name used by the modules is ``notion_speaking``. Extending the
package path lets Python find the existing top-level modules without moving the
whole repository.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
__path__.append(str(_PROJECT_ROOT))  # type: ignore[name-defined]

"""Compatibility module for the historical ``consensus.src`` import path.

``control_robots.py`` used ``from consensus.src.MultipleTurtlebotsEnv import
MultipleTurtlebotsEnv``, but the module only ever existed one level up at
``consensus/MultipleTurtlebotsEnv.py``.  Rather than duplicating the
implementation, re-export the canonical classes here.
"""

from __future__ import annotations

from consensus.MultipleTurtlebotsEnv import (  # noqa: F401
    Formation,
    InfoGetter,
    MultipleTurtlebotsEnv,
    PID,
)

__all__ = ["Formation", "InfoGetter", "MultipleTurtlebotsEnv", "PID"]

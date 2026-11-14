"""DOMAS-MAPPO: multi-agent PPO training stack for the DOMAS TurtleBot fleet.

Subpackages are imported lazily on purpose.  Several of them require optional
third-party dependencies (torch, tensorboardX, ...); eager imports here would
make even ``import mappo.config`` fail on a machine without them.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "algorithms",
    "envs",
    "runner",
    "scripts",
    "train",
    "utils",
    "config",
]

_LAZY = {name: f"mappo.{name}" for name in __all__}


def __getattr__(name):
    """PEP 562 lazy submodule access."""
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

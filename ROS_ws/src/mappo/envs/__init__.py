"""Environments for DOMAS-MAPPO.

The previous version of this file parsed ``absl`` flags at import time
(``FLAGS(['train_sc.py'])``), which crashed on ``import mappo.envs`` whenever
absl-py was absent or argv had already been consumed.  Package initialisation
must be side-effect free; use the submodules explicitly instead.
"""

from __future__ import annotations

import importlib

__all__ = [
    "env_core",
    "env_continuous",
    "env_discrete",
    "env_wrappers",
    "VirtualTurtlebot",
    "spaces_compat",
    "mpe",
]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"mappo.envs.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

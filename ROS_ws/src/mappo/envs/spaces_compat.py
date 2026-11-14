"""Observation / action space compatibility layer.

The original code imported ``gym.spaces`` unconditionally.  ``gym`` is an
optional (and nowadays often replaced by ``gymnasium``) dependency, so a
missing install made the whole ``mappo`` package unimportable -- including
modules that never build a neural network.  This module resolves the real
``spaces`` implementation when available and otherwise provides a minimal,
API-compatible fallback (``shape`` / ``dtype`` / ``sample`` / ``contains``).

Usage::

    from mappo.envs.spaces_compat import spaces
    box = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
"""

from __future__ import annotations

import sys
import numpy as np

try:  # classic OpenAI Gym
    from gym import spaces as spaces  # type: ignore  # noqa: F811
except ImportError:  # pragma: no cover - depends on the host environment
    try:  # maintained successor
        from gymnasium import spaces as spaces  # type: ignore  # noqa: F811
    except ImportError:
        # ---- Minimal, dependency-free fallback -------------------------------
        class Box:  # type: ignore[no-redef]
            """Continuous box space, shape-compatible with ``gym.spaces.Box``."""

            def __init__(self, low, high, shape=None, dtype=np.float32):
                if shape is None:
                    shape = np.shape(low)
                self.shape = tuple(int(s) for s in shape)
                self.dtype = dtype
                self.low = np.full(self.shape, low, dtype=dtype)
                self.high = np.full(self.shape, high, dtype=dtype)

            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)

            def contains(self, x):
                arr = np.asarray(x)
                return arr.shape == self.shape and bool(
                    np.all(arr >= self.low) and np.all(arr <= self.high)
                )

            def __repr__(self):
                return f"Box(low={self.low.min()}, high={self.high.max()}, shape={self.shape})"

        class Discrete:  # type: ignore[no-redef]
            """Discrete action space with ``n`` available actions."""

            def __init__(self, n):
                self.n = int(n)
                self.shape = ()
                self.dtype = np.int64

            def sample(self):
                return int(np.random.randint(self.n))

            def contains(self, x):
                return bool(0 <= int(x) < self.n)

            def __repr__(self):
                return f"Discrete({self.n})"

        class MultiDiscrete:  # type: ignore[no-redef]
            def __init__(self, nvec):
                self.nvec = np.asarray(nvec, dtype=np.int64)
                self.shape = self.nvec.shape
                self.dtype = np.int64

            def sample(self):
                return np.array([np.random.randint(n) for n in self.nvec])

            def contains(self, x):
                arr = np.asarray(x)
                return arr.shape == self.shape and bool(np.all(arr >= 0) and np.all(arr < self.nvec))

            def __repr__(self):
                return f"MultiDiscrete({self.nvec.tolist()})"

        class MultiBinary:  # type: ignore[no-redef]
            def __init__(self, n):
                self.n = int(n)
                self.shape = (self.n,)
                self.dtype = np.int8

            def sample(self):
                return np.random.randint(0, 2, size=self.n).astype(self.dtype)

            def contains(self, x):
                arr = np.asarray(x)
                return arr.shape == self.shape and bool(np.all((arr == 0) | (arr == 1)))

            def __repr__(self):
                return f"MultiBinary({self.n})"

        # Expose this module itself as the ``spaces`` namespace so that
        # ``from mappo.envs.spaces_compat import spaces`` behaves like gym.
        spaces = sys.modules[__name__]  # type: ignore[assignment]

__all__ = ["spaces"]

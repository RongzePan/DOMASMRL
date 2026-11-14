"""Allow ``python -m mappo.train`` as used by ``train_mpe.sh``.

Previously the shell launcher invoked ``python3 -m mappo.train`` but the
package had no ``__main__.py``, so the command always failed with
``No module named mappo.train.__main__``.
"""

from __future__ import annotations

import sys

_RUNTIME_PACKAGES = {"numpy", "torch", "yaml", "imageio", "tensorboardX"}

try:
    from mappo.train.train import main
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the host
    root = (exc.name or "").split(".")[0]
    if root in _RUNTIME_PACKAGES:
        print(f"[domas] Cannot start training: missing Python package '{root}'.", file=sys.stderr)
        print("[domas] Install the training dependencies with:", file=sys.stderr)
        print(f"[domas]   {sys.executable} -m pip install numpy torch pyyaml", file=sys.stderr)
        raise SystemExit(2) from exc
    raise

if __name__ == "__main__":
    main(sys.argv[1:])

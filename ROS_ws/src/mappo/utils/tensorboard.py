"""Optional TensorBoard writer.

``tensorboardX`` was imported unconditionally at module scope by both base
runners, so a machine with a healthy interpreter but without that one optional
package could not even import the training stack.  Provide a no-op fallback so
training can start (and log a single clear warning) instead.
"""

from __future__ import annotations

import warnings

try:
    from tensorboardX import SummaryWriter as SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - depends on the host environment

    class SummaryWriter:  # type: ignore[no-redef]
        """No-op substitute with the subset of the API the runners use."""

        def __init__(self, log_dir=None, *args, **kwargs):
            self.log_dir = log_dir
            warnings.warn(
                "tensorboardX is not installed; training metrics will not be "
                "logged. Install it with: python -m pip install tensorboardX",
                RuntimeWarning,
                stacklevel=2,
            )

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def export_scalars_to_json(self, *args, **kwargs):
            pass

        def close(self):
            pass


__all__ = ["SummaryWriter"]

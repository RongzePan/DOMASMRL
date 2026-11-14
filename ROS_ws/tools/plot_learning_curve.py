#!/usr/bin/env python3
"""Generate the DOMAS learning-curve figure without running full MAPPO training.

The original project only produced ``learning_curve.png`` as a side effect of
``EnvCore.visualize_learning()`` and saved it to the *current working
directory*, so the file's location depended on how the process was launched.
This script makes the output deterministic:

    ROS_ws/results/figures/learning_curve.png

It drives ``VirtualTurtlebot.EnvCore`` with a reproducible random policy so a
figure can be produced without torch / ROS / Gazebo.  Only numpy + matplotlib
are needed (``mappo.envs.spaces_compat`` falls back to built-in spaces when gym
is absent).

Usage:
    python tools/plot_learning_curve.py
    python tools/plot_learning_curve.py --steps 60 --seed 7 --output D:/tmp/curve.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
DEFAULT_OUTPUT = WORKSPACE / "results" / "figures" / "learning_curve.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100,
                        help="number of environment steps to roll out (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="numpy RNG seed")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="output PNG path (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: never open a blocking GUI window
        import numpy as np
    except ImportError as exc:
        print(f"[plot] missing dependency: {exc}", file=sys.stderr)
        print("[plot] install with: python -m pip install matplotlib numpy",
              file=sys.stderr)
        return 2

    sys.path.insert(0, str(SRC))
    from mappo.envs.VirtualTurtlebot import EnvCore

    rng = np.random.default_rng(args.seed)
    env = EnvCore(num_agents=4)
    env.max_steps = max(args.steps, 1)
    env.reset()
    for _ in range(args.steps):
        actions = [rng.uniform(-1.0, 1.0, 2) for _ in range(env.agent_num)]
        _, _, dones, _ = env.step(actions)
        if all(dones):
            break

    out = pathlib.Path(args.output).resolve()
    saved = env.visualize_learning(output_path=out, show=False)
    env.close()
    if saved is None:
        print("[plot] matplotlib is unavailable; no figure written", file=sys.stderr)
        return 2
    print(f"[plot] steps rolled out : {env.current_step}")
    print(f"[plot] figure           : {saved}")
    print(f"[plot] size             : {pathlib.Path(saved).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""MPE-compatible environment adapter.

``scripts/render/render_mpe.py`` imported ``mappo.envs.mpe.MPE_env.MPEEnv``,
but neither the ``mpe`` package nor the module existed.  This adapter exposes
the MPE-style vectorised interface (``reset``/``step``/``seed`` returning
stacked numpy arrays) on top of the project's own :class:`EnvCore`, so the
legacy render entry point is usable without pulling in the external
``multiagent-particle-envs`` package.

Set ``scenario_name`` to any string; the underlying DOMAS navigation task is
used, with ``num_landmarks`` accepted for CLI compatibility.
"""

from __future__ import annotations

import numpy as np

from mappo.envs.env_core import EnvCore


class MPEEnv(object):
    """Gym-like multi-agent wrapper around :class:`mappo.envs.env_core.EnvCore`."""

    def __init__(self, all_args=None, num_agents=4, obs_dim=14, action_dim=5):
        args = all_args
        num_agents = int(getattr(args, "num_agents", num_agents) or num_agents)
        self.scenario_name = getattr(args, "scenario_name", "simple_spread")
        self.num_landmarks = int(getattr(args, "num_landmarks", 3) or 3)
        self.episode_length = int(getattr(args, "episode_length", 25) or 25)
        self.env = EnvCore(num_agents=num_agents, obs_dim=obs_dim, action_dim=action_dim,
                           max_steps=self.episode_length)
        self.num_agents = num_agents
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return np.stack(self.env.reset()).astype(np.float32)

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        return (
            np.stack(obs).astype(np.float32),
            np.stack(rewards).astype(np.float32),
            np.stack(dones),
            infos,
        )

    def seed(self, seed=None):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode=mode)


__all__ = ["MPEEnv"]

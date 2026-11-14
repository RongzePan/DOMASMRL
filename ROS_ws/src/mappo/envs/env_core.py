"""Lightweight multi-agent environment core for DOMAS-MAPPO.

This module was referenced throughout the repository (``from mappo.envs import
env_core`` in ``control_robots.py``, and as a commented alternative import in
``env_continuous.py``) but was never present.  It implements the standard
``light_mappo`` ``EnvCore`` contract used by the runners and buffers:

* ``observation_space`` / ``action_space`` / ``share_observation_space`` lists;
* ``reset() -> list[np.ndarray]`` with one observation per agent;
* ``step(actions) -> (obs, rewards, dones, infos)`` where ``rewards`` is a list
  of single-element lists (the convention consumed by ``DummyVecEnv``);
* ``close()`` / ``render()`` / ``seed()``.

Defaults (4 agents, 14-dim observation, 5 discrete actions) match
``mappo/config/train_config.yaml``.  The module intentionally depends only on
numpy so it can be imported on a ROS robot without gym/torch/matplotlib.
"""

from __future__ import annotations

import numpy as np

from mappo.envs.spaces_compat import spaces


class EnvCore(object):
    """Consensus-oriented multi-agent navigation environment.

    Each agent moves on a bounded 2-D plane towards its own target.  An action
    is either a discrete integer in ``[0, action_dim)`` or a one-hot / logits
    vector of length ``action_dim``; the five discrete actions map to
    (stay, +x, -x, +y, -y).
    """

    _DIRECTIONS = np.array(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    def __init__(self, num_agents=4, obs_dim=14, action_dim=5, max_steps=200):
        if num_agents < 1 or obs_dim < 8 or action_dim < 2:
            raise ValueError("num_agents >= 1, obs_dim >= 8 and action_dim >= 2 are required")
        self.agent_num = int(num_agents)
        self.num_agents = self.agent_num  # alias used by env_discrete.py
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_steps = int(max_steps)
        self.step_size = 0.12
        self.success_threshold = 0.25
        self.world_size = 5.0
        self.current_step = 0
        self._rng = np.random.default_rng(0)

        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.agent_num)]
        share_obs_dim = self.obs_dim * self.agent_num
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

        self.positions = np.zeros((self.agent_num, 2), dtype=np.float32)
        self.targets = np.zeros((self.agent_num, 2), dtype=np.float32)
        self._initial_targets()

    # ------------------------------------------------------------------ setup
    def _initial_targets(self):
        angles = np.linspace(0.0, 2.0 * np.pi, self.agent_num, endpoint=False)
        self.targets[:, 0] = 2.0 * np.cos(angles)
        self.targets[:, 1] = 2.0 * np.sin(angles)

    def seed(self, seed=None):
        """Seed the internal RNG; returns the seed list (gym-compatible)."""
        self._rng = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        """Reset all agents and return a list of per-agent observations."""
        self.current_step = 0
        self.positions = self._rng.uniform(-1.0, 1.0, size=(self.agent_num, 2)).astype(np.float32)
        return self._get_obs()

    # -------------------------------------------------------------- transition
    @staticmethod
    def _as_action_index(action):
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 1:
            return int(np.clip(round(float(arr[0])), 0, 4))
        return int(np.argmax(arr[:5]))

    def _get_obs(self):
        obs = []
        for i in range(self.agent_num):
            relative = self.targets[i] - self.positions[i]
            parts = [
                self.positions[i] / self.world_size,
                relative / self.world_size,
                [np.linalg.norm(relative) / (2.0 * self.world_size)],
            ]
            for j in range(self.agent_num):
                if j == i:
                    continue
                parts.append((self.positions[j] - self.positions[i]) / (2.0 * self.world_size))
            vec = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
            if vec.size < self.obs_dim:
                vec = np.pad(vec, (0, self.obs_dim - vec.size), mode="constant")
            obs.append(vec[: self.obs_dim].astype(np.float32))
        return obs

    def step(self, actions):
        """Advance one timestep.

        Returns ``(obs, rewards, dones, infos)`` with rewards shaped
        ``[[r0], [r1], ...]`` as expected by the MAPPO replay buffers.
        """
        if actions is None:
            raise ValueError("actions must not be None")
        if len(actions) != self.agent_num:
            raise ValueError(f"expected {self.agent_num} actions, got {len(actions)}")

        self.current_step += 1
        for i, action in enumerate(actions):
            idx = self._as_action_index(action) % self._DIRECTIONS.shape[0]
            self.positions[i] = np.clip(
                self.positions[i] + self.step_size * self._DIRECTIONS[idx],
                -self.world_size,
                self.world_size,
            )

        rewards, dones = [], []
        for i in range(self.agent_num):
            distance = float(np.linalg.norm(self.targets[i] - self.positions[i]))
            reward = -distance + (10.0 if distance < self.success_threshold else 0.0)
            rewards.append([reward])
            dones.append(distance < self.success_threshold)
        timeout = self.current_step >= self.max_steps
        if timeout:
            dones = [True] * self.agent_num
        return self._get_obs(), rewards, dones, [{} for _ in range(self.agent_num)]

    # ------------------------------------------------------------------ misc
    def close(self):
        """No external resources are held by the lightweight core."""

    def render(self, mode="human"):
        """Text rendering is always available; graphical rendering is optional."""
        if mode == "human":
            for i in range(self.agent_num):
                print(f"agent {i}: pos={self.positions[i]}, target={self.targets[i]}")
        elif mode != "rgb_array":
            raise NotImplementedError(f"Render mode not implemented: {mode}")
        return None


__all__ = ["EnvCore"]

"""Lightweight MAPPO agent adapter for the ROS control node.

``consensus/src/control_robots.py`` instantiates ``r_mappo.RMAPPOAgent`` and
calls ``select_action`` / ``update`` on it, but no such class existed in the
repository -- ``r_mappo.py`` only provides the off-line PPO *trainer*
(:class:`~mappo.algorithms.algorithm.r_mappo.RMAPPO`).  This module supplies the
missing on-line agent.

Scope: this is a deliberately small, numpy-only linear-Gaussian policy with a
running reward baseline and a REINFORCE-style update.  It gives the ROS control
loop a working ``select_action`` / ``update`` contract and can be hot-swapped
with a trained :class:`RMAPPOPolicy` via the ``policy`` argument.  Full MAPPO
training (actor/critic networks, replay buffer, PPO epochs) is implemented in
``mappo.train`` + ``RMAPPO`` and is the recommended training path.
"""

from __future__ import annotations

import numpy as np


class RMAPPOAgent(object):
    """State -> multi-agent continuous action adapter used by the ROS node.

    :param env: optional environment exposing ``agent_num``/``num_agents``,
        ``obs_dim`` and ``action_dim``.
    :param config: optional dict with ``num_agents``, ``obs_dim``,
        ``action_dim`` (or gym-style ``observation_space`` / ``action_space``).
    :param policy: optional pre-trained policy exposing ``act(obs, ...)``;
        when given, inference is delegated to it.
    :param lr: step size for the on-line policy update.
    :param seed: RNG seed (also used for the initial weights).
    """

    def __init__(self, env=None, config=None, policy=None, lr=1e-3, seed=0):
        config = dict(config or {})
        num_agents = config.get("num_agents") or getattr(env, "agent_num", None) \
            or getattr(env, "num_agents", None) or 4
        obs_dim = config.get("obs_dim")
        action_dim = config.get("action_dim") or config.get("act_dim")

        if obs_dim is None:
            space = config.get("observation_space") or getattr(env, "observation_space", None)
            obs_dim = self._space_dim(space, default=14)
        if action_dim is None:
            space = config.get("action_space") or getattr(env, "action_space", None)
            action_dim = self._space_dim(space, default=2)

        self.num_agents = int(num_agents)
        self.obs_dim = max(int(obs_dim), 1)
        # The ROS node reads action[0] / action[1] as linear & angular velocity.
        self.action_dim = max(int(action_dim), 2)
        self.lr = float(lr)
        self.policy = policy
        self._rng = np.random.default_rng(seed)
        self._weights = self._rng.normal(
            0.0, 0.05, size=(self.action_dim, self.obs_dim)
        ).astype(np.float32)
        self._baseline = 0.0
        self._baseline_count = 0
        self._last_state = None
        self._last_action = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _space_dim(space, default):
        if space is None:
            return default
        if isinstance(space, (list, tuple)):
            space = space[0]
        shape = getattr(space, "shape", None)
        if shape:
            return int(np.prod(shape))
        n = getattr(space, "n", None)
        if n is not None:
            return int(n)
        return default

    def _state_matrix(self, state):
        """Coerce any supported state representation to (num_agents, obs_dim)."""
        if state is None:
            return np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        if isinstance(state, dict):
            state = state.get("obs", state.get("state"))
        arr = np.asarray(state, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(self.num_agents, -1) if arr.size % self.num_agents == 0 \
                else np.tile(arr, (self.num_agents, 1))
        elif arr.ndim > 2:
            arr = arr.reshape(self.num_agents, -1)
        if arr.shape[0] != self.num_agents:
            arr = np.resize(arr, (self.num_agents, arr.shape[1]))
        if arr.shape[1] < self.obs_dim:
            arr = np.pad(arr, ((0, 0), (0, self.obs_dim - arr.shape[1])))
        return arr[:, : self.obs_dim].astype(np.float32)

    @staticmethod
    def _scalar_reward(reward):
        arr = np.asarray(reward, dtype=np.float32)
        return float(np.mean(arr)) if arr.size else 0.0

    # ------------------------------------------------------------------- API
    def select_action(self, state, deterministic=False):
        """Return an array of shape ``(num_agents, action_dim)`` in ``[-1, 1]``."""
        obs = self._state_matrix(state)
        if self.policy is not None and hasattr(self.policy, "act"):
            action = np.asarray(self.policy.act(obs, None, None, deterministic=deterministic)[0])
            action = np.asarray(action, dtype=np.float32).reshape(self.num_agents, -1)
        else:
            logits = obs @ self._weights.T
            action = np.tanh(logits)
            if not deterministic:
                action = action + self._rng.normal(0.0, 0.05, size=action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
        if action.shape[1] < self.action_dim:
            action = np.pad(action, ((0, 0), (0, self.action_dim - action.shape[1])))
        self._last_state = obs
        self._last_action = action[:, : self.action_dim]
        return self._last_action

    def update(self, state, action, reward, next_state=None, done=None):
        """One on-line REINFORCE-style policy improvement step.

        Returns a dict with the observed reward, the centred advantage and the
        policy gradient norm, which is convenient for ROS logging.
        """
        obs = self._state_matrix(state)
        r = self._scalar_reward(reward)
        self._baseline_count += 1
        self._baseline += (r - self._baseline) / self._baseline_count
        advantage = r - self._baseline
        if self._last_state is None:
            self._last_state = obs
        grad = np.outer(np.mean(self._last_action, axis=0)
                        if self._last_action is not None else np.zeros(self.action_dim),
                        np.mean(self._last_state, axis=0))
        grad *= advantage
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e-8:
            self._weights += self.lr * grad[: self.action_dim, : self.obs_dim] / grad_norm
        return {"reward": r, "advantage": float(advantage), "grad_norm": grad_norm}

    def save(self, path):
        np.savez(str(path), weights=self._weights, baseline=self._baseline,
                 baseline_count=self._baseline_count)

    def load(self, path):
        with np.load(str(path)) as data:
            self._weights = data["weights"]
            self._baseline = float(data["baseline"])
            self._baseline_count = int(data["baseline_count"])
        return self


__all__ = ["RMAPPOAgent"]

"""
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        # Initialize the environments list
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # Step through environments synchronously and collect the results for each environment
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            # if 'bool' in done.__class__.__name__:
            if isinstance(done, bool):
                # check if done is a Boolean
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    # Check done status of all agents
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        """
        Resets all environments and returns the initial observation.
        """
        obs = [env.reset() for env in self.envs]
        # Return format： [env_num, agent_num, obs_dim]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError("Render mode not implemented: " + mode)


def _subproc_worker(remote, parent_remote, env_fn):
    """Worker entry point for :class:`SubprocVecEnv` (fork start method)."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "render":
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.share_observation_space,
                             env.action_space))
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocVecEnv(DummyVecEnv):
    """Run each environment in a separate process when ``fork`` is available.

    ``scripts/render/render_mpe.py`` imports this class but it was never
    implemented.  On platforms without ``fork`` (Windows / spawn start method)
    it degrades transparently to in-process execution via :class:`DummyVecEnv`
    so that the public API is identical everywhere.
    """

    def __init__(self, env_fns):
        try:
            import multiprocessing as mp
            self._ctx = mp.get_context("fork")
            self._mp = mp
        except (ValueError, ImportError):
            self._ctx = None
        if self._ctx is None:
            super().__init__(env_fns)
            self._remotes = None
            self._procs = []
            return

        self.num_envs = len(env_fns)
        self._remotes, work_remotes = zip(*[self._ctx.Pipe() for _ in env_fns])
        self._procs = []
        for work_remote, remote, env_fn in zip(work_remotes, self._remotes, env_fns):
            proc = self._ctx.Process(
                target=_subproc_worker, args=(work_remote, remote, env_fn), daemon=True
            )
            proc.start()
            work_remote.close()
            self._procs.append(proc)
        for remote in self._remotes:
            remote.send(("get_spaces", None))
        spaces = [remote.recv() for remote in self._remotes][0]
        self.observation_space, self.share_observation_space, self.action_space = spaces
        self.actions = None

    def step_async(self, actions):
        self.actions = actions
        if self._remotes is not None:
            for remote, action in zip(self._remotes, actions):
                remote.send(("step", action))

    def step_wait(self):
        if self._remotes is None:
            return super().step_wait()
        results = [remote.recv() for remote in self._remotes]
        obs, rews, dones, infos = map(np.array, zip(*results))
        for i, done in enumerate(dones):
            if np.all(done):
                self._remotes[i].send(("reset", None))
                obs[i] = self._remotes[i].recv()
        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        if self._remotes is None:
            return super().reset()
        for remote in self._remotes:
            remote.send(("reset", None))
        return np.array([remote.recv() for remote in self._remotes])

    def close(self):
        if self._remotes is None:
            return super().close()
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=1.0)

    def render(self, mode="human"):
        if self._remotes is None:
            return super().render(mode=mode)
        for remote in self._remotes:
            remote.send(("render", ((mode,), {})))
        return np.array([remote.recv() for remote in self._remotes])
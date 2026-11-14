from mappo.envs.spaces_compat import spaces
import numpy as np

from mappo.envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
Wrapper for continuous action environments
Provides interface for multi-agent reinforcement learning
"""

    def __init__(self):
        """Initialize the environment wrapper"""
        self.env = EnvCore()  # Replace with EnvCore instantiation
        self.num_agent = self.env.agent_num  # Get number of agents from DOMAS
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # If true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True  # Flag indicating if agents can move

        # Configure observation and action spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0  # Dimension for shared observations
        total_action_space = []

        for agent in range(self.num_agent):
            # Physical action space using DOMAS definitions
            u_action_space = spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # Total action space
            self.action_space.append(total_action_space[agent])

            # Observation space using DOMAS definitions
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf] range

        # Configure shared observation space
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        return self.env.seed(seed)

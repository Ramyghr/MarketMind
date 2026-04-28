import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    State: 128-dim SSL embedding
    Actions: 0=Flat, 1=Long, 2=Short
    Reward: position * price_return
    """
    def __init__(self, embeddings, prices):
        super().__init__()
        self.embeddings = embeddings  # (N, 128)
        self.prices = prices          # (N,)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(128,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.portfolio = 1.0
        return self.embeddings[0].astype(np.float32), {}

    def step(self, action):
        self.position = {0: 0, 1: 1, 2: -1}[action]
        ret = (self.prices[self.t+1] - self.prices[self.t]) / self.prices[self.t]
        reward = float(self.position * ret)
        self.portfolio *= (1 + reward)
        self.t += 1
        done = self.t >= len(self.embeddings) - 1
        return self.embeddings[self.t].astype(np.float32), reward, done, False, {}
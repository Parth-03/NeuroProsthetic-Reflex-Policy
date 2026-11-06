"""A minimal stand-in for ``myoElbowPose1D6MRandom-v0``.

Same observation layout (``[q, qdot, act(6), target - q]``) and action space,
with simple first-order joint dynamics, so the wrapper can be tested without
MuJoCo or MyoSuite.
"""

import gymnasium as gym
import numpy as np


class FakeElbowEnv(gym.Env):
    q_min, q_max = 0.0, 2.27
    dt = 0.02
    gain = 4.0

    def __init__(self, episode_length=100, target=None):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(9,), dtype=np.float32)
        self.episode_length = episode_length
        self.fixed_target = target
        self._rng = np.random.default_rng(0)

    def _obs(self):
        return np.array([self.q, self.qdot, *self.act, self.target - self.q],
                        dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.q = float(self._rng.uniform(self.q_min, self.q_max))
        self.qdot = 0.0
        self.act = np.zeros(6)
        self.target = (self.fixed_target if self.fixed_target is not None
                       else float(self._rng.uniform(self.q_min, self.q_max)))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
        self.act = action
        self.qdot = self.gain * (action[:3].mean() - action[3:].mean())
        self.q = float(np.clip(self.q + self.dt * self.qdot, self.q_min, self.q_max))
        self.t += 1
        truncated = self.t >= self.episode_length
        return self._obs(), 0.0, False, truncated, {}

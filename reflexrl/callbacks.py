"""Stable-Baselines3 callback that tracks pain and withdrawal per episode."""

import csv
import os

from stable_baselines3.common.callbacks import BaseCallback

EPISODE_FIELDS = ("episode", "length", "reward", "pain_steps", "withdraw_steps")


class PainLoggingCallback(BaseCallback):
    """Records per-episode length, reward, pain steps and withdrawal steps.

    Assumes a single (non-vectorized) training environment. Episode summaries
    are also written to the SB3 logger under ``pain/`` and ``withdraw/``.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episodes = []
        self._reset_counters()

    def _reset_counters(self):
        self._pain = 0
        self._withdraw = 0
        self._length = 0
        self._reward = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if len(infos) == 0:
            return True
        info = infos[0]
        done = bool(self.locals["dones"][0])

        self._pain += int(info.get("pain", 0))
        self._withdraw += int(info.get("in_withdrawal", 0))
        self._length += 1
        self._reward += float(self.locals["rewards"][0])

        if done:
            episode = dict(
                episode=len(self.episodes) + 1,
                length=self._length,
                reward=self._reward,
                pain_steps=self._pain,
                withdraw_steps=self._withdraw,
            )
            self.episodes.append(episode)
            if self.verbose > 0:
                print(f"[Episode {episode['episode']}] len={self._length}, "
                      f"pain_steps={self._pain}, withdraw_steps={self._withdraw}, "
                      f"reward={self._reward:.2f}")
            self.logger.record("pain/steps_per_episode", self._pain)
            self.logger.record("withdraw/steps_per_episode", self._withdraw)
            self._reset_counters()
        return True

    @property
    def episode_pain(self):
        return [e["pain_steps"] for e in self.episodes]

    @property
    def episode_withdraw(self):
        return [e["withdraw_steps"] for e in self.episodes]

    @property
    def episode_rewards(self):
        return [e["reward"] for e in self.episodes]

    @property
    def episode_lengths(self):
        return [e["length"] for e in self.episodes]

    def save_csv(self, path):
        """Writes the per-episode records to a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
            writer.writeheader()
            writer.writerows(self.episodes)
        return path

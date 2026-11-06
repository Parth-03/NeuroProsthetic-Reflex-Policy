"""Figures for training and evaluation."""

import matplotlib.pyplot as plt
import numpy as np


def moving_average(x, window):
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_reflex_only_raw(q, q_target, error, ax=None):
    """Joint angle, target and error for the bare reflex."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q, label="q")
    ax.plot(q_target, label="q_target")
    ax.plot(error, label="error")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Angle / error (rad)")
    ax.set_title("Reflex-only behavior (raw environment)")
    ax.legend()
    return ax


def plot_training_pain(episode_pain, episode_withdraw=None, window=50, ax=None):
    """Pain (and optionally withdrawal) steps per training episode."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    episodes = np.arange(1, len(episode_pain) + 1)
    line, = ax.plot(episodes, episode_pain, alpha=0.3, linewidth=0.8,
                    label="Pain steps / episode")
    smooth = moving_average(episode_pain, window)
    if len(smooth) < len(episode_pain):
        ax.plot(episodes[window - 1:], smooth, color=line.get_color(), linewidth=2,
                label=f"Pain ({window}-episode mean)")
    if episode_withdraw is not None:
        ax.plot(episodes, episode_withdraw, alpha=0.5, linewidth=0.8, linestyle="--",
                label="Withdrawal steps / episode")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Steps")
    ax.set_title("Pain and withdrawal during training")
    ax.legend()
    return ax


def plot_reward_comparison(reflex_rewards, rl_rewards, ax=None):
    """Total reward per evaluation episode for both controllers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    episodes = np.arange(len(reflex_rewards))
    for rewards, label in [(reflex_rewards, "Reflex only"), (rl_rewards, "Reflex + RL")]:
        points = ax.scatter(episodes, rewards, label=label, alpha=0.7)
        ax.axhline(np.mean(rewards), linestyle="--", color=points.get_facecolor()[0],
                   label=f"{label} mean: {np.mean(rewards):.1f}")
    ax.set_xlabel("Evaluation episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Episode reward: reflex vs. reflex + RL")
    ax.legend()
    return ax


def plot_episode_trace(reflex, rl, ax=None, title="Error, pain and withdrawal"):
    """Control error over one episode, with pain events and RL withdrawal shaded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    t_reflex = np.arange(len(reflex["error"]))
    t_rl = np.arange(len(rl["error"]))
    ax.plot(t_reflex, reflex["error"], label="Reflex only")
    ax.plot(t_rl, rl["error"], label="Reflex + RL")

    pain_rl = np.where(rl["pain"] > 0)[0]
    ax.scatter(pain_rl, rl["error"][pain_rl], marker="x", label="Pain (RL)")
    pain_reflex = np.where(reflex["pain"] > 0)[0]
    ax.scatter(pain_reflex, reflex["error"][pain_reflex], marker="o", s=12,
               label="Pain (reflex)")

    withdraw = rl["withdraw"] > 0
    if withdraw.any():
        lo = min(np.min(rl["error"]), np.min(reflex["error"]))
        hi = max(np.max(rl["error"]), np.max(reflex["error"]))
        ax.fill_between(t_rl, lo, hi, where=withdraw, alpha=0.15,
                        label="Withdrawal (RL)")

    ax.axhline(0.0, linestyle="--", linewidth=1, color="k")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Control error (rad)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax

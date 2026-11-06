"""Rollouts for the reflex-only and reflex + RL controllers."""

import numpy as np

from reflexrl.observations import ERROR_IDX, extract_q_qdot_qtarget
from reflexrl.reflex import reflex_controller


def zero_residual_policy(env):
    """Policy that adds nothing to the reflex (reflex-only control)."""
    zeros = np.zeros(env.action_space.shape, dtype=np.float32)
    return lambda state: zeros


def model_policy(model, deterministic=True):
    """Wraps a Stable-Baselines3 model as ``state -> action``."""
    return lambda state: model.predict(state, deterministic=deterministic)[0]


def run_episode(env, policy, max_steps=200, seed=None, frame_fn=None):
    """Runs one episode of a wrapped environment.

    Args:
        env: A :class:`reflexrl.ReflexResidualElbow` environment.
        policy: Callable mapping the policy state to a residual action.
        max_steps: Step limit (the MyoSuite task itself ends after 100 steps).
        seed: Reset seed. Using the same seed for two controllers gives them
            the same target and the same random pain events.
        frame_fn: Optional ``env -> RGB array`` called after every step; the
            frames are returned under ``frames``.

    Returns:
        dict with the total ``reward`` and per-step arrays ``q``, ``q_env_target``,
        ``q_ctrl_target``, ``error`` (control error), ``pain``, ``withdraw`` and
        ``max_act`` (largest muscle activation).
    """
    state, _ = env.reset(seed=seed)
    keys = ("q", "q_env_target", "q_ctrl_target", "error", "pain", "withdraw", "max_act")
    log = {k: [] for k in keys}
    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        state, reward, terminated, truncated, info = env.step(policy(state))
        total_reward += reward
        log["q"].append(float(state[0]))
        log["q_env_target"].append(float(info["q_env_target"]))
        log["q_ctrl_target"].append(float(info["q_ctrl_target"]))
        log["error"].append(float(state[ERROR_IDX]))
        log["pain"].append(int(info["pain"]))
        log["withdraw"].append(int(info["in_withdrawal"]))
        log["max_act"].append(float(np.max(np.abs(info["muscle_act"]))))
        if frame_fn is not None:
            frames.append(frame_fn(env))
        if terminated or truncated:
            break

    result = {k: np.asarray(v) for k, v in log.items()}
    result["reward"] = total_reward
    if frame_fn is not None:
        result["frames"] = frames
    return result


def compare_controllers(env, model, num_episodes=10, max_steps=200, seed=0):
    """Runs reflex-only and reflex + RL on the same seeds.

    Returns:
        (reflex_results, rl_results): lists of :func:`run_episode` outputs.
    """
    reflex_policy = zero_residual_policy(env)
    rl_policy = model_policy(model)
    reflex_results, rl_results = [], []
    for ep in range(num_episodes):
        reflex_results.append(run_episode(env, reflex_policy, max_steps, seed=seed + ep))
        rl_results.append(run_episode(env, rl_policy, max_steps, seed=seed + ep))
    return reflex_results, rl_results


def run_reflex_only_raw(base_env, max_steps=200, k_p=5.0, dead_zone=0.02, seed=None):
    """Runs the bare reflex on the unwrapped environment (no pain, no RL).

    Returns:
        (q, q_target, error) arrays.
    """
    obs, _ = base_env.reset(seed=seed) if seed is not None else base_env.reset()
    qs, qts, errs = [], [], []
    for _ in range(max_steps):
        q, _, q_target = extract_q_qdot_qtarget(obs)
        obs, _, terminated, truncated, _ = base_env.step(
            reflex_controller(q, q_target, k_p=k_p, dead_zone=dead_zone))
        qs.append(q)
        qts.append(q_target)
        errs.append(q_target - q)
        if terminated or truncated:
            break
    return np.array(qs), np.array(qts), np.array(errs)


def summarize(results):
    """Mean and std of reward, pain steps and withdrawal steps over episodes."""
    rewards = np.array([r["reward"] for r in results])
    pain = np.array([r["pain"].sum() for r in results])
    withdraw = np.array([r["withdraw"].sum() for r in results])
    return dict(
        reward_mean=rewards.mean(), reward_std=rewards.std(),
        pain_mean=pain.mean(), pain_std=pain.std(),
        withdraw_mean=withdraw.mean(), withdraw_std=withdraw.std(),
    )

import numpy as np
import pytest

from reflexrl import (ERROR_IDX, EXTENSOR_INDICES, FLEXOR_INDICES, ReflexResidualElbow,
                      build_state, compute_custom_reward, extract_q_qdot_qtarget,
                      reflex_controller)
from reflexrl.evaluation import run_episode, summarize, zero_residual_policy
from tests.fake_elbow import FakeElbowEnv

# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------


def test_extract_q_qdot_qtarget():
    obs = np.array([1.0, -0.5, 0, 0, 0, 0, 0, 0, 0.3])
    q, qdot, q_target = extract_q_qdot_qtarget(obs)
    assert (q, qdot) == (1.0, -0.5)
    assert q_target == pytest.approx(1.3)


def test_build_state_layout():
    obs = np.array([1.0, 0.2, 0, 0, 0, 0, 0, 0, 0.0])
    state = build_state(obs, q_ctrl_target=1.5, pain=1, in_withdrawal=True)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state, [1.0, 0.2, 1.5, 0.5, 1.0, 1.0], rtol=1e-6)
    assert state[ERROR_IDX] == pytest.approx(0.5)


# ----------------------------------------------------------------------------
# Reflex
# ----------------------------------------------------------------------------


def test_reflex_silent_inside_dead_zone():
    assert not reflex_controller(1.0, 1.01, dead_zone=0.02).any()


def test_reflex_flexes_when_target_is_larger():
    act = reflex_controller(1.0, 1.1, k_p=5.0, dead_zone=0.02)
    np.testing.assert_allclose(act[list(FLEXOR_INDICES)], 0.4, rtol=1e-6)
    assert not act[list(EXTENSOR_INDICES)].any()


def test_reflex_extends_when_target_is_smaller():
    act = reflex_controller(1.0, 0.9, k_p=5.0, dead_zone=0.02)
    np.testing.assert_allclose(act[list(EXTENSOR_INDICES)], 0.4, rtol=1e-6)
    assert not act[list(FLEXOR_INDICES)].any()


def test_reflex_saturates_at_one():
    assert reflex_controller(0.0, 2.0, k_p=5.0).max() == 1.0


# ----------------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------------


def test_reward_terms():
    r = compute_custom_reward(q=1.0, qdot=2.0, q_ctrl_target=1.5,
                              muscle_act=np.ones(6), pain=1)
    expected = -10.0 * 0.25 - 0.1 * 4.0 - 0.001 * 6.0 - 2.0
    assert r == pytest.approx(expected)


def test_reward_is_zero_at_rest_on_target():
    assert compute_custom_reward(1.0, 0.0, 1.0, np.zeros(6), 0) == 0.0


# ----------------------------------------------------------------------------
# Wrapper
# ----------------------------------------------------------------------------


def make(**kwargs):
    kwargs.setdefault("random_pain_prob", 0.0)
    target = kwargs.pop("target", None)
    return ReflexResidualElbow(FakeElbowEnv(target=target), **kwargs)


def set_joint(env, q, target=None):
    """Moves the fake joint (and optionally its target) after a reset."""
    env.unwrapped.q = q
    if target is not None:
        env.unwrapped.target = target
    env._last_raw_obs = env.unwrapped._obs()


def test_spaces():
    env = make(residual_scale=0.2)
    assert env.action_space.shape == (6,)
    assert np.allclose(env.action_space.high, 0.2)
    assert env.observation_space.shape == (6,)
    state, _ = env.reset(seed=0)
    assert state.shape == (6,)


def test_residual_is_clipped_to_scale():
    env = make(residual_scale=0.2, pain_threshold=2.0, target=1.0)
    env.reset(seed=0)
    set_joint(env, 1.0)  # on target: the reflex is silent
    _, _, _, _, info = env.step(np.full(6, 5.0))
    np.testing.assert_allclose(info["muscle_act"], 0.2)
    _, _, _, _, info = env.step(np.full(6, -5.0))
    np.testing.assert_allclose(info["muscle_act"], -0.2)


def test_info_fields():
    env = make()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(6))
    for key in ("pain", "pain_from_action", "pain_from_random", "in_withdrawal",
                "q_env_target", "q_ctrl_target", "muscle_act"):
        assert key in info


def test_pain_from_large_activation_starts_withdrawal():
    env = make(pain_threshold=0.9, safe_offset=0.5)
    env.reset(seed=0)
    set_joint(env, 0.2, target=2.0)  # large error -> reflex saturates at 1.0
    state, _, _, _, info = env.step(np.zeros(6))
    assert info["pain_from_action"] == 1 and info["in_withdrawal"] == 1
    assert info["q_ctrl_target"] == pytest.approx(2.0 + 0.5)
    assert state[2] == pytest.approx(2.5) and state[5] == 1.0


def test_withdrawal_ends_when_pain_stops():
    env = make(pain_threshold=0.9, safe_offset=0.05, target=1.0)
    env.reset(seed=0)
    set_joint(env, 0.2)
    _, _, _, _, info = env.step(np.zeros(6))
    assert info["in_withdrawal"] == 1
    for _ in range(100):
        _, _, _, truncated, info = env.step(np.zeros(6))
        if not info["in_withdrawal"] or truncated:
            break
    assert info["in_withdrawal"] == 0


def test_saturated_reflex_keeps_itself_in_withdrawal():
    """With the safe target out of reach, the saturated reflex keeps causing pain."""
    env = make(pain_threshold=0.9, safe_offset=0.5, target=2.2)  # safe target 2.7 > q_max
    env.reset(seed=0)
    set_joint(env, 0.2)
    pains = [env.step(np.zeros(6))[4]["pain"] for _ in range(60)]
    assert all(pains[-30:])


def test_negative_residual_breaks_the_pain_loop():
    """Subtracting 0.2 from the agonists keeps activation at 0.8 < 0.9."""
    env = make(pain_threshold=0.9, safe_offset=0.5, target=2.2, residual_scale=0.2)
    env.reset(seed=0)
    set_joint(env, 0.2)
    for _ in range(60):
        state, _, _, _, info = env.step(np.zeros(6))
    assert info["in_withdrawal"] == 1
    residual = np.full(6, -0.2, dtype=np.float32)
    pains = [env.step(residual)[4]["pain"] for _ in range(5)]
    assert pains == [0, 0, 0, 0, 0]


def test_timer_withdrawal_lasts_fixed_steps():
    env = make(pain_threshold=0.9, withdraw_steps=3, safe_offset=0.0, target=1.0)
    env.reset(seed=0)
    set_joint(env, 0.2)
    flags = [env.step(np.zeros(6))[4]["in_withdrawal"]]  # pain onset
    # A residual of -0.2 keeps activation at 0.8, so no further pain occurs.
    flags += [env.step(np.full(6, -0.2))[4]["in_withdrawal"] for _ in range(4)]
    assert flags == [1, 1, 1, 0, 0]


def test_random_pain_events_have_fixed_length():
    env = make(random_pain_prob=1.0, random_pain_steps=5, pain_threshold=2.0)
    env.reset(seed=0)
    pains = [env.step(np.zeros(6))[4]["pain_from_random"] for _ in range(20)]
    assert all(pains)


def test_no_random_pain_when_probability_is_zero():
    env = make(random_pain_prob=0.0, pain_threshold=2.0)
    env.reset(seed=0)
    assert not any(env.step(np.zeros(6))[4]["pain"] for _ in range(100))


def test_same_seed_gives_same_random_pain():
    def pain_sequence(seed):
        env = make(random_pain_prob=0.1, pain_threshold=2.0)
        env.reset(seed=seed)
        return [env.step(np.zeros(6))[4]["pain_from_random"] for _ in range(100)]

    assert pain_sequence(3) == pain_sequence(3)
    assert pain_sequence(3) != pain_sequence(4)


def test_reset_clears_withdrawal():
    env = make(pain_threshold=0.9, target=2.2)
    env.reset(seed=0)
    set_joint(env, 0.2)
    for _ in range(30):
        env.step(np.zeros(6))
    assert env.in_withdrawal
    state, _ = env.reset(seed=1)
    assert not env.in_withdrawal and state[4] == 0 and state[5] == 0


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------


def test_run_episode_and_summary():
    env = make(random_pain_prob=0.1)
    result = run_episode(env, zero_residual_policy(env), max_steps=200, seed=0)
    assert len(result["error"]) == 100  # fake episode length
    for key in ("q", "q_env_target", "q_ctrl_target", "pain", "withdraw", "max_act"):
        assert len(result[key]) == 100
    s = summarize([result, result])
    assert s["reward_std"] == 0.0 and s["reward_mean"] == pytest.approx(result["reward"])


def test_run_episode_is_reproducible_with_seed():
    env = make(random_pain_prob=0.1)
    a = run_episode(env, zero_residual_policy(env), seed=5)
    b = run_episode(env, zero_residual_policy(env), seed=5)
    assert a["reward"] == b["reward"]
    np.testing.assert_array_equal(a["pain"], b["pain"])

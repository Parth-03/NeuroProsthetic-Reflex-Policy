"""Environment construction."""

from reflexrl.wrapper import ReflexResidualElbow

ENV_ID = "myoElbowPose1D6MRandom-v0"

# Wrapper settings used for the main experiments.
DEFAULT_WRAPPER_KWARGS = dict(
    residual_scale=0.2,
    dead_zone=0.02,
    reflex_gain=3.0,
    pain_threshold=0.9,
    withdraw_steps=None,
    safe_offset=0.5,
    lambda_pain=2.0,
    random_pain_prob=0.1,
    random_pain_steps=5,
)


def make_base_env(env_id=ENV_ID):
    """Creates the raw MyoSuite environment."""
    from myosuite.utils import gym as myo_gym

    return myo_gym.make(env_id)


def make_env(env_id=ENV_ID, seed=None, **wrapper_kwargs):
    """Creates the MyoSuite elbow wrapped in :class:`ReflexResidualElbow`.

    Keyword arguments override :data:`DEFAULT_WRAPPER_KWARGS`.
    """
    kwargs = dict(DEFAULT_WRAPPER_KWARGS)
    kwargs.update(wrapper_kwargs)
    return ReflexResidualElbow(make_base_env(env_id), seed=seed, **kwargs)

"""Dead-zone stretch reflex for an antagonistic flexor/extensor muscle group."""

import numpy as np

FLEXOR_INDICES = (0, 1, 2)
EXTENSOR_INDICES = (3, 4, 5)
NUM_MUSCLES = len(FLEXOR_INDICES) + len(EXTENSOR_INDICES)


def reflex_controller(q, q_target, k_p=5.0, dead_zone=0.02):
    """Stretch-like reflex with a dead zone.

    * If ``|q_target - q| < dead_zone`` no muscle fires.
    * Otherwise the agonist group (flexors when the joint must flex, extensors
      when it must extend) fires at ``clip(k_p * (|error| - dead_zone), 0, 1)``
      and the antagonists stay silent.

    Args:
        q: Current joint angle (rad).
        q_target: Target joint angle (rad).
        k_p: Reflex gain.
        dead_zone: Error below which the reflex is silent (rad).

    Returns:
        Muscle activations, shape (6,), in [0, 1].
    """
    error = q_target - q  # > 0: flex, < 0: extend
    action = np.zeros(NUM_MUSCLES, dtype=np.float32)

    if abs(error) < dead_zone:
        return action

    level = np.clip(k_p * (abs(error) - dead_zone), 0.0, 1.0)
    agonists = FLEXOR_INDICES if error > 0 else EXTENSOR_INDICES
    action[list(agonists)] = level
    return action

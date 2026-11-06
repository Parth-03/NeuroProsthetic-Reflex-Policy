"""Reflex + residual reinforcement learning for a musculoskeletal elbow.

A hand-designed stretch reflex drives six elbow muscles toward a target angle,
and a PPO policy learns a bounded residual on top of it. A pain signal and a
protective withdrawal response are layered on top of the controller.
"""

from reflexrl.observations import ERROR_IDX, build_state, extract_q_qdot_qtarget
from reflexrl.reflex import EXTENSOR_INDICES, FLEXOR_INDICES, reflex_controller
from reflexrl.reward import compute_custom_reward
from reflexrl.wrapper import ReflexResidualElbow

__all__ = [
    "ERROR_IDX",
    "EXTENSOR_INDICES",
    "FLEXOR_INDICES",
    "ReflexResidualElbow",
    "build_state",
    "compute_custom_reward",
    "extract_q_qdot_qtarget",
    "reflex_controller",
]

__version__ = "0.1.0"

"""Observation parsing for ``myoElbowPose1D6MRandom-v0``.

Raw observation layout (9 values):

    obs[0]    qpos      elbow angle (rad)
    obs[1]    qvel      elbow angular velocity
    obs[2:8]  act       activations of the 6 muscles
    obs[8]    pose_err  target angle - qpos
"""

import numpy as np

# Layout of the state seen by the RL policy.
STATE_LABELS = ("q", "qdot", "q_ctrl_target", "error_ctrl", "pain", "in_withdrawal")
ERROR_IDX = STATE_LABELS.index("error_ctrl")
STATE_DIM = len(STATE_LABELS)


def extract_q_qdot_qtarget(obs):
    """Returns (q, qdot, q_target) from a raw environment observation."""
    q = float(obs[0])
    qdot = float(obs[1])
    pose_err = float(obs[8])
    return q, qdot, q + pose_err


def build_state(raw_obs, q_ctrl_target, pain, in_withdrawal):
    """Builds the policy state ``[q, qdot, q_ctrl_target, error_ctrl, pain, in_withdrawal]``.

    ``q_ctrl_target`` is the target the controller is currently tracking: the
    environment's target normally, or the safe target during withdrawal.
    """
    q, qdot, _ = extract_q_qdot_qtarget(raw_obs)
    error_ctrl = q_ctrl_target - q
    return np.array(
        [q, qdot, q_ctrl_target, error_ctrl, float(pain), float(in_withdrawal)],
        dtype=np.float32,
    )

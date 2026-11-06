"""Shaped reward used to train the residual policy."""

import numpy as np


def compute_custom_reward(q, qdot, q_ctrl_target, muscle_act, pain,
                          w_pos=10.0, w_vel=0.1, w_act=0.001, lambda_pain=2.0):
    """Negative weighted sum of tracking error, velocity, effort and pain.

        r = -w_pos * (q_ctrl_target - q)^2
            - w_vel * qdot^2
            - w_act * sum(muscle_act^2)
            - lambda_pain * pain
    """
    pos_err = q_ctrl_target - q
    r_pos = -w_pos * pos_err ** 2
    r_vel = -w_vel * qdot ** 2
    r_act = -w_act * float(np.sum(np.square(muscle_act)))
    r_pain = -lambda_pain * float(pain)
    return float(r_pos + r_vel + r_act + r_pain)

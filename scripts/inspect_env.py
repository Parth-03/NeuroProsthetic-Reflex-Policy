"""Print the MyoSuite elbow's spaces and step the wrapped environment.

    python scripts/inspect_env.py
    python scripts/inspect_env.py --steps 20 --residual 0.2
"""

import argparse

import numpy as np

from reflexrl import extract_q_qdot_qtarget
from reflexrl.config import add_wrapper_args, wrapper_kwargs_from_args
from reflexrl.env import ENV_ID, make_base_env
from reflexrl.wrapper import ReflexResidualElbow


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--residual", type=float, default=0.0,
                   help="constant residual added to every muscle")
    p.add_argument("--seed", type=int, default=0)
    add_wrapper_args(p)
    args = p.parse_args()

    base_env = make_base_env(args.env_id or ENV_ID)
    print("Raw action space:     ", base_env.action_space)
    print("Raw observation space:", base_env.observation_space)
    obs, _ = base_env.reset(seed=args.seed)
    print("Raw observation:      ", np.round(obs, 4))
    print("(q, qdot, q_target):  ", tuple(round(v, 4) for v in extract_q_qdot_qtarget(obs)))

    env = ReflexResidualElbow(base_env, **wrapper_kwargs_from_args(args))
    print("\nPolicy action space:     ", env.action_space)
    print("Policy observation space:", env.observation_space)
    state, _ = env.reset(seed=args.seed)
    print("Initial state [q, qdot, q_ctrl, err, pain, withdraw]:", np.round(state, 4))

    residual = np.full(env.action_space.shape, args.residual, dtype=np.float32)
    for t in range(args.steps):
        state, reward, terminated, truncated, info = env.step(residual)
        print(f"t={t:03d}  state={np.round(state, 3)}  reward={reward:8.3f}  "
              f"max|act|={np.max(np.abs(info['muscle_act'])):.3f}  "
              f"pain={info['pain']}  withdraw={info['in_withdrawal']}")
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()

"""Run the bare stretch reflex on the raw elbow (no RL, no pain).

    python scripts/reflex_only.py --out figures/reflex_only_raw.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reflexrl.env import ENV_ID, make_base_env  # noqa: E402
from reflexrl.evaluation import run_reflex_only_raw  # noqa: E402
from reflexrl.plotting import plot_reflex_only_raw  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default=ENV_ID)
    p.add_argument("--k-p", type=float, default=5.0, help="reflex gain")
    p.add_argument("--dead-zone", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("figures", "reflex_only_raw.png"))
    args = p.parse_args()

    q, q_target, error = run_reflex_only_raw(make_base_env(args.env_id),
                                             max_steps=args.max_steps, k_p=args.k_p,
                                             dead_zone=args.dead_zone, seed=args.seed)
    print(f"Steps: {len(q)}   final |error|: {abs(error[-1]):.4f} rad")

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_reflex_only_raw(q, q_target, error, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

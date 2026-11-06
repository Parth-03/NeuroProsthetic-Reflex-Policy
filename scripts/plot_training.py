"""Plot pain and withdrawal per episode from a training run.

    python scripts/plot_training.py --run runs/reflex_rl
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reflexrl.plotting import moving_average, plot_training_pain  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder containing episodes.csv")
    p.add_argument("--window", type=int, default=50, help="moving-average window")
    p.add_argument("--show-withdraw", action="store_true")
    p.add_argument("--out", default=None, help="default: <run>/training_pain.png")
    args = p.parse_args()

    with open(os.path.join(args.run, "episodes.csv")) as f:
        rows = list(csv.DictReader(f))
    pain = [int(r["pain_steps"]) for r in rows]
    withdraw = [int(r["withdraw_steps"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    plot_training_pain(pain, withdraw if args.show_withdraw else None,
                       window=args.window, ax=axes[0])
    axes[1].plot(reward, alpha=0.3, linewidth=0.8, label="Reward / episode")
    smooth = moving_average(reward, args.window)
    axes[1].plot(range(len(reward) - len(smooth), len(reward)), smooth, linewidth=2,
                 label=f"{args.window}-episode mean")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Episode reward")
    axes[1].set_title("Reward during training")
    axes[1].legend()
    fig.tight_layout()

    out = args.out or os.path.join(args.run, "training_pain.png")
    fig.savefig(out, dpi=150)
    n = min(100, len(pain))
    print(f"Episodes: {len(pain)}. Mean pain steps: first {n} = "
          f"{sum(pain[:n]) / n:.1f}, last {n} = {sum(pain[-n:]) / n:.1f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

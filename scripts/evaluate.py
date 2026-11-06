"""Compare reflex-only control with reflex + learned residual.

Both controllers run on the same seeds, so each pair of episodes shares the
target angle and the random pain events. Uses the wrapper settings saved with
the run unless overridden on the command line.

    python scripts/evaluate.py --model runs/reflex_rl --episodes 10
    python scripts/evaluate.py --model runs/reflex_rl --pain-threshold 0.8
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from reflexrl.config import (add_wrapper_args, load_config,  # noqa: E402
                             resolve_model_path, wrapper_kwargs_from_args)
from reflexrl.env import ENV_ID, make_env  # noqa: E402
from reflexrl.evaluation import compare_controllers, summarize  # noqa: E402
from reflexrl.plotting import plot_episode_trace, plot_reward_comparison  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="run folder or model .zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000,
                   help="first evaluation seed (kept apart from training seeds)")
    p.add_argument("--traces", type=int, default=3,
                   help="number of per-episode trace plots to save")
    p.add_argument("--out", default=None, help="output folder (default: <run>/eval)")
    add_wrapper_args(p)
    args = p.parse_args()

    run_dir, model_file = resolve_model_path(args.model)
    config = load_config(run_dir)
    env_id = args.env_id or config.get("env_id", ENV_ID)
    wrapper_kwargs = wrapper_kwargs_from_args(args, base=config.get("wrapper"))
    out = args.out or os.path.join(run_dir, "eval")
    os.makedirs(out, exist_ok=True)

    env = make_env(env_id, **wrapper_kwargs)
    model = PPO.load(model_file, device="cpu")
    reflex, rl = compare_controllers(env, model, num_episodes=args.episodes,
                                     max_steps=args.max_steps, seed=args.seed)

    print(f"{'controller':<12} {'reward':>18} {'pain steps':>14} {'withdraw steps':>16}")
    for label, results in [("Reflex only", reflex), ("Reflex + RL", rl)]:
        s = summarize(results)
        print(f"{label:<12} {s['reward_mean']:>9.1f} ± {s['reward_std']:<6.1f} "
              f"{s['pain_mean']:>7.1f} ± {s['pain_std']:<4.1f} "
              f"{s['withdraw_mean']:>8.1f} ± {s['withdraw_std']:<4.1f}")

    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "controller", "reward", "pain_steps",
                         "withdraw_steps", "length"])
        for i, (a, b) in enumerate(zip(reflex, rl)):
            for label, r in [("reflex", a), ("reflex_rl", b)]:
                writer.writerow([i, args.seed + i, label, f"{r['reward']:.4f}",
                                 int(r["pain"].sum()), int(r["withdraw"].sum()),
                                 len(r["error"])])

    fig, ax = plt.subplots(figsize=(6.5, 4))
    plot_reward_comparison([r["reward"] for r in reflex], [r["reward"] for r in rl], ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reward_comparison.png"), dpi=150)
    plt.close(fig)

    for i in range(min(args.traces, args.episodes)):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_episode_trace(reflex[i], rl[i], ax=ax,
                           title=f"Error, pain and withdrawal (seed {args.seed + i})")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trace_episode_{i}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()

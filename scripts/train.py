"""Train the residual policy with PPO on top of the reflex.

Saves to ``runs/<name>/``: ``model.zip``, ``config.json`` (wrapper settings)
and ``episodes.csv`` (per-episode reward, pain and withdrawal).

    python scripts/train.py --name reflex_rl --timesteps 200000
    python scripts/train.py --name no_random_pain --random-pain-prob 0
"""

import argparse
import os

from stable_baselines3 import PPO

from reflexrl.callbacks import PainLoggingCallback
from reflexrl.config import add_wrapper_args, save_config, wrapper_kwargs_from_args
from reflexrl.env import ENV_ID, make_env


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="reflex_rl")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="PPO with an MLP policy runs faster on CPU than GPU")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--tensorboard", action="store_true",
                   help="also log to runs/<name>/tb")
    p.add_argument("--verbose", type=int, default=1)
    add_wrapper_args(p)
    args = p.parse_args()

    run_dir = os.path.join(args.runs_dir, args.name)
    env_id = args.env_id or ENV_ID
    wrapper_kwargs = wrapper_kwargs_from_args(args)
    save_config(run_dir, dict(env_id=env_id, seed=args.seed, timesteps=args.timesteps,
                              wrapper=wrapper_kwargs))

    env = make_env(env_id, seed=args.seed, **wrapper_kwargs)
    model = PPO("MlpPolicy", env, seed=args.seed, device=args.device,
                verbose=args.verbose,
                tensorboard_log=os.path.join(run_dir, "tb") if args.tensorboard else None)
    callback = PainLoggingCallback(verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=callback)

    model.save(os.path.join(run_dir, "model"))
    callback.save_csv(os.path.join(run_dir, "episodes.csv"))

    n = len(callback.episodes)
    tail = callback.episodes[-min(100, n):] if n else []
    if tail:
        mean = lambda key: sum(e[key] for e in tail) / len(tail)  # noqa: E731
        print(f"\nEpisodes: {n}. Last {len(tail)} episodes: "
              f"reward {mean('reward'):.1f}, pain steps {mean('pain_steps'):.1f}, "
              f"withdrawal steps {mean('withdraw_steps'):.1f}")
    print(f"Saved run to {run_dir}/")


if __name__ == "__main__":
    main()

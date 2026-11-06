"""Record one episode as a video.

By default the forearm is drawn as a 2D schematic, which needs no OpenGL.
With ``--mujoco`` the MyoSuite scene is rendered instead (set
``MUJOCO_GL=egl`` or ``osmesa`` on a headless machine).

    python scripts/make_video.py --model runs/reflex_rl
    python scripts/make_video.py --model runs/reflex_rl --reflex-only --out videos/reflex_only.mp4
    MUJOCO_GL=egl python scripts/make_video.py --model runs/reflex_rl --mujoco
"""

import argparse
import os

from reflexrl.config import (add_wrapper_args, load_config, resolve_model_path,
                             wrapper_kwargs_from_args)
from reflexrl.env import ENV_ID, make_env
from reflexrl.evaluation import model_policy, run_episode, zero_residual_policy
from reflexrl.video import render_mujoco_frame, schematic_frames, write_video


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="run folder or model .zip")
    p.add_argument("--reflex-only", action="store_true",
                   help="record the reflex without the learned residual")
    p.add_argument("--mujoco", action="store_true", help="render the MuJoCo scene")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", default=None)
    add_wrapper_args(p)
    args = p.parse_args()

    if not args.reflex_only and args.model is None:
        p.error("--model is required unless --reflex-only is given")

    config = {}
    if args.model:
        run_dir, model_file = resolve_model_path(args.model)
        config = load_config(run_dir)
    env_id = args.env_id or config.get("env_id", ENV_ID)
    env = make_env(env_id, **wrapper_kwargs_from_args(args, base=config.get("wrapper")))

    if args.reflex_only:
        policy, label = zero_residual_policy(env), "reflex_only"
    else:
        from stable_baselines3 import PPO
        policy, label = model_policy(PPO.load(model_file, device="cpu")), "reflex_rl"

    frame_fn = None
    if args.mujoco:
        def frame_fn(e):
            return render_mujoco_frame(e, camera_id=args.camera_id)

    traj = run_episode(env, policy, max_steps=args.max_steps, seed=args.seed,
                       frame_fn=frame_fn)
    frames = traj["frames"] if args.mujoco else schematic_frames(traj)
    kind = "mujoco" if args.mujoco else "schematic"
    out = args.out or os.path.join("videos", f"elbow_{label}_{kind}.mp4")
    write_video(out, frames, fps=args.fps)
    print(f"Reward {traj['reward']:.1f}, pain steps {int(traj['pain'].sum())}, "
          f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()

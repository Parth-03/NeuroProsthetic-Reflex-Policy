"""Episode videos: a 2D schematic of the forearm, or MuJoCo renders."""

import os

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _figure_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def schematic_frames(traj, length=1.0, size=4):
    """Draws the forearm as a rotating segment for every step of a trajectory.

    Args:
        traj: Output of :func:`reflexrl.evaluation.run_episode`.
        length: Drawn forearm length.
        size: Figure size in inches.

    Returns:
        List of RGB frames.
    """
    frames = []
    for t in range(len(traj["q"])):
        fig, ax = plt.subplots(figsize=(size, size))
        withdrawing = traj["withdraw"][t] == 1
        ax.set_facecolor("#ffeeee" if withdrawing else "white")

        x, y = length * np.cos(traj["q"][t]), length * np.sin(traj["q"][t])
        ax.plot([0, x], [0, y], linewidth=4, color="tab:blue")
        ax.scatter([0], [0], s=50, color="tab:blue")
        ax.scatter([x], [y], s=50, color="tab:blue")

        q_env, q_ctrl = traj["q_env_target"][t], traj["q_ctrl_target"][t]
        ax.scatter([length * np.cos(q_env)], [length * np.sin(q_env)], s=40,
                   marker="x", color="tab:green", label="Task target")
        ax.scatter([length * np.cos(q_ctrl)], [length * np.sin(q_ctrl)], s=40,
                   marker="o", facecolors="none", edgecolors="tab:orange",
                   label="Control target")
        if traj["pain"][t] == 1:
            ax.scatter([x], [y], s=120, edgecolors="red", facecolors="none",
                       linewidths=2, label="Pain")

        ax.set_xlim(-1.2 * length, 1.2 * length)
        ax.set_ylim(-1.2 * length, 1.2 * length)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {t}   pain={traj['pain'][t]}   withdrawal={traj['withdraw'][t]}")
        ax.legend(loc="lower left", fontsize=7)
        frames.append(_figure_to_rgb(fig))
        plt.close(fig)
    return frames


def render_mujoco_frame(env, width=480, height=480, camera_id=0):
    """Renders the MyoSuite scene offscreen.

    MyoSuite environments do not implement Gymnasium's ``render()``, so the
    simulator's own renderer is used. Requires an offscreen OpenGL backend
    (e.g. ``MUJOCO_GL=egl``).
    """
    sim = env.unwrapped.sim
    return sim.renderer.render_offscreen(width=width, height=height, camera_id=camera_id)


def write_video(path, frames, fps=20):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path

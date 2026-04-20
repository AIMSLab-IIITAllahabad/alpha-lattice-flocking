"""Demo: Algorithm 2 (free-flocking in 2D with gamma navigation).

Reproduces paper Fig. 7:  n=150 alpha-agents, Gaussian initial positions,
uniform initial velocities, dynamic gamma-agent pulls them along.

Modes:
    python3 demo_algorithm2_flocking.py                       # live animation
    python3 demo_algorithm2_flocking.py --save out.mp4 --headless
    python3 demo_algorithm2_flocking.py --snapshots           # save paper-Fig.7 grid
    python3 demo_algorithm2_flocking.py --snapshots --save paper_fig7_repro.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from flocking import FlockSim


PAPER_SNAPSHOT_TIMES = [0.0, 0.57, 2.37, 3.57, 4.77, 8.97]  # matches Fig. 7


def build_edges(q, r):
    n = len(q)
    diffs = q[None, :, :] - q[:, None, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    mask = dists[iu] < r
    return np.stack([q[iu[0][mask]], q[iu[1][mask]]], axis=1)


def build_sim(n=150, seed=0):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(scale=35.0, size=(n, 2))
    p0 = rng.uniform(-2.0, -1.0, size=(n, 2))
    p_d = np.array([5.0, 0.0])
    return FlockSim(
        q0, p0, algorithm=2,
        target_pos=lambda t: p_d * t, target_vel=p_d,
        c1_alpha=3.0, c1_gamma=0.3,
    )


def snapshots(outfile="paper_fig7_repro.png", n=150, seed=0, dt=0.03,
              times=None):
    times = list(PAPER_SNAPSHOT_TIMES if times is None else times)
    sim = build_sim(n=n, seed=seed)
    recorded = {}
    targets = {int(round(tt / dt)): tt for tt in times}
    for k in range(int(round(max(times) / dt)) + 1):
        if k in targets:
            recorded[targets[k]] = (sim.q.copy(), sim.p.copy())
        sim.step(dt, k * dt)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, tt in zip(axes.flat, times):
        q, p = recorded[tt]
        ax.set_aspect("equal")
        ax.add_collection(LineCollection(build_edges(q, sim.r),
                                         colors="lightgray", linewidths=0.5))
        ax.scatter(q[:, 0], q[:, 1], marker="^", s=12, c="black")
        cx, cy = q.mean(0)
        R = max(50.0, 1.3 * np.max(np.linalg.norm(q - q.mean(0), axis=1)))
        ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
        ax.set_title(f"t={tt} sec")
    fig.suptitle("Paper Fig. 7:  2-D flocking for n=150 (Algorithm 2)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def live(n=150, steps=600, dt=0.02, seed=0, save=None, live_show=True):
    sim = build_sim(n=n, seed=seed)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    scat = ax.scatter(sim.q[:, 0], sim.q[:, 1], s=18, c="black", zorder=3)
    quiv = ax.quiver(sim.q[:, 0], sim.q[:, 1], sim.p[:, 0], sim.p[:, 1],
                     color="steelblue", scale=60, width=0.003, zorder=2)
    edges = LineCollection([], colors="lightgray", linewidths=0.7, zorder=1)
    ax.add_collection(edges)
    target_dot, = ax.plot([], [], "rx", markersize=10, zorder=4)
    title = ax.set_title("")

    def update(k):
        t = k * dt
        sim.step(dt, t=t)
        qr = sim.gamma_state(t)[0]
        scat.set_offsets(sim.q)
        quiv.set_offsets(sim.q)
        quiv.set_UVC(sim.p[:, 0], sim.p[:, 1])
        edges.set_segments(build_edges(sim.q, sim.r))
        target_dot.set_data([qr[0]], [qr[1]])
        cx, cy = sim.q.mean(axis=0)
        R = max(40.0, 1.2 * sim.cohesion_radius())
        ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
        title.set_text(
            f"Algorithm 2 - t={t:5.2f}s  n={n}  "
            f"cohesion={sim.cohesion_radius():.1f}"
        )
        return scat, quiv, edges, target_dot, title

    anim = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
    if save:
        anim.save(save, fps=30, dpi=120)
        print(f"saved animation to {save}")
    if live_show:
        plt.show()
    return sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshots", action="store_true",
                    help="save paper-Fig.7 snapshot grid instead of animation")
    args = ap.parse_args()
    if args.snapshots:
        out = args.save or "paper_fig7_repro.png"
        print(f"wrote {snapshots(out, n=args.n, seed=args.seed, dt=args.dt)}")
    else:
        live(args.n, args.steps, args.dt, args.seed, args.save, not args.headless)

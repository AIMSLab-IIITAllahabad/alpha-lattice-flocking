"""Demo: Algorithm 1 (alpha,alpha protocol) - fragmentation.

Reproduces paper Fig. 8:  n=40 alpha-agents with no group objective
typically fragment into multiple disconnected components.

Modes:
    python3 demo_algorithm1_fragmentation.py                 # live animation
    python3 demo_algorithm1_fragmentation.py --snapshots     # paper-Fig.8 grid
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from flocking import FlockSim


PAPER_SNAPSHOT_TIMES = [0.0, 1.77, 2.97, 17.97]  # matches Fig. 8


def _edges(q, r):
    n = len(q)
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    iu = np.triu_indices(n, 1)
    m = d[iu] < r
    return np.stack([q[iu[0][m]], q[iu[1][m]]], axis=1)


def build_sim(n=40, seed=3):
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(0.0, 90.0, size=(n, 2))
    p0 = rng.uniform(-1.0, 1.0, size=(n, 2))
    return FlockSim(q0, p0, algorithm=1, c1_alpha=3.0)


def snapshots(outfile="paper_fig8_repro.png", n=40, seed=3, dt=0.03,
              times=None):
    times = list(PAPER_SNAPSHOT_TIMES if times is None else times)
    sim = build_sim(n=n, seed=seed)
    recorded = {}
    targets = {int(round(tt / dt)): tt for tt in times}
    for k in range(int(round(max(times) / dt)) + 1):
        if k in targets:
            recorded[targets[k]] = (sim.q.copy(), sim.p.copy())
        sim.step(dt, k * dt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, tt in zip(axes.flat, times):
        q, p = recorded[tt]
        ax.set_aspect("equal")
        ax.add_collection(LineCollection(_edges(q, sim.r),
                                         colors="lightgray", linewidths=0.5))
        ax.scatter(q[:, 0], q[:, 1], marker="^", s=20, c="black")
        cx, cy = q.mean(0)
        R = max(60.0, 1.3 * np.max(np.linalg.norm(q - q.mean(0), axis=1)))
        ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
        ax.set_title(f"t={tt} sec")
    fig.suptitle("Paper Fig. 8:  Fragmentation for n=40 (Algorithm 1)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def live(n=40, steps=900, dt=0.02, seed=3, save=None, live_show=True):
    sim = build_sim(n=n, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect("equal")
    scat = ax.scatter(sim.q[:, 0], sim.q[:, 1], s=18, c="black", zorder=3)
    quiv = ax.quiver(sim.q[:, 0], sim.q[:, 1], sim.p[:, 0], sim.p[:, 1],
                     color="firebrick", scale=40, width=0.003, zorder=2)
    edges = LineCollection([], colors="lightgray", linewidths=0.7, zorder=1)
    ax.add_collection(edges)
    title = ax.set_title("")

    def update(k):
        t = k * dt
        sim.step(dt, t=t)
        scat.set_offsets(sim.q)
        quiv.set_offsets(sim.q)
        quiv.set_UVC(sim.p[:, 0], sim.p[:, 1])
        edges.set_segments(_edges(sim.q, sim.r))
        cx, cy = sim.q.mean(axis=0)
        R = max(60.0, 1.2 * sim.cohesion_radius())
        ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R)
        title.set_text(
            f"Algorithm 1 - t={t:5.2f}s  n={n}  "
            f"cohesion={sim.cohesion_radius():.1f}"
        )
        return scat, quiv, edges, title

    anim = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
    if save:
        anim.save(save, fps=30, dpi=120)
        print(f"saved animation to {save}")
    if live_show:
        plt.show()
    return sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshots", action="store_true",
                    help="save paper-Fig.8 snapshot grid instead of animation")
    args = ap.parse_args()
    if args.snapshots:
        out = args.save or "paper_fig8_repro.png"
        print(f"wrote {snapshots(out, n=args.n, seed=args.seed, dt=args.dt)}")
    else:
        live(args.n, args.steps, args.dt, args.seed, args.save, not args.headless)

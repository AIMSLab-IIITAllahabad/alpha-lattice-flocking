"""Demo: Squeezing maneuver (paper Fig. 11).

n=150 alpha-agents pulled through a narrow pathway between two large
circular obstacles. A special case of the escape-panic phenomenon.

Paper obstacle matrix:  y1=(150,30) R=25,  y2=(150,100) R=25.

Modes:
    python3 demo_squeezing.py                # live animation
    python3 demo_squeezing.py --snapshots    # paper-Fig.11 grid
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from flocking import FlockSim, SphereObstacle


PAPER_OBSTACLES = [
    ((150.0, 30.0), 25.0),
    ((150.0, 100.0), 25.0),
]
PAPER_SNAPSHOT_TIMES = [0.0, 6.0, 15.48, 22.98, 27.98, 43.98]  # Fig. 11


def _edges(q, r):
    n = len(q)
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    iu = np.triu_indices(n, 1)
    m = d[iu] < r
    return np.stack([q[iu[0][m]], q[iu[1][m]]], axis=1)


def build_sim(n=150, seed=5):
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(low=[0.0, 0.0], high=[120.0, 120.0], size=(n, 2))
    p0 = np.zeros_like(q0)
    obstacles = [SphereObstacle(y, R) for (y, R) in PAPER_OBSTACLES]
    p_d = np.array([6.0, 0.0])
    return FlockSim(
        q0, p0, algorithm=3,
        target_pos=lambda t: np.array([0.0, 60.0]) + p_d * t, target_vel=p_d,
        c1_alpha=3.0, c1_gamma=2.0, c1_beta=1500.0,
        obstacles=obstacles,
    )


def snapshots(outfile="paper_fig11_repro.png", n=150, seed=5, dt=0.02,
              times=None):
    times = list(PAPER_SNAPSHOT_TIMES if times is None else times)
    sim = build_sim(n=n, seed=seed)
    recorded = {}
    targets = {int(round(tt / dt)): tt for tt in times}
    for k in range(int(round(max(times) / dt)) + 1):
        if k in targets:
            recorded[targets[k]] = (sim.q.copy(), sim.p.copy())
        sim.step(dt, k * dt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, tt in zip(axes.flat, times):
        q, _ = recorded[tt]
        ax.set_aspect("equal")
        for (y, R) in PAPER_OBSTACLES:
            ax.add_patch(Circle(y, R, color="black", zorder=4))
        ax.add_collection(LineCollection(_edges(q, sim.r),
                                         colors="lightgray", linewidths=0.5))
        ax.scatter(q[:, 0], q[:, 1], marker="^", s=12, c="black")
        ax.set_xlim(-30, 400); ax.set_ylim(-40, 160)
        ax.set_title(f"t={tt} sec")
    fig.suptitle("Paper Fig. 11:  Squeezing maneuver, n=150 (Algorithm 3)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def live(n=150, steps=2500, dt=0.02, seed=5, save=None, live_show=True):
    sim = build_sim(n=n, seed=seed)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect("equal")
    for (y, R) in PAPER_OBSTACLES:
        ax.add_patch(Circle(y, R, color="black", zorder=4))
    scat = ax.scatter(sim.q[:, 0], sim.q[:, 1], s=14, c="black", zorder=3)
    quiv = ax.quiver(sim.q[:, 0], sim.q[:, 1], sim.p[:, 0], sim.p[:, 1],
                     color="steelblue", scale=80, width=0.002, zorder=2)
    edges = LineCollection([], colors="lightgray", linewidths=0.5, zorder=1)
    ax.add_collection(edges)
    target_dot, = ax.plot([], [], "rx", markersize=9, zorder=5)
    title = ax.set_title("")

    def update(k):
        t = k * dt
        sim.step(dt, t=t)
        qr = sim.gamma_state(t)[0]
        scat.set_offsets(sim.q)
        quiv.set_offsets(sim.q)
        quiv.set_UVC(sim.p[:, 0], sim.p[:, 1])
        edges.set_segments(_edges(sim.q, sim.r))
        target_dot.set_data([qr[0]], [qr[1]])
        cx, _ = sim.q.mean(axis=0)
        ax.set_xlim(cx - 100, cx + 120); ax.set_ylim(-20, 150)
        title.set_text(
            f"Squeezing  t={t:5.2f}s  n={n}  "
            f"C(t)={sim.relative_connectivity():.2f}"
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
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshots", action="store_true",
                    help="save paper-Fig.11 snapshot grid instead of animation")
    args = ap.parse_args()
    if args.snapshots:
        out = args.save or "paper_fig11_repro.png"
        print(f"wrote {snapshots(out, n=args.n, seed=args.seed, dt=args.dt)}")
    else:
        live(args.n, args.steps, args.dt, args.seed, args.save, not args.headless)

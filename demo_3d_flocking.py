"""Demo: 3-D flocking / automated rendezvous (paper Fig. 9).

n=50 UAV-like alpha-agents in R^3, gamma-agent pulls them along
p_d = (3,2,1). They should self-assemble into a 3-D alpha-lattice.

Modes:
    python3 demo_3d_flocking.py                 # live animation
    python3 demo_3d_flocking.py --snapshots     # paper-Fig.9 grid
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from flocking import FlockSim, GammaAgent


PAPER_SNAPSHOT_TIMES = [0.0, 1.17, 2.57, 7.17]  # matches Fig. 9


def _edges3d(q, r):
    d = np.linalg.norm(q[None, :, :] - q[:, None, :], axis=-1)
    n = len(q)
    iu = np.triu_indices(n, 1)
    m = d[iu] < r
    return np.stack([q[iu[0][m]], q[iu[1][m]]], axis=1)


def build_sim(n=50, seed=1):
    rng = np.random.default_rng(seed)
    q0 = rng.normal(scale=15.0, size=(n, 3))
    p0 = rng.uniform(-0.5, 0.5, size=(n, 3))
    gamma = GammaAgent(q0=np.zeros(3), p0=np.array([3.0, 2.0, 1.0]))
    return FlockSim(q0, p0, algorithm=2,
                    target_pos=gamma, target_vel=gamma.p,
                    c1_alpha=3.0, c1_gamma=0.3)


def snapshots(outfile="paper_fig9_repro.png", n=50, seed=1, dt=0.03,
              times=None):
    times = list(PAPER_SNAPSHOT_TIMES if times is None else times)
    sim = build_sim(n=n, seed=seed)
    recorded = {}
    targets = {int(round(tt / dt)): tt for tt in times}
    for k in range(int(round(max(times) / dt)) + 1):
        if k in targets:
            recorded[targets[k]] = (sim.q.copy(),)
        sim.step(dt, k * dt)

    fig = plt.figure(figsize=(14, 10))
    for idx, tt in enumerate(times, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        q = recorded[tt][0]
        segs = _edges3d(q, sim.r)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], marker="^", s=20, c="black")
        for a, b in segs[:400]:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color="gray", linewidth=0.4)
        cx, cy, cz = q.mean(0)
        R = max(20.0, 1.2 * np.max(np.linalg.norm(q - q.mean(0), axis=1)))
        ax.set_xlim(cx - R, cx + R)
        ax.set_ylim(cy - R, cy + R)
        ax.set_zlim(cz - R, cz + R)
        ax.set_title(f"t={tt} sec")
    fig.suptitle("Paper Fig. 9:  3-D flocking for n=50 (Algorithm 2)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile


def live(n=50, steps=800, dt=0.03, seed=1, save=None, live_show=True):
    sim = build_sim(n=n, seed=seed)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    pts = ax.scatter(sim.q[:, 0], sim.q[:, 1], sim.q[:, 2], s=15, c="black")
    line_collection = []
    title = ax.set_title("")

    def update(k):
        t = k * dt
        sim.step(dt, t=t)
        pts._offsets3d = (sim.q[:, 0], sim.q[:, 1], sim.q[:, 2])
        for ln in line_collection:
            ln.remove()
        line_collection.clear()
        segs = _edges3d(sim.q, sim.r)
        if len(segs) > 600:
            segs = segs[:: max(1, len(segs) // 600)]
        for (a, b) in segs:
            ln, = ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                          color="lightgray", linewidth=0.5)
            line_collection.append(ln)
        cx, cy, cz = sim.q.mean(axis=0)
        R = max(25.0, 1.2 * sim.cohesion_radius())
        ax.set_xlim(cx - R, cx + R)
        ax.set_ylim(cy - R, cy + R)
        ax.set_zlim(cz - R, cz + R)
        title.set_text(
            f"3-D flocking  t={t:5.2f}s  n={n}  "
            f"cohesion={sim.cohesion_radius():.1f}"
        )
        return (pts, title, *line_collection)

    anim = FuncAnimation(fig, update, frames=steps, interval=20, blit=False)
    if save:
        anim.save(save, fps=30, dpi=120)
        print(f"saved animation to {save}")
    if live_show:
        plt.show()
    return sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshots", action="store_true",
                    help="save paper-Fig.9 snapshot grid instead of animation")
    args = ap.parse_args()
    if args.snapshots:
        out = args.save or "paper_fig9_repro.png"
        print(f"wrote {snapshots(out, n=args.n, seed=args.seed, dt=args.dt)}")
    else:
        live(args.n, args.steps, args.dt, args.seed, args.save, not args.headless)

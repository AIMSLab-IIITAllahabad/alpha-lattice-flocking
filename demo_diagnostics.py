"""Demo: alpha-flocking verification curves (paper Fig. 12).

Tracks the four Sec. IX.A indicators over the course of two simulations:
    C(t)       = rank(L(q)) / (n-1)      - relative connectivity
    R(t)       = max_i ||q_i - q_c||     - cohesion radius
    K_tilde(t) = K(v) / n                - normalized velocity mismatch
    E_tilde(t) = E(q) / d^2               - normalized deviation energy

Top row: Algorithm 2 (n=150) -> connectivity snaps to 1, cohesion shrinks,
velocity mismatch and deviation energy decay.

Bottom row: Algorithm 1 (n=50) -> fragmentation: connectivity stays low,
cohesion grows, mismatch never dies, energy stays high.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from flocking import FlockSim


def track(sim, steps, dt):
    C, R, K, E = [], [], [], []
    for k in range(steps):
        sim.step(dt, t=k * dt)
        C.append(sim.relative_connectivity())
        R.append(sim.cohesion_radius())
        K.append(sim.normalized_velocity_mismatch())
        E.append(sim.normalized_deviation_energy())
    return np.array(C), np.array(R), np.array(K), np.array(E)


def run(steps=400, dt=0.03, save=None, live=True):
    # ---- (a) Algorithm 2 -> expect flocking ----
    rng = np.random.default_rng(0)
    n_flock = 150
    q0 = rng.normal(scale=30.0, size=(n_flock, 2))
    p0 = rng.uniform(-2.0, -1.0, size=(n_flock, 2))
    p_d = np.array([5.0, 0.0])
    sim_f = FlockSim(q0, p0, algorithm=2,
                     target_pos=lambda t: p_d * t, target_vel=p_d,
                     c1_alpha=3.0, c1_gamma=0.3)
    Cf, Rf, Kf, Ef = track(sim_f, steps, dt)

    # ---- (b) Algorithm 1 -> expect fragmentation ----
    rng = np.random.default_rng(3)
    n_frag = 50
    q0 = rng.uniform(0.0, 90.0, (n_frag, 2))
    p0 = rng.uniform(-1.0, 1.0, (n_frag, 2))
    sim_g = FlockSim(q0, p0, algorithm=1, c1_alpha=3.0)
    Cg, Rg, Kg, Eg = track(sim_g, steps, dt)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    it = np.arange(steps)

    # flocking row
    axes[0, 0].plot(it, Cf); axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("connectivity C(t)")
    axes[0, 1].plot(it, Rf); axes[0, 1].set_title("cohesion radius R(t)")
    axes[0, 2].plot(it, Kf); axes[0, 2].set_title("K~(v)")
    axes[0, 3].plot(it, Ef); axes[0, 3].set_title("E~(q)")
    axes[0, 0].set_ylabel(f"(a) Algorithm 2  n={n_flock}")

    # fragmentation row
    axes[1, 0].plot(it, Cg, color="firebrick"); axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("connectivity C(t)")
    axes[1, 1].plot(it, Rg, color="firebrick"); axes[1, 1].set_title("cohesion radius R(t)")
    axes[1, 2].plot(it, Kg, color="firebrick"); axes[1, 2].set_title("K~(v)")
    axes[1, 3].plot(it, Eg, color="firebrick"); axes[1, 3].set_title("E~(q)")
    axes[1, 0].set_ylabel(f"(b) Algorithm 1  n={n_frag}")

    for a in axes.flat:
        a.set_xlabel("iteration")
        a.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=120)
        print(f"saved figure to {save}")
    if live:
        plt.show()

    # summary
    print(f"flocking:  C_final={Cf[-1]:.2f}  R_final={Rf[-1]:.1f}  "
          f"K~_final={Kf[-1]:.4f}  E~_final={Ef[-1]:.4f}")
    print(f"fragment:  C_final={Cg[-1]:.2f}  R_final={Rg[-1]:.1f}  "
          f"K~_final={Kg[-1]:.4f}  E~_final={Eg[-1]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    run(args.steps, args.dt, args.save, not args.headless)

"""Olfati-Saber flocking for multi-agent dynamic systems.

Implements Algorithms 1, 2 and 3 from:
    Olfati-Saber, R. (2006). "Flocking for Multi-Agent Dynamic Systems:
    Algorithms and Theory." IEEE TAC 51(3), 401-420.

Notation follows the paper:
    q_i, p_i in R^m        - position, velocity of alpha-agent i
    q_hat, p_hat           - position, velocity of induced beta-agent
    (q_r, p_r)             - state of gamma-agent (nav target)
    d = desired lattice spacing,  r = interaction range  (kappa = r/d)
    d' = obstacle safety distance,  r' = obstacle interaction range
    sigma-norm, bump rho_h, action phi(.) - see eqs. (8)-(16)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# sigma-norm and smooth primitives  (Sec. II.C, eqs. 8-10, 15)
# ---------------------------------------------------------------------------

def sigma_norm(z, eps=0.1):
    """sigma-norm ||z||_sigma = (1/eps)(sqrt(1 + eps||z||^2) - 1).

    Accepts a scalar (treated as a 1-vector) or an (..., m) array whose
    last axis is reduced. Returns shape z.shape[:-1] for arrays.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        return (np.sqrt(1.0 + eps * float(z) ** 2) - 1.0) / eps
    sq = np.sum(z * z, axis=-1)
    return (np.sqrt(1.0 + eps * sq) - 1.0) / eps


def sigma_norm_scalar(x, eps=0.1):
    """sigma-norm of a scalar distance."""
    return (np.sqrt(1.0 + eps * x * x) - 1.0) / eps


def sigma_eps(z, eps=0.1):
    """Gradient of sigma-norm: sigma_eps(z) = z / sqrt(1 + eps||z||^2).

    z shape (..., m) -> output shape (..., m).
    """
    z = np.asarray(z, dtype=float)
    denom = np.sqrt(1.0 + eps * np.sum(z * z, axis=-1, keepdims=True))
    return z / denom


def bump(z, h):
    """Smooth bump rho_h(z) on [0,1] (eq. 10). Scalar or array input."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    lo = (z >= 0.0) & (z < h)
    mi = (z >= h) & (z <= 1.0)
    out = np.where(lo, 1.0, out)
    out = np.where(mi, 0.5 * (1.0 + np.cos(np.pi * (z - h) / (1.0 - h))), out)
    return out


def _sigma_1(z):
    return z / np.sqrt(1.0 + z * z)


def phi(z, a=5.0, b=5.0):
    """Uneven sigmoid action (eq. 15).  0 < a <= b,  c = |a-b|/sqrt(4ab)."""
    c = abs(a - b) / np.sqrt(4.0 * a * b)
    return 0.5 * ((a + b) * _sigma_1(z + c) + (a - b))


def phi_alpha(z, r_alpha, d_alpha, h=0.2, a=5.0, b=5.0):
    """Attractive/repulsive action between alpha-agents (combine 11,15)."""
    return bump(z / r_alpha, h) * phi(z - d_alpha, a, b)


def phi_beta(z, d_beta, h=0.9):
    """Repulsive action against beta-agents (obstacles), eq. (56)."""
    return bump(z / d_beta, h) * (_sigma_1(z - d_beta) - 1.0)


# ---------------------------------------------------------------------------
# Pairwise force assembly for a configuration q, p
# ---------------------------------------------------------------------------

def alpha_forces(q, p, r_alpha, d_alpha, eps=0.1,
                 h=0.2, a=5.0, b=5.0):
    """Gradient + velocity-consensus terms (eqs. 23 / first two of 59).

    Returns (grad_term, consensus_term, adjacency), each position term has
    shape (n, m), adjacency is (n, n).
    """
    n = q.shape[0]
    # pairwise differences:  diff_qij = q_j - q_i
    diff_q = q[None, :, :] - q[:, None, :]          # (n, n, m)
    diff_p = p[None, :, :] - p[:, None, :]          # (n, n, m)

    # sigma-norm distance and its gradient (n_ij)
    sq_norm = np.sum(diff_q * diff_q, axis=-1)      # (n, n)
    sn = (np.sqrt(1.0 + eps * sq_norm) - 1.0) / eps  # (n, n)
    n_ij = diff_q / np.sqrt(1.0 + eps * sq_norm)[..., None]  # (n, n, m)

    # mask out self pairs
    eye = np.eye(n, dtype=bool)
    sn = np.where(eye, 0.0, sn)

    # adjacency a_ij(q) = rho_h(||q_j - q_i||_sigma / r_alpha); zero on diag
    adj = bump(sn / r_alpha, h)                     # (n, n)
    adj = np.where(eye, 0.0, adj)

    # gradient-based term: sum_j phi_alpha(sn_ij) * n_ij
    phi_vals = phi_alpha(sn, r_alpha, d_alpha, h=h, a=a, b=b)  # (n, n)
    phi_vals = np.where(eye, 0.0, phi_vals)
    grad = np.sum(phi_vals[..., None] * n_ij, axis=1)          # (n, m)

    # velocity consensus: sum_j a_ij (p_j - p_i)
    cons = np.sum(adj[..., None] * diff_p, axis=1)             # (n, m)

    return grad, cons, adj


# ---------------------------------------------------------------------------
# Obstacle -> beta-agent projection  (Lemma 4, Sec. VII.D)
# ---------------------------------------------------------------------------

class GammaAgent:
    """Dynamic gamma-agent (navigation target), eq. (25).

    State (q_r, p_r) evolves as:
        dq_r/dt = p_r
        dp_r/dt = f_r(q_r, p_r)

    `f_r` defaults to zero (q_r moves at constant velocity p_r0). A user
    can pass any callable (q_r, p_r) -> acceleration in R^m.
    """

    __slots__ = ("q", "p", "f_r")

    def __init__(self, q0, p0, f_r=None):
        self.q = np.asarray(q0, dtype=float).copy()
        self.p = np.asarray(p0, dtype=float).copy()
        self.f_r = f_r if f_r is not None else (lambda q, p: np.zeros_like(p))

    def step(self, dt):
        u = self.f_r(self.q, self.p)
        self.p = self.p + dt * u
        self.q = self.q + dt * self.p


class SphereObstacle:
    """Solid sphere of radius R centered at y. Used for beta-agents."""

    __slots__ = ("y", "R")

    def __init__(self, y, R):
        self.y = np.asarray(y, dtype=float)
        self.R = float(R)

    def beta_state(self, q, p):
        """Return (q_hat, p_hat, active) for each alpha-agent.

        q_hat, p_hat have shape (n, m). `active` (n,) is True when the
        alpha-agent sits outside the sphere (otherwise the projection is
        undefined / the agent is penetrating the obstacle).
        """
        diff = q - self.y                                # (n, m)
        dist = np.linalg.norm(diff, axis=-1)             # (n,)
        active = dist > 1e-9
        safe_dist = np.where(active, dist, 1.0)
        mu = self.R / safe_dist                          # (n,)
        a_k = diff / safe_dist[:, None]                  # unit radial (n, m)
        # projection matrix P_i = I - a a^T, applied to p_i
        ap = np.sum(a_k * p, axis=-1, keepdims=True)     # (n, 1)
        Pp = p - a_k * ap                                # (n, m)
        q_hat = mu[:, None] * q + (1.0 - mu)[:, None] * self.y
        p_hat = mu[:, None] * Pp
        return q_hat, p_hat, active


class WallObstacle:
    """Infinite half-space boundary: hyperplane through y with unit normal a."""

    __slots__ = ("y", "a")

    def __init__(self, y, a):
        self.y = np.asarray(y, dtype=float)
        a = np.asarray(a, dtype=float)
        self.a = a / np.linalg.norm(a)

    def beta_state(self, q, p):
        a = self.a
        # P = I - a a^T applied to q_i - y_k, added to y_k
        ap_q = (q - self.y) @ a                          # (n,)
        q_hat = q - ap_q[:, None] * a                    # projection
        ap_p = p @ a
        p_hat = p - ap_p[:, None] * a
        active = np.ones(q.shape[0], dtype=bool)
        return q_hat, p_hat, active


def beta_forces(q, p, obstacles, r_prime, d_beta, eps=0.1, h=0.9):
    """Sum obstacle repulsion + damping across all obstacles.

    Returns grad_beta (n, m) and cons_beta (n, m).
    """
    grad = np.zeros_like(q)
    cons = np.zeros_like(q)
    r_prime_sigma = sigma_norm_scalar(r_prime, eps)
    for obs in obstacles:
        q_hat, p_hat, active = obs.beta_state(q, p)
        diff_q = q_hat - q                                   # (n, m)
        diff_p = p_hat - p                                   # (n, m)
        sq = np.sum(diff_q * diff_q, axis=-1)
        sn = (np.sqrt(1.0 + eps * sq) - 1.0) / eps           # (n,)
        # only interact when q_hat is within r_prime of the agent
        in_range = (sn < r_prime_sigma) & active
        if not np.any(in_range):
            continue
        n_hat = diff_q / np.sqrt(1.0 + eps * sq)[:, None]    # (n, m)
        b_ik = bump(sn / d_beta, h)                          # (n,)
        phi_b = phi_beta(sn, d_beta, h)                      # (n,)
        mask = in_range.astype(float)[:, None]
        grad += (phi_b[:, None] * n_hat) * mask
        cons += (b_ik[:, None] * diff_p) * mask
    return grad, cons


# ---------------------------------------------------------------------------
# High-level simulation  (glues Algorithms 1 / 2 / 3)
# ---------------------------------------------------------------------------

class FlockSim:
    """Second-order particle system with flocking control input.

    dq/dt = p;   dp/dt = u

    Algorithm is chosen via constructor args:
        algorithm = 1  ->  (alpha,alpha) protocol (eq. 23), fragments.
        algorithm = 2  ->  Algorithm 1 + gamma navigational feedback (eq. 24).
        algorithm = 3  ->  Algorithm 2 + beta obstacle avoidance (eq. 59).
    """

    def __init__(self, q0, p0, *, algorithm=2,
                 d=7.0, r=None, eps=0.1,
                 c1_alpha=3.0, c2_alpha=None,
                 c1_beta=1500.0, c2_beta=None,
                 c1_gamma=1.1, c2_gamma=None,
                 h_alpha=0.2, h_beta=0.9, a_phi=5.0, b_phi=5.0,
                 obstacles=None,
                 target_pos=None, target_vel=None):
        self.q = np.asarray(q0, dtype=float).copy()
        self.p = np.asarray(p0, dtype=float).copy()
        self.n, self.m = self.q.shape
        self.algorithm = algorithm

        self.d = float(d)
        self.r = 1.2 * self.d if r is None else float(r)
        self.eps = float(eps)
        self.d_alpha = sigma_norm_scalar(self.d, eps)
        self.r_alpha = sigma_norm_scalar(self.r, eps)

        # beta parameters (Sec. VIII: d' = 0.6 d, r' = 1.2 d')
        self.d_prime = 0.6 * self.d
        self.r_prime = 1.2 * self.d_prime
        self.d_beta = sigma_norm_scalar(self.d_prime, eps)

        self.c1_alpha = c1_alpha
        self.c2_alpha = 2.0 * np.sqrt(c1_alpha) if c2_alpha is None else c2_alpha
        self.c1_beta = c1_beta
        self.c2_beta = 2.0 * np.sqrt(c1_beta) if c2_beta is None else c2_beta
        self.c1_gamma = c1_gamma
        self.c2_gamma = 2.0 * np.sqrt(c1_gamma) if c2_gamma is None else c2_gamma

        self.h_alpha = h_alpha
        self.h_beta = h_beta
        self.a_phi = a_phi
        self.b_phi = b_phi

        self.obstacles = list(obstacles) if obstacles else []

        # gamma-agent (navigational target).  May be None or callable(t) -> (q_r, p_r)
        self.target_pos = target_pos
        self.target_vel = target_vel

    # -----------------------------------------------------------------
    def gamma_state(self, t):
        qr, pr = self.target_pos, self.target_vel
        if isinstance(qr, GammaAgent):
            return qr.q, qr.p
        if callable(qr):
            qr = qr(t)
        if callable(pr):
            pr = pr(t)
        return (np.asarray(qr, dtype=float) if qr is not None else None,
                np.asarray(pr, dtype=float) if pr is not None else None)

    def control(self, t):
        """Assemble u_i for every agent (returns shape (n, m))."""
        grad_a, cons_a, adj = alpha_forces(
            self.q, self.p, self.r_alpha, self.d_alpha,
            eps=self.eps, h=self.h_alpha, a=self.a_phi, b=self.b_phi)
        u = self.c1_alpha * grad_a + self.c2_alpha * cons_a

        if self.algorithm >= 3 and self.obstacles:
            grad_b, cons_b = beta_forces(
                self.q, self.p, self.obstacles,
                r_prime=self.r_prime, d_beta=self.d_beta,
                eps=self.eps, h=self.h_beta)
            u = u + self.c1_beta * grad_b + self.c2_beta * cons_b

        if self.algorithm >= 2:
            qr, pr = self.gamma_state(t)
            if qr is not None:
                u = u - self.c1_gamma * _navfb(self.q, qr)
            if pr is not None:
                u = u - self.c2_gamma * (self.p - pr)
        return u, adj

    def step(self, dt, t=0.0):
        """One explicit Euler / symplectic step. Returns the used adjacency."""
        u, adj = self.control(t)
        self.p = self.p + dt * u
        self.q = self.q + dt * self.p
        # advance dynamic gamma-agent, if any
        if isinstance(self.target_pos, GammaAgent):
            self.target_pos.step(dt)
        return adj

    # -----------------------------------------------------------------
    # Diagnostics (Sec. IX.A)
    # -----------------------------------------------------------------
    def cohesion_radius(self):
        qc = self.q.mean(axis=0)
        return np.max(np.linalg.norm(self.q - qc, axis=-1))

    def velocity_mismatch(self):
        pc = self.p.mean(axis=0)
        return float(np.sum((self.p - pc) ** 2))

    def proximity_adjacency(self):
        """0/1 adjacency matrix of the proximity net G(q) (||q_i-q_j|| < r)."""
        d = np.linalg.norm(self.q[None, :, :] - self.q[:, None, :], axis=-1)
        adj = (d < self.r).astype(float)
        np.fill_diagonal(adj, 0.0)
        return adj

    def relative_connectivity(self):
        """C(t) = rank(L(q)) / (n-1) in [0, 1]  (Sec. IX.A, i)."""
        A = self.proximity_adjacency()
        D = np.diag(A.sum(axis=1))
        L = D - A
        r = np.linalg.matrix_rank(L, tol=1e-6)
        return float(r) / max(self.n - 1, 1)

    def normalized_velocity_mismatch(self):
        """K_tilde = K(v) / n  (Sec. IX.A, iv)."""
        return self.velocity_mismatch() / self.n

    def normalized_deviation_energy(self):
        """E_tilde = E(q) / d^2  (Sec. IX.A, iii)."""
        adj = self.proximity_adjacency()
        return self.deviation_energy(adj) / (self.d ** 2)

    def deviation_energy(self, adj):
        """E(q) from eq. (7), restricted to edges of the proximity net."""
        # use the 0/1 version of adjacency (neighbors only)
        neighbors = (np.linalg.norm(
            self.q[None, :, :] - self.q[:, None, :], axis=-1) < self.r)
        np.fill_diagonal(neighbors, False)
        edges = np.count_nonzero(neighbors) / 2
        if edges == 0:
            return 0.0
        dists = np.linalg.norm(
            self.q[None, :, :] - self.q[:, None, :], axis=-1)
        sn = (np.sqrt(1.0 + self.eps * dists ** 2) - 1.0) / self.eps
        dev = (sn - self.d_alpha) ** 2 * neighbors
        return float(dev.sum() / (2 * (edges + 1)))


def _navfb(q, qr):
    """Navigation pos term: (q_i - q_r) with q_r possibly scalar-per-dim."""
    qr = np.asarray(qr, dtype=float)
    if qr.ndim == 1:
        return q - qr[None, :]
    return q - qr

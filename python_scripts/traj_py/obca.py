from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass
class ObcaInput:
    ref: np.ndarray
    x_initial: np.ndarray
    u_initial: np.ndarray
    lambda_initial: tuple
    mu_initial: tuple
    Aob: tuple
    bob: tuple
    G: np.ndarray
    g: np.ndarray
    L: float
    start: np.ndarray
    goal: np.ndarray
    Ts: float
    v_max: float
    a_max: float
    delta_max: float
    ddelta_max: float
    dmin: float
    solver_options: dict
    parking: bool = False
    collision_constraints: bool = True
    bounds: np.ndarray | None = None
    parking_objective: str = "repeated"
    collision_midpoints: int = 0
    # 'euler' keeps the historical short Euler step from node k.
    # 'linear' interpolates (x,y,theta) like eval densify_poses.
    collision_midpoint_kind: str = "euler"
    # Added to dmin in *constraints only*. Evaluation stays at dmin.
    dmin_buffer: float = 0.0


@dataclass
class ObcaResult:
    X: np.ndarray
    U: np.ndarray
    objective: float
    iterations: int
    status: str


def solve_obca(p: ObcaInput) -> ObcaResult:
    n = p.ref.shape[0] - 1
    opti = ca.Opti()
    X = opti.variable(4, n + 1)
    U = opti.variable(2, n)
    lam, mu = [], []
    for A in p.Aob if p.collision_constraints else ():
        lam.append(opti.variable(A.shape[0], n + 1))
        mu.append(opti.variable(p.G.shape[0], n + 1))
    J = 0
    for k in range(n):
        if p.parking:
            J = J + 0.5*U[0, k]**2 + 0.1*U[1, k]**2
            if k < n - 1:
                J = J + 0.2*(U[0, k+1]-U[0, k])**2 + 2.0*(U[1, k+1]-U[1, k])**2
            # Preserve MATLAB run_obca_parking.m:62-67: repeated N times.
            if p.parking_objective != "once":
                for q in range(n + 1):
                    J = J + 0.3*((X[0, q]-p.x_initial[0, q])**2 + (X[1, q]-p.x_initial[1, q])**2)
                for q in range(n - 1):
                    J = J + (X[3, q+1]-X[3, q])**2
        else:
            J = J + 0.1 * U[0, k] ** 2 + 0.05 * U[1, k] ** 2
            if k < n - 1:
                J = J + (U[0, k + 1] - U[0, k]) ** 2 + 0.5 * (U[1, k + 1] - U[1, k]) ** 2
    if p.parking and p.parking_objective == "once":
        # Same terms as the source, accumulated once instead of N times.
        for q in range(n + 1):
            J = J + 0.3*((X[0, q]-p.x_initial[0, q])**2 + (X[1, q]-p.x_initial[1, q])**2)
        for q in range(n - 1):
            J = J + (X[3, q+1]-X[3, q])**2
    opti.minimize(J)
    for k in range(n):
        opti.subject_to(X[0, k + 1] == X[0, k] + X[3, k] * ca.cos(X[2, k]) * p.Ts)
        opti.subject_to(X[1, k + 1] == X[1, k] + X[3, k] * ca.sin(X[2, k]) * p.Ts)
        opti.subject_to(X[2, k + 1] == X[2, k] + X[3, k] / p.L * ca.tan(U[0, k]) * p.Ts)
        opti.subject_to(X[3, k + 1] == X[3, k] + U[1, k] * p.Ts)
    opti.subject_to(X[:3, 0] == p.start)
    opti.subject_to(X[3, 0] == 0)
    if p.parking:
        for q in range(3):
            opti.subject_to(opti.bounded(p.goal[q]-.05, X[q, n], p.goal[q]+.05))
        opti.subject_to(opti.bounded(-.1, X[3, n], .1))
    else:
        opti.subject_to(X[:3, n] == p.goal)
        opti.subject_to(X[3, n] == 0)
    opti.subject_to(opti.bounded(-p.v_max, X[3, :], p.v_max))
    opti.subject_to(opti.bounded(-p.delta_max, U[0, :], p.delta_max))
    opti.subject_to(opti.bounded(-p.a_max, U[1, :], p.a_max))
    for k in range(n - 1):
        opti.subject_to(opti.bounded(-p.ddelta_max * p.Ts, U[0, k + 1] - U[0, k], p.ddelta_max * p.Ts))
    if p.parking:
        opti.subject_to(opti.bounded(p.bounds[0]+1, X[0, :], p.bounds[1]-1))
        opti.subject_to(opti.bounded(p.bounds[2], X[1, :], p.bounds[3]-1))
    for k in range(n + 1) if p.collision_constraints else ():
        th = X[2, k]
        R = ca.vertcat(ca.horzcat(ca.cos(th), -ca.sin(th)), ca.horzcat(ca.sin(th), ca.cos(th)))
        t = X[:2, k]
        for m, (A, b) in enumerate(zip(p.Aob, p.bob)):
            l, u = lam[m][:, k], mu[m][:, k]
            opti.subject_to(l >= 0)
            opti.subject_to(u >= 0)
            opti.subject_to(-(ca.DM(p.g).T @ u) + (A @ t - b).T @ l >= (p.dmin + float(getattr(p, 'dmin_buffer', 0.0) or 0.0)))
            opti.subject_to(p.G.T @ u + R.T @ (A.T @ l) == 0)
            opti.subject_to((A.T @ l).T @ (A.T @ l) <= 1)
    # Optional interior samples between collocated nodes. These close a sampled
    # gap in the exact-distance predicate; they are not a continuous swept-volume
    # certificate. Off by default so solve_obca keeps its original formulation.
    n_mid = int(getattr(p, "collision_midpoints", 0) or 0)
    if p.collision_constraints and n_mid > 0:
        mid_lams, mid_mus = [], []
        for s in range(n_mid):
            mid_lams.append([opti.variable(A.shape[0], n) for A in p.Aob])
            mid_mus.append([opti.variable(p.G.shape[0], n) for _ in p.Aob])
        kind = str(getattr(p, "collision_midpoint_kind", "euler") or "euler")
        for s in range(n_mid):
            frac = (s + 1) / (n_mid + 1)
            Ts_m = p.Ts * frac
            for k in range(n):
                if kind == "linear":
                    # Same interpolation family as geometry.densify_poses.
                    th0 = X[2, k]
                    th1 = X[2, k + 1]
                    dth = ca.atan2(ca.sin(th1 - th0), ca.cos(th1 - th0))
                    t = ca.vertcat(
                        (1 - frac) * X[0, k] + frac * X[0, k + 1],
                        (1 - frac) * X[1, k] + frac * X[1, k + 1],
                    )
                    thm = th0 + frac * dth
                else:
                    th = X[2, k]
                    t = ca.vertcat(
                        X[0, k] + X[3, k] * ca.cos(th) * Ts_m,
                        X[1, k] + X[3, k] * ca.sin(th) * Ts_m,
                    )
                    thm = th + X[3, k] / p.L * ca.tan(U[0, k]) * Ts_m
                R = ca.vertcat(ca.horzcat(ca.cos(thm), -ca.sin(thm)), ca.horzcat(ca.sin(thm), ca.cos(thm)))
                for m, (A, b) in enumerate(zip(p.Aob, p.bob)):
                    l, u = mid_lams[s][m][:, k], mid_mus[s][m][:, k]
                    opti.subject_to(l >= 0)
                    opti.subject_to(u >= 0)
                    opti.subject_to(-(ca.DM(p.g).T @ u) + (A @ t - b).T @ l >= (p.dmin + float(getattr(p, 'dmin_buffer', 0.0) or 0.0)))
                    opti.subject_to(p.G.T @ u + R.T @ (A.T @ l) == 0)
                    opti.subject_to((A.T @ l).T @ (A.T @ l) <= 1)
            for m in range(len(p.Aob)):
                opti.set_initial(mid_lams[s][m], 0.1)
                opti.set_initial(mid_mus[s][m], 0.1)
    opti.set_initial(X, p.x_initial)
    opti.set_initial(U, p.u_initial)
    for m in range(len(lam)):
        opti.set_initial(lam[m], p.lambda_initial[m])
        opti.set_initial(mu[m], p.mu_initial[m])
    options = dict(p.solver_options)
    options['print_time'] = bool(options['print_time'])
    opti.solver('ipopt', options)
    sol = opti.solve()
    stats = opti.stats()
    return ObcaResult(sol.value(X), sol.value(U), float(sol.value(J)), int(stats['iter_count']), stats['return_status'])

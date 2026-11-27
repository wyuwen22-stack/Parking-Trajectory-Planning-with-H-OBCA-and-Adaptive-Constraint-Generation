from dataclasses import dataclass

import casadi as ca
import numpy as np


@dataclass
class MpcInput:
    Xref: np.ndarray
    Uref: np.ndarray
    initial_state: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Ts: float
    Np: int
    L: float
    delta_max: float
    a_max: float
    solver_options: dict


@dataclass
class MpcResult:
    X_actual: np.ndarray
    U_applied: np.ndarray
    statuses: list[str]


def track_mpc(p: MpcInput) -> MpcResult:
    xs, us = ca.MX.sym('x', 4), ca.MX.sym('u', 2)
    xnext = ca.vertcat(
        xs[0] + xs[3] * ca.cos(xs[2]) * p.Ts,
        xs[1] + xs[3] * ca.sin(xs[2]) * p.Ts,
        xs[2] + xs[3] / p.L * ca.tan(us[0]) * p.Ts,
        xs[3] + us[1] * p.Ts,
    )
    f_dyn = ca.Function('f', [xs, us], [xnext])
    A_fun = ca.Function('A', [xs, us], [ca.jacobian(xnext, xs)])
    B_fun = ca.Function('B', [xs, us], [ca.jacobian(xnext, us)])
    n = p.Uref.shape[1]
    x_now = p.initial_state.copy()
    X_actual = np.zeros((4, n + 1))
    X_actual[:, 0] = x_now
    U_applied = np.zeros((2, n))
    statuses = []
    for k in range(n):
        horizon = min(int(p.Np), n - k)
        # Preserve MATLAB's per-step rebuild and default initial guess.
        opti = ca.Opti()
        dx = opti.variable(4, horizon + 1)
        du = opti.variable(2, horizon)
        J = 0
        opti.subject_to(dx[:, 0] == x_now - p.Xref[:, k])
        for j in range(horizon):
            xr, ur = p.Xref[:, k + j], p.Uref[:, k + j]
            Ad, Bd = A_fun(xr, ur).full(), B_fun(xr, ur).full()
            opti.subject_to(dx[:, j + 1] == Ad @ dx[:, j] + Bd @ du[:, j])
            J = J + dx[:, j + 1].T @ p.Q @ dx[:, j + 1] + du[:, j].T @ p.R @ du[:, j]
            opti.subject_to(opti.bounded(-p.delta_max, ur[0] + du[0, j], p.delta_max))
            opti.subject_to(opti.bounded(-p.a_max, ur[1] + du[1, j], p.a_max))
        opti.minimize(J)
        options = dict(p.solver_options)
        options['print_time'] = bool(options['print_time'])
        opti.solver('ipopt', options)
        try:
            sol = opti.solve()
            du0 = np.asarray(sol.value(du[:, 0])).reshape(2)
            statuses.append(opti.stats()['return_status'])
        except RuntimeError:
            du0 = np.zeros(2)
            statuses.append('solve_failed')
        u_apply = p.Uref[:, k] + du0
        u_apply[0] = np.clip(u_apply[0], -p.delta_max, p.delta_max)
        u_apply[1] = np.clip(u_apply[1], -p.a_max, p.a_max)
        U_applied[:, k] = u_apply
        x_now = f_dyn(x_now, u_apply).full().reshape(4)
        X_actual[:, k + 1] = x_now
    return MpcResult(X_actual, U_applied, statuses)

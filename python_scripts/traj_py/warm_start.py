from dataclasses import dataclass
import numpy as np

from .obca import ObcaInput
from .scenario import Scenario


@dataclass
class WarmStart:
    indices: np.ndarray
    problem: ObcaInput


def choose_parking_horizon(path, Ts=0.4, v_nom=1.0, n_min=40, n_max=160) -> int:
    """Allocate collocated steps from path length, not a fixed 20 s horizon.

    n ≈ path_length / (v_nom * Ts), clipped. This is a discretization choice,
    not a standardized maneuver duration.
    """
    import math
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return n_min
    length = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
    n = int(math.ceil(length / max(float(v_nom) * float(Ts), 1e-3)))
    return int(min(n_max, max(n_min, n)))


def make_warm_start(path: np.ndarray, scene: Scenario, parking=False, collision=True,
                    horizon=None, collision_midpoints=0) -> WarmStart:
    n = 50 if parking else 40
    if horizon is not None:
        n = int(horizon)
    Ts = .4
    # MATLAB round uses half-away-from-zero; keep exported indices 1-based.
    indices = np.floor(np.linspace(1, len(path), n+1) + .5).astype(int)
    ref = path[indices-1]
    X = np.zeros((4, n+1))
    X[:3] = ref.T
    for k in range(n):
        delta = ref[k+1, :2] - ref[k, :2]
        sign = 1.
        if parking:
            sign = np.sign(delta @ np.array([np.cos(ref[k, 2]), np.sin(ref[k, 2])]))
            if sign == 0:
                sign = 1.
        X[3, k] = sign*np.linalg.norm(delta)/Ts
    U = np.zeros((2, n))
    for k in range(n):
        dth = ref[k+1, 2]-ref[k, 2]
        if parking:
            dth = np.arctan2(np.sin(dth), np.cos(dth))
            U[0, k] = np.arctan(dth*scene.veh.L/max(abs(X[3, k])*Ts, .1))*np.sign(X[3, k]+np.finfo(float).eps)
        else:
            U[0, k] = np.arctan(dth*scene.veh.L/max(X[3, k]*Ts, .1))
        U[1, k] = (X[3, min(k+1, n-1)]-X[3, k])/Ts
    xu, yu, xl, yl = scene.veh.ego
    options = {'ipopt': {'print_level': 5, 'max_iter': 8000 if parking else (5000 if collision else 3000)}, 'print_time': True}
    if collision:
        options['ipopt']['tol'] = 1e-4
    p = ObcaInput(ref, X, U,
        tuple(np.full((len(A), n+1), .1) for A in scene.Aineq),
        tuple(np.full((4, n+1), .1) for _ in scene.obs),
        scene.Aineq, scene.bineq, np.array([[1., 0], [-1, 0], [0, 1], [0, -1]]),
        np.array([xu, xl, yu, yl]), scene.veh.L, scene.start, scene.goal, Ts,
        3. if parking else 6., 1. if parking else 1.5, np.arctan(scene.veh.L/scene.veh.Rmin), .6, .05,
        options, parking, collision, scene.XYbounds)
    p.collision_midpoints = int(collision_midpoints)
    return WarmStart(indices, p)

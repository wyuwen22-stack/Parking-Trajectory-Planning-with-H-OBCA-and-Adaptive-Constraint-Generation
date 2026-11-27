"""Success predicate and trajectory metrics for the parking suite."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from traj_py.analysis import path_curvature, vehicle_vertices
from traj_py.geometry import (
    densify_poses,
    polygon_distance,
    polygons_overlap,
    pose_clearance as geom_pose_clearance,
)

DMIN = 0.05
CLEARANCE_TOL = 1e-12
CONS_TOL = 1e-3
TERM_POS_TOL = 0.05
TERM_HEADING_TOL = 0.05
TERM_V_TOL = 0.1
SWEEP_STEP = 0.05
CUSP_V_THRESH = 1e-3


def wrap_angle(delta) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    return np.arctan2(np.sin(delta), np.cos(delta))



def pose_clearance(pose, scene) -> tuple[float, bool]:
    return geom_pose_clearance(pose, scene.obs, scene.veh.ego, bounds=None)


def node_and_swept_clearance(poses: np.ndarray, scene) -> tuple[float, float, bool]:
    poses = np.asarray(poses, dtype=float)
    node_clearance = np.inf
    node_overlap = False
    for pose in poses:
        clearance, overlapped = pose_clearance(pose, scene)
        node_clearance = min(node_clearance, clearance)
        node_overlap = node_overlap or overlapped
    swept = densify_poses(poses, scene.veh.ego)
    swept_clearance = np.inf
    swept_overlap = False
    for pose in swept:
        clearance, overlapped = pose_clearance(pose, scene)
        swept_clearance = min(swept_clearance, clearance)
        swept_overlap = swept_overlap or overlapped
    return float(node_clearance), float(swept_clearance), bool(node_overlap or swept_overlap)


def cusp_count_from_velocity(velocity: np.ndarray, thresh: float = CUSP_V_THRESH) -> int:
    signs = np.sign(np.asarray(velocity, dtype=float))
    signs[np.abs(velocity) <= thresh] = 0
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def path_direction_signs(poses: np.ndarray, thresh: float = 1e-9) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    signs = []
    for i in range(len(poses) - 1):
        delta = poses[i + 1, :2] - poses[i, :2]
        heading = np.array([np.cos(poses[i, 2]), np.sin(poses[i, 2])])
        projection = float(delta @ heading)
        if abs(projection) <= thresh and np.linalg.norm(delta) <= thresh:
            signs.append(0.0)
        else:
            signs.append(1.0 if projection >= 0 else -1.0)
    return np.asarray(signs)


def path_length_xy(poses: np.ndarray) -> float:
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)))


def parking_cost(X: np.ndarray, U: np.ndarray, X_ws: np.ndarray, mode: str = "once") -> float:
    """Closed-form parking objective. Δv uses k = 0..n-2, matching the source loops."""
    n = U.shape[1]
    control = (
        0.5 * np.sum(U[0] ** 2)
        + 0.1 * np.sum(U[1] ** 2)
        + 0.2 * np.sum(np.diff(U[0]) ** 2)
        + 2.0 * np.sum(np.diff(U[1]) ** 2)
    )
    position = 0.3 * np.sum((X[:2] - X_ws[:2]) ** 2)
    # Source loop is `for q in range(n - 1)`, i.e. exclude the last velocity step.
    delta_v = np.sum(np.diff(X[3, :-1]) ** 2)
    if mode == "once":
        return float(control + position + delta_v)
    return float(control + n * (position + delta_v))


def constraint_residuals(X: np.ndarray, U: np.ndarray, limits) -> dict:
    v_max, a_max, delta_max, ddelta_max, Ts = limits
    return {
        "v": float(max(0.0, np.max(np.abs(X[3])) - v_max)),
        "a": float(max(0.0, np.max(np.abs(U[1])) - a_max)),
        "delta": float(max(0.0, np.max(np.abs(U[0])) - delta_max)),
        "ddelta": float(max(0.0, np.max(np.abs(np.diff(U[0]))) - ddelta_max * Ts)),
    }


def terminal_errors(pose, velocity, goal) -> dict:
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return {
        "dx": float(pose[0] - goal[0]),
        "dy": float(pose[1] - goal[1]),
        "dpsi": float(wrap_angle(pose[2] - goal[2])),
        "v": float(velocity) if velocity is not None else float("nan"),
    }


def first_fail_code(flags: dict) -> str:
    """Return the first failing code in WS, OPT, CONS, COLL, TERM order."""
    for code in ("WS", "OPT", "CONS", "COLL", "TERM"):
        if flags.get(code):
            return code
    return ""


def extra_violations(flags: dict, primary: str) -> str:
    extras = [code for code in ("WS", "OPT", "CONS", "COLL", "TERM") if flags.get(code) and code != primary]
    return ",".join(extras)


def collision_failed(node_clearance: float, swept_clearance: float, overlapped: bool) -> bool:
    limit = DMIN - CLEARANCE_TOL
    return bool(overlapped or node_clearance < limit or swept_clearance < limit)


def terminal_failed(errors: dict, has_velocity: bool) -> bool:
    if abs(errors["dx"]) > TERM_POS_TOL or abs(errors["dy"]) > TERM_POS_TOL:
        return True
    if abs(errors["dpsi"]) > TERM_HEADING_TOL:
        return True
    if has_velocity and abs(errors["v"]) > TERM_V_TOL:
        return True
    return False


@dataclass
class TrajectoryMetrics:
    min_node_clearance: float = float("nan")
    min_swept_clearance: float = float("nan")
    overlapped: bool = False
    term_dx: float = float("nan")
    term_dy: float = float("nan")
    term_dpsi: float = float("nan")
    term_v: float = float("nan")
    cusp_count: int = -1
    max_kappa: float = float("nan")
    max_delta: float = float("nan")
    max_a: float = float("nan")
    max_ddelta: float = float("nan")
    path_length: float = float("nan")
    parking_time: float = float("nan")
    pos_rms: float = float("nan")
    heading_rms: float = float("nan")
    cons_residual: float = float("nan")
    flags: dict = field(default_factory=dict)

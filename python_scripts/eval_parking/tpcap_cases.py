"""Load the official TPCAP preliminary 20 cases.

Official preliminary case records are stored in inputs/tpcap_preliminary_20.official.json.
This module does not invent official v_max, steer limits, or R_min.
Cases 13-15 are rigidly translated to a local frame for numerical
conditioning; the inverse is stored and tested. That is a numerical
pre-process, not a change of scene geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from traj_py.geometry import close_vertices, convex_decompose, is_convex_polygon, open_vertices, pose_clearance
from traj_py.scenario import Scenario, Vehicle, _poly2hrep

OFFICIAL_ROOT = Path(__file__).resolve().parents[2] / "inputs"
OFFICIAL_JSON = OFFICIAL_ROOT / "tpcap_preliminary_20.official.json"
OFFICIAL_CSV_DIR = OFFICIAL_ROOT / "cases_original"

# Official demonstration vehicle (TPCAP_demo_Python RunMe.py). R_min is not official.
TPCAP_WHEELBASE = 2.8
TPCAP_FRONT = 0.96
TPCAP_REAR = 0.929
TPCAP_WIDTH = 1.942
TPCAP_LENGTH = TPCAP_FRONT + TPCAP_WHEELBASE + TPCAP_REAR  # 4.689
TPCAP_EGO = np.array([
    TPCAP_WHEELBASE + TPCAP_FRONT,  # xu = 3.76
    TPCAP_WIDTH / 2.0,              # yu = 0.971
    TPCAP_REAR,                     # xl = 0.929
    TPCAP_WIDTH / 2.0,              # yl = 0.971
], dtype=np.float64)

# Project vehicle (unchanged). Used for the adapted TPCAP evaluation.
from parking_config import VEHICLE as VEHICLE_CONFIG

PROJECT_VEHICLE = Vehicle(VEHICLE_CONFIG['L'], np.array(VEHICLE_CONFIG['ego'], dtype=np.float64), VEHICLE_CONFIG['Rmin'])
# Official body on the project planner kinematics. Rmin is the project setting, not an official limit.
OFFICIAL_BODY_PROJECT_KINEMATICS = Vehicle(TPCAP_WHEELBASE, TPCAP_EGO.copy(), VEHICLE_CONFIG['Rmin'])

_PAD = 8.0  # planner workspace padding; not an official boundary


@dataclass(frozen=True)
class TpcapCase:
    case_id: int
    name: str
    sha256: str
    official_start: np.ndarray
    official_goal: np.ndarray
    official_obstacles: tuple[np.ndarray, ...]
    origin_shift: np.ndarray  # official = local + shift; headings unchanged
    start: np.ndarray
    goal: np.ndarray
    obstacles: tuple[np.ndarray, ...]
    convex_pieces: tuple[np.ndarray, ...]
    bounds: np.ndarray
    requires_double_precision: bool
    intent: str
    vehicle: Vehicle
    vehicle_name: str
    min_clearance: float
    start_goal_status: str


def _as_closed(verts) -> np.ndarray:
    return close_vertices(np.asarray(verts, dtype=np.float64))


def _origin_shift(start, goal, obstacles) -> np.ndarray:
    """Rigid translation origin: official start rear-axle (x, y). Headings unchanged."""
    return np.array([float(start[0]), float(start[1])], dtype=np.float64)


def _translate(points: np.ndarray, shift: np.ndarray) -> np.ndarray:
    out = np.array(points, dtype=np.float64, copy=True)
    out[..., :2] = out[..., :2] - shift
    return out


def _workspace_bounds(start, goal, obstacles, pad: float = _PAD) -> np.ndarray:
    xs = [float(start[0]), float(goal[0])]
    ys = [float(start[1]), float(goal[1])]
    for obs in obstacles:
        pts = open_vertices(obs)
        xs.extend(pts[:, 0].tolist())
        ys.extend(pts[:, 1].tolist())
    return np.array([min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad], dtype=np.float64)


def _fill_occupancy(bounds, obstacles, resolution: float = 5.0) -> np.ndarray:
    rows = max(1, int(round((bounds[3] - bounds[2]) * resolution)))
    cols = max(1, int(round((bounds[1] - bounds[0]) * resolution)))
    occupancy = np.zeros((rows, cols), dtype=bool)
    # Cell-center test against convex pieces (already decomposed).
    xs = bounds[0] + (np.arange(cols) + 0.5) / resolution
    ys = bounds[2] + (np.arange(rows) + 0.5) / resolution
    # occupancy row 0 is the top (larger y), matching scenario.py
    for obs in obstacles:
        pts = open_vertices(obs)
        if len(pts) < 3:
            continue
        # Bounding-box pre-cut, then even-odd PIP on cell centers.
        xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
        ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
        c0 = max(0, int(np.floor((xmin - bounds[0]) * resolution)))
        c1 = min(cols, int(np.ceil((xmax - bounds[0]) * resolution)) + 1)
        r0 = max(0, int(np.floor((ymin - bounds[2]) * resolution)))
        r1 = min(rows, int(np.ceil((ymax - bounds[2]) * resolution)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
        inside = _points_in_convex(gx.reshape(-1), gy.reshape(-1), pts).reshape(gy.shape)
        # convert y-up r0..r1 to occupancy rows (top = large y)
        for li, y_index in enumerate(range(r0, r1)):
            occ_row = rows - 1 - y_index
            occupancy[occ_row, c0:c1] |= inside[li]
    return occupancy


def _points_in_convex(x, y, pts) -> np.ndarray:
    # All edge half-planes, orientation from signed area of pts.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(pts)
    area = 0.5 * float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    sign = 1.0 if area >= 0 else -1.0
    inside = np.ones(x.shape, dtype=bool)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        inside &= (sign * cross) >= -1e-12
    return inside


def assemble_general(veh: Vehicle, bounds, start, goal, convex_obstacles, resolution: float = 5.0) -> Scenario:
    obstacles = tuple(close_vertices(np.asarray(o, dtype=np.float64)) for o in convex_obstacles)
    halfspaces = tuple(_poly2hrep(vertices) for vertices in obstacles)
    occupancy = _fill_occupancy(bounds, obstacles, resolution=resolution)
    return Scenario(
        veh=veh,
        XYbounds=np.asarray(bounds, dtype=np.float64),
        start=np.asarray(start, dtype=np.float64),
        goal=np.asarray(goal, dtype=np.float64),
        obs=obstacles,
        Aineq=tuple(A for A, _ in halfspaces),
        bineq=tuple(b for _, b in halfspaces),
        occupancy=occupancy,
        resolution=resolution,
    )


def load_official_payload(path: Path | None = None) -> dict:
    path = path or OFFICIAL_JSON
    with path.open() as handle:
        return json.load(handle)


def invert_translation(local_xy: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """official_xy = local_xy + shift (headings unchanged)."""
    out = np.array(local_xy, dtype=np.float64, copy=True)
    out[..., :2] = out[..., :2] + shift
    return out


def _start_goal_status(start, goal, pieces, veh, min_clearance: float) -> str:
    flags = []
    for name, pose in (("start", start), ("goal", goal)):
        clearance, overlapped = pose_clearance(pose, pieces, veh.ego, bounds=None)
        if overlapped:
            flags.append(f"{name}_overlap")
        elif clearance < min_clearance:
            flags.append(f"{name}_clearance_{clearance:.4f}")
    return ",".join(flags) if flags else "ok"


def parse_official_cases(
    vehicle: Vehicle,
    vehicle_name: str,
    min_clearance: float,
    json_path: Path | None = None,
) -> list[TpcapCase]:
    payload = load_official_payload(json_path)
    cases = []
    for raw in payload["cases"]:
        start_o = np.array([
            raw["start_pose"]["x_m"], raw["start_pose"]["y_m"], raw["start_pose"]["theta_rad"]
        ], dtype=np.float64)
        goal_o = np.array([
            raw["goal_pose"]["x_m"], raw["goal_pose"]["y_m"], raw["goal_pose"]["theta_rad"]
        ], dtype=np.float64)
        obs_o = tuple(_as_closed(item["vertices_xy_m"]) for item in raw["obstacles"])
        shift = np.zeros(2, dtype=np.float64)
        if raw.get("requires_double_precision") or float(raw.get("coord_abs_max_m", 0.0)) > 1e6:
            shift = _origin_shift(start_o, goal_o, obs_o)
        start = _translate(start_o, shift)
        goal = _translate(goal_o, shift)
        obstacles = tuple(_translate(obs, shift) for obs in obs_o)
        pieces = []
        for obs in obstacles:
            pieces.extend(convex_decompose(obs))
        pieces = tuple(pieces)
        bounds = _workspace_bounds(start, goal, obstacles)
        status = _start_goal_status(start, goal, pieces, vehicle, min_clearance)
        cases.append(TpcapCase(
            case_id=int(raw["id"]),
            name=str(raw["name"]),
            sha256=str(raw["sha256"]),
            official_start=start_o,
            official_goal=goal_o,
            official_obstacles=obs_o,
            origin_shift=shift,
            start=start,
            goal=goal,
            obstacles=obstacles,
            convex_pieces=pieces,
            bounds=bounds,
            requires_double_precision=bool(raw.get("requires_double_precision") or np.any(shift != 0)),
            intent=str((raw.get("official_case_intent") or {}).get("intent", "")),
            vehicle=vehicle,
            vehicle_name=vehicle_name,
            min_clearance=float(min_clearance),
            start_goal_status=status,
        ))
    return cases


def scene_from_tpcap(case: TpcapCase) -> Scenario:
    return assemble_general(case.vehicle, case.bounds, case.start, case.goal, case.convex_pieces)


def translation_roundtrip_ok(case: TpcapCase, atol: float = 1e-9) -> bool:
    if np.allclose(case.origin_shift, 0.0):
        return np.allclose(case.start, case.official_start) and np.allclose(case.goal, case.official_goal)
    rec_s = invert_translation(case.start, case.origin_shift)
    rec_g = invert_translation(case.goal, case.origin_shift)
    if not (np.allclose(rec_s, case.official_start, atol=atol) and np.allclose(rec_g, case.official_goal, atol=atol)):
        return False
    if not np.allclose(case.start[2], case.official_start[2]) or not np.allclose(case.goal[2], case.official_goal[2]):
        return False
    for loc, off in zip(case.obstacles, case.official_obstacles):
        if not np.allclose(invert_translation(open_vertices(loc), case.origin_shift), open_vertices(off), atol=atol):
            return False
    return True

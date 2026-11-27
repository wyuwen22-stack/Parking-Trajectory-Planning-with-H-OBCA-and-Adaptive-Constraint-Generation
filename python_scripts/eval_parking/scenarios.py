"""Parametric perpendicular and parallel parking scenes for the success-rate suite.

The dissertation helper `traj_py.scenario.parking_scenario()` is left untouched.
These scenes keep the same vehicle, bounds, aisle height and wall as that
tight perpendicular case, and only change the slot opening that defines
tightness. They are reference geometries, not a pixel-accurate replica of
CDC 2018.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from traj_py.hybrid_astar import PlannerSettings
from traj_py.scenario import Vehicle, _assemble

from parking_config import VEHICLE as VEHICLE_CONFIG, PLANNER

VEHICLE = Vehicle(VEHICLE_CONFIG['L'], np.array(VEHICLE_CONFIG['ego']), VEHICLE_CONFIG['Rmin'])
BOUNDS = np.array([-15.0, 15.0, 0.0, 12.0])
DMIN = 0.05
OBSTACLE_X0 = -12.0
OBSTACLE_X1 = 12.0
WALL_Y0 = 11.0
WALL_Y1 = 12.0
AISLE_Y0 = 5.0
AISLE_Y1 = 11.0

PERP_DEPTH = 5.0
PERP_WIDTHS = {"tight": 2.60, "medium": 2.80, "wide": 3.20}
PERP_GOAL_HEADING = np.pi / 2

# Parallel bay sits under the same aisle: curb [0, 2.5], bay [2.5, 5.0], aisle [5, 11].
PARA_DEPTH = 2.50
PARA_BAY_Y0 = 2.50
PARA_BAY_Y1 = 5.00
PARA_LENGTHS = {"tight": 6.00, "medium": 6.50, "wide": 7.00}
PARA_GOAL_HEADING = 0.0

# Rear-axle frame is asymmetric (front 3.7 m, rear 1.0 m). Body-centering offset:
# goal_x = slot_center_x + (xl - xu) / 2.
XU, YU, XL, YL = VEHICLE.ego
BODY_LENGTH = XU + XL
BODY_WIDTH = YU + YL
PARA_GOAL_X = 0.0 + (XL - XU) / 2.0  # -1.35
PARA_GOAL_Y = 0.5 * (PARA_BAY_Y0 + PARA_BAY_Y1)  # 3.75

START_X = (-8.0, -4.0, 0.0, 4.0, 8.0)
START_Y = (7.5, 8.5)
START_PSI = 0.0

DISSERTATION_START = np.array([-9.0, 7.0, np.pi])
DISSERTATION_GOAL = np.array([0.0, 1.3, np.pi / 2])

TIGHTNESS_LEVELS = ("tight", "medium", "wide")
MANEUVERS = ("perpendicular", "parallel")

# 历史基线：只检查参考点加 0.2 m 膨胀，用来复现 4/60 与 18/60。
PARKING_PLANNER_SETTINGS = PlannerSettings(
    primitive_length=1.0,
    reverse_cost=1.5,
    interpolation=0.3,
    validation_distance=0.05,
    inflation=0.2,
    collision_model="inflated_point",
)

# 第一步：矩形车身碰撞。60 例几何保持冻结；搜索参数按官方 H-OBCA
# (Zhang et al., CDC 2018 / github H-OBCA hybrid_a_star.jl) 对齐，不是新算法。
#   primitives=11     ← N_STEER=5 → 11 个转角：0, ±δmax/5, …, ±δmax；δmax 仍是 arctan(L/Rmin)≈0.495，κ_max=0.2
#   primitive_length=0.3  ← 官方 XY_GRID_RESOLUTION / 运动弧长
#   reverse_cost=1.0  ← 官方 BACK_COST=0，表示倒车不加额外倍率；我们的代价是长度×倍率，故取 1.0 与前进相同
#   heading_heuristic_gain=2.0  ← 标准非完整航向项，避免无障碍启发式钻进车位后以约 90° 朝向卡死
#   rs_heading_limit=0.6  ← 航向偏差过大时跳过 Reeds-Shepp（精确矩形碰撞下解析衔接几乎总是穿障）
#   max_time=60s      ← 单例搜索上限，失败记为 WS
# 规划与评价使用同一几何谓词：原矩形到障碍的精确欧氏最短距离 >= 0.05 m。
# collision_margin 仍为 0（不用扩大矩形近似）。min_clearance 才是净距阈值。
# max_time / max_expansions 只是运算资源保险，不是时长验收标准。
# rs_goal_radius：靠近目标时仍尝试 Reeds-Shepp，避免仅因航向超限而放弃解析衔接。
# 半径取 3 m（约一车长），避免在通道里对每个扩展都做昂贵的 Reeds-Shepp。
RECTANGLE_PLANNER_SETTINGS = replace(
    PARKING_PLANNER_SETTINGS,
    radius=VEHICLE_CONFIG['Rmin'],
    inflation=PLANNER['inflation_m'],
    collision_model="rectangle",
    collision_margin=0.0,
    min_clearance=0.05,
    sweep_step=0.05,
    primitives=11,
    primitive_length=0.3,
    reverse_cost=1.0,
    heading_heuristic_gain=2.0,
    rs_heading_limit=0.6,
    rs_goal_radius=3.0,
    max_time=PLANNER.get('max_time'),
    max_expansions=200000,
)


def _closed_rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float)


def _wall() -> np.ndarray:
    return _closed_rect(OBSTACLE_X0, WALL_Y0, OBSTACLE_X1, WALL_Y1)


def perpendicular_obstacles(width: float) -> tuple[np.ndarray, ...]:
    half = width / 2.0
    left = _closed_rect(OBSTACLE_X0, 0.0, -half, PERP_DEPTH)
    right = _closed_rect(half, 0.0, OBSTACLE_X1, PERP_DEPTH)
    return left, right, _wall()


def parallel_obstacles(length: float) -> tuple[np.ndarray, ...]:
    half = length / 2.0
    curb = _closed_rect(OBSTACLE_X0, 0.0, OBSTACLE_X1, PARA_BAY_Y0)
    left = _closed_rect(OBSTACLE_X0, PARA_BAY_Y0, -half, PARA_BAY_Y1)
    right = _closed_rect(half, PARA_BAY_Y0, OBSTACLE_X1, PARA_BAY_Y1)
    return curb, left, right, _wall()


def perpendicular_goal() -> np.ndarray:
    # Same as parking_scenario(): rear axle at y=1.3 so the front bumper is at y=5.0.
    return np.array([0.0, 1.3, PERP_GOAL_HEADING])


def parallel_goal() -> np.ndarray:
    return np.array([PARA_GOAL_X, PARA_GOAL_Y, PARA_GOAL_HEADING])


def build_scene(maneuver: str, tightness: str, start: np.ndarray):
    if tightness not in TIGHTNESS_LEVELS:
        raise ValueError(f"unknown tightness: {tightness}")
    if maneuver == "perpendicular":
        obstacles = perpendicular_obstacles(PERP_WIDTHS[tightness])
        goal = perpendicular_goal()
    elif maneuver == "parallel":
        obstacles = parallel_obstacles(PARA_LENGTHS[tightness])
        goal = parallel_goal()
    else:
        raise ValueError(f"unknown maneuver: {maneuver}")
    return _assemble(VEHICLE, BOUNDS, np.asarray(start, dtype=float), goal, obstacles)


def start_grid() -> list[np.ndarray]:
    poses = []
    for x in START_X:
        for y in START_Y:
            poses.append(np.array([x, y, START_PSI], dtype=float))
    return poses


@dataclass(frozen=True)
class CaseSpec:
    method: str
    maneuver: str
    tightness: str
    start: np.ndarray
    is_regression: bool = False
    family: str = "standard"
    vehicle_name: str = "project_4.7x2"
    scene_name: str = ""
    angle_deg: float | None = None

    @property
    def case_id(self) -> str:
        if self.family == "tpcap":
            return f"{self.method}_tpcap_{self.tightness}_{self.vehicle_name}"
        if self.family == "diagonal":
            tag = f"a{int(self.angle_deg)}_x{self.start[0]:g}_y{self.start[1]:g}"
            return f"{self.method}_diagonal_{self.tightness}_{tag}"
        if self.family == "irregular":
            tag = f"{self.scene_name}_x{self.start[0]:g}_y{self.start[1]:g}"
            return f"{self.method}_irregular_{tag}"
        tag = "dissertation" if self.is_regression else f"x{self.start[0]:g}_y{self.start[1]:g}"
        return f"{self.method}_{self.maneuver}_{self.tightness}_{tag}"


def iter_suite_specs(methods=("hybrid_astar", "h_obca")) -> list[CaseSpec]:
    specs = []
    for method in methods:
        for maneuver in MANEUVERS:
            for tightness in TIGHTNESS_LEVELS:
                for start in start_grid():
                    specs.append(CaseSpec(method, maneuver, tightness, start, False))
    return specs


def dissertation_regression_specs(methods=("hybrid_astar", "h_obca")) -> list[CaseSpec]:
    return [
        CaseSpec(method, "perpendicular", "tight", DISSERTATION_START.copy(), True)
        for method in methods
    ]


def slot_vertices_dict() -> dict:
    """Exact obstacle vertices, for the report and for unit tests."""
    data = {"perpendicular": {}, "parallel": {}}
    for name, width in PERP_WIDTHS.items():
        left, right, wall = perpendicular_obstacles(width)
        data["perpendicular"][name] = {
            "width": width,
            "depth": PERP_DEPTH,
            "goal": perpendicular_goal().tolist(),
            "obstacles": {"left": left.tolist(), "right": right.tolist(), "wall": wall.tolist()},
        }
    for name, length in PARA_LENGTHS.items():
        curb, left, right, wall = parallel_obstacles(length)
        data["parallel"][name] = {
            "length": length,
            "depth": PARA_DEPTH,
            "bay_y": [PARA_BAY_Y0, PARA_BAY_Y1],
            "goal": parallel_goal().tolist(),
            "obstacles": {
                "curb": curb.tolist(),
                "left": left.tolist(),
                "right": right.tolist(),
                "wall": wall.tolist(),
            },
        }
    return data


def scene_for_spec(spec: "CaseSpec"):
    """Build the scene for any suite family. The original 60 geometries stay on build_scene()."""
    if spec.family == "diagonal":
        from eval_parking.extra_scenes import build_diagonal_scene
        return build_diagonal_scene(spec.tightness, float(spec.angle_deg), spec.start)
    if spec.family == "irregular":
        from eval_parking.extra_scenes import build_irregular_scene
        return build_irregular_scene(spec.scene_name, spec.start)
    if spec.family == "tpcap":
        from eval_parking.tpcap_cases import scene_from_tpcap, parse_official_cases, PROJECT_VEHICLE
        case_id = int(spec.tightness.replace("case", ""))
        for case in parse_official_cases(spec_vehicle(spec), spec.vehicle_name, 0.05):
            if case.case_id == case_id:
                return scene_from_tpcap(case)
        raise KeyError(f"TPCAP case not found: {spec.tightness}")
    return build_scene(spec.maneuver, spec.tightness, spec.start)


def spec_vehicle(spec: "CaseSpec"):
    from eval_parking.tpcap_cases import PROJECT_VEHICLE, OFFICIAL_BODY_PROJECT_KINEMATICS
    if spec.vehicle_name == "tpcap_official_body":
        return OFFICIAL_BODY_PROJECT_KINEMATICS
    return PROJECT_VEHICLE


def iter_diagonal_specs(methods=("hybrid_astar", "h_obca")) -> list[CaseSpec]:
    from eval_parking.extra_scenes import DIAGONAL_ANGLES, DIAGONAL_STARTS
    specs = []
    for method in methods:
        for angle in DIAGONAL_ANGLES:
            for tightness in TIGHTNESS_LEVELS:
                for start in DIAGONAL_STARTS:
                    specs.append(CaseSpec(
                        method, "diagonal", tightness, np.asarray(start, dtype=float),
                        False, family="diagonal", scene_name=f"diagonal_{int(angle)}",
                        angle_deg=float(angle),
                    ))
    return specs


def iter_irregular_specs(methods=("hybrid_astar", "h_obca")) -> list[CaseSpec]:
    from eval_parking.extra_scenes import IRREGULAR_STARTS, irregular_catalog
    specs = []
    for method in methods:
        for name in irregular_catalog():
            for start in IRREGULAR_STARTS:
                specs.append(CaseSpec(
                    method, "irregular", name, np.asarray(start, dtype=float),
                    False, family="irregular", scene_name=name,
                ))
    return specs


def iter_tpcap_specs(
    methods=("hybrid_astar", "h_obca"),
    vehicle_name: str = "project_4.7x2",
    min_clearance: float = 0.05,
) -> list[CaseSpec]:
    """Official 20 TPCAP scenes, evaluated under the labelled vehicle and clearance."""
    from eval_parking.tpcap_cases import (
        parse_official_cases, PROJECT_VEHICLE as PV, OFFICIAL_BODY_PROJECT_KINEMATICS as OV,
    )
    vehicle = OV if vehicle_name == "tpcap_official_body" else PV
    specs = []
    for case in parse_official_cases(vehicle, vehicle_name, min_clearance):
        for method in methods:
            specs.append(CaseSpec(
                method, "tpcap", f"case{case.case_id:02d}", case.start.copy(),
                False, family="tpcap", vehicle_name=vehicle_name, scene_name=case.name,
            ))
    return specs

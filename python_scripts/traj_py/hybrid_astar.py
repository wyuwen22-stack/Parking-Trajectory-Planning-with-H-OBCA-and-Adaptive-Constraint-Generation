"""Independent Hybrid A*: MATLAB public settings, documented internal choices."""
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import math
import time

import numpy as np
from scipy.ndimage import binary_dilation

from .analysis import vehicle_vertices
from .geometry import aabb_separation, densify_poses, polygon_distance, polygons_overlap, segment_meets_clearance
from .scenario import Scenario
from .reeds_shepp import calc_paths


@dataclass
class PlannerSettings:
    radius: float = 5.0
    primitive_length: float = 2.0
    primitives: int = 5
    forward_cost: float = 1.0
    reverse_cost: float = 3.0
    interpolation: float = 0.5
    validation_distance: float = 0.1
    inflation: float = 1.0
    # inflated_point: 只在膨胀栅格上检查参考点（与现有 MATLAB/Python 基线一致）。
    # rectangle: 用 4.7×2.0 矩形车身对障碍多边形做分离轴（SAT）检测。
    collision_model: str = "inflated_point"
    collision_margin: float = 0.0
    # 0 keeps the original holonomic-with-obstacle heuristic (MATLAB baseline).
    # A positive gain adds gain * Rmin * |Δψ| / π, a standard nonholonomic heading term.
    heading_heuristic_gain: float = 0.0
    # If set, skip Reeds-Shepp analytic expansion when heading error exceeds this (rad).
    rs_heading_limit: float | None = None
    # Still try Reeds-Shepp when the rear-axle is this close to the goal (metres),
    # even if the heading error exceeds rs_heading_limit.
    rs_goal_radius: float = 0.0
    max_expansions: int = 100000
    max_time: float | None = None
    # Exact Euclidean clearance required between the original body rectangle and
    # obstacles. This is not the same as collision_margin (which expands the four
    # half-lengths of the rectangle). 0 keeps overlap-only checking.
    min_clearance: float = 0.0
    # Machine-eps tolerance only. A pose is accepted when
    # distance >= min_clearance - clearance_tol. Do not use 1e-3 here: that
    # would be a 1 mm relaxation of the 0.05 m physical rule.
    clearance_tol: float = 1e-12
    # Sampled-sweep step (max corner travel) used when min_clearance > 0.
    # This is a discrete sample, not a continuous swept-volume guarantee.
    sweep_step: float = 0.05


@dataclass
class PlanResult:
    states: np.ndarray
    expanded: int
    cost: float
    inflated_occupancy: np.ndarray



class SearchStopped(RuntimeError):
    def __init__(self, reason, expanded):
        self.reason, self.expanded = reason, expanded
        super().__init__(f'Hybrid A* stopped: {reason} after {expanded} expansions')


class SearchBudget:
    """Hard wall-clock cap so a search cannot run indefinitely."""

    def __init__(self, max_time: float | None):
        self.max_time = max_time
        self.t0 = time.perf_counter()
        self.expanded = 0

    def remaining(self) -> float:
        if self.max_time is None:
            return float("inf")
        return self.max_time - (time.perf_counter() - self.t0)

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def check(self) -> None:
        if self.expired():
            raise SearchStopped('TIME_LIMIT',self.expanded)


def inflate_occupancy(scene: Scenario, radius: float) -> np.ndarray:
    r = int(np.ceil(radius * scene.resolution))
    offsets = np.arange(-r, r + 1)
    # R2025b occupancy-map convention includes cells touching the disk.
    footprint = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= (r + 0.75) ** 2
    return binary_dilation(scene.occupancy, structure=footprint)


def grid_indices(points, scene):
    col = np.floor((points[:, 0] - scene.XYbounds[0]) * scene.resolution).astype(int)
    row = scene.occupancy.shape[0] - 1 - np.floor((points[:, 1] - scene.XYbounds[2]) * scene.resolution).astype(int)
    return row, col


def collision_free(points, scene, occupancy):
    """参考点是否落在给定占用栅格的空闲单元格内（含地图边界）。"""
    rows, cols = grid_indices(points, scene)
    valid = (rows >= 0) & (cols >= 0) & (rows < occupancy.shape[0]) & (cols < occupancy.shape[1])
    return bool(valid.all() and not occupancy[rows, cols].any())


def _open_polygon(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) >= 2 and np.allclose(vertices[0], vertices[-1], atol=1e-12):
        return vertices[:-1]
    return vertices


def _projection_gap(poly_a, poly_b, axis):
    a = _open_polygon(poly_a) @ axis
    b = _open_polygon(poly_b) @ axis
    return float(max(a.min() - b.max(), b.min() - a.max()))


def _polygon_axes(vertices):
    pts = _open_polygon(vertices)
    axes = []
    for i, p1 in enumerate(pts):
        edge = pts[(i + 1) % len(pts)] - p1
        axis = np.array([edge[1], -edge[0]], dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axes.append(axis / norm)
    return axes


def polygons_overlap(poly_a, poly_b, tol=0.0) -> bool:
    """凸多边形 SAT：某轴间隙大于 tol 则分离。间隙为 0（刚好贴边）视为碰撞。"""
    axes = _polygon_axes(poly_a) + _polygon_axes(poly_b)
    if not axes:
        return False
    return all(_projection_gap(poly_a, poly_b, axis) <= tol for axis in axes)


def _in_bounds(corners, bounds) -> bool:
    return bool(
        corners[:, 0].min() >= bounds[0]
        and corners[:, 0].max() <= bounds[1]
        and corners[:, 1].min() >= bounds[2]
        and corners[:, 1].max() <= bounds[3]
    )


def _is_axis_aligned(poly) -> bool:
    pts = _open_polygon(poly)
    if len(pts) < 3:
        return False
    nxt = np.roll(pts, -1, axis=0)
    edges = nxt - pts
    return bool(np.all((np.abs(edges[:, 0]) < 1e-12) | (np.abs(edges[:, 1]) < 1e-12)))


def _pose_free_aabb(x, y, th, xu, yu, xl, yl, bounds, aabbs) -> bool:
    """旋转矩形对轴对齐障碍的 SAT，热路径里不用 NumPy 分配。"""
    c = math.cos(th)
    s = math.sin(th)
    # 车体四角（车体坐标系 x 沿航向）：(xu,yu), (xu,-yl), (-xl,-yl), (-xl,yu)
    x0 = x + xu * c - yu * s
    y0 = y + xu * s + yu * c
    x1 = x + xu * c - (-yl) * s
    y1 = y + xu * s + (-yl) * c
    x2 = x + (-xl) * c - (-yl) * s
    y2 = y + (-xl) * s + (-yl) * c
    x3 = x + (-xl) * c - yu * s
    y3 = y + (-xl) * s + yu * c
    xmin, xmax, ymin, ymax = bounds
    cx_min = x0 if x0 < x1 else x1
    if x2 < cx_min:
        cx_min = x2
    if x3 < cx_min:
        cx_min = x3
    cx_max = x0 if x0 > x1 else x1
    if x2 > cx_max:
        cx_max = x2
    if x3 > cx_max:
        cx_max = x3
    cy_min = y0 if y0 < y1 else y1
    if y2 < cy_min:
        cy_min = y2
    if y3 < cy_min:
        cy_min = y3
    cy_max = y0 if y0 > y1 else y1
    if y2 > cy_max:
        cy_max = y2
    if y3 > cy_max:
        cy_max = y3
    if cx_min < xmin or cx_max > xmax or cy_min < ymin or cy_max > ymax:
        return False
    origin_x = x * c + y * s   # 障碍投影到车体 x 轴时，参考点的坐标
    origin_y = -x * s + y * c  # 车体 y 轴
    for ox0, ox1, oy0, oy1 in aabbs:
        if cx_max < ox0 or cx_min > ox1 or cy_max < oy0 or cy_min > oy1:
            continue
        # 车体 x 轴 (c, s)：车体区间 [origin_x - xl, origin_x + xu]
        p0 = ox0 * c + oy0 * s
        p1 = ox1 * c + oy0 * s
        p2 = ox1 * c + oy1 * s
        p3 = ox0 * c + oy1 * s
        pmin = p0 if p0 < p1 else p1
        if p2 < pmin:
            pmin = p2
        if p3 < pmin:
            pmin = p3
        pmax = p0 if p0 > p1 else p1
        if p2 > pmax:
            pmax = p2
        if p3 > pmax:
            pmax = p3
        if pmax < origin_x - xl or pmin > origin_x + xu:
            continue
        # 车体 y 轴 (-s, c)：车体区间 [origin_y - yl, origin_y + yu]
        q0 = -ox0 * s + oy0 * c
        q1 = -ox1 * s + oy0 * c
        q2 = -ox1 * s + oy1 * c
        q3 = -ox0 * s + oy1 * c
        qmin = q0 if q0 < q1 else q1
        if q2 < qmin:
            qmin = q2
        if q3 < qmin:
            qmin = q3
        qmax = q0 if q0 > q1 else q1
        if q2 > qmax:
            qmax = q2
        if q3 > qmax:
            qmax = q3
        if qmax < origin_y - yl or qmin > origin_y + yu:
            continue
        return False
    return True




def _point_aabb_distance(px, py, xmin, xmax, ymin, ymax) -> float:
    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.hypot(dx, dy)


def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / length2
    if t <= 0.0:
        return math.hypot(px - ax, py - ay)
    if t >= 1.0:
        return math.hypot(px - bx, py - by)
    return math.hypot(px - ax - t * vx, py - ay - t * vy)


def _convex_distance(poly_a, poly_b) -> float:
    """Exact Euclidean distance between two open convex polygons (no overlap test)."""
    dist = float("inf")
    na, nb = len(poly_a), len(poly_b)
    for i, (px, py) in enumerate(poly_a):
        ax, ay = poly_b[0]
        for j in range(nb):
            bx, by = poly_b[(j + 1) % nb]
            d = _point_segment_distance(px, py, ax, ay, bx, by)
            if d < dist:
                dist = d
            ax, ay = bx, by
    for i, (px, py) in enumerate(poly_b):
        ax, ay = poly_a[0]
        for j in range(na):
            bx, by = poly_a[(j + 1) % na]
            d = _point_segment_distance(px, py, ax, ay, bx, by)
            if d < dist:
                dist = d
            ax, ay = bx, by
    return dist


def _sat_overlap(poly_a, poly_b) -> bool:
    """True if two open convex polygons overlap or touch (zero gap)."""
    for poly in (poly_a, poly_b):
        n = len(poly)
        x0, y0 = poly[-1]
        for i in range(n):
            x1, y1 = poly[i]
            ax, ay = y0 - y1, x1 - x0  # outward-ish normal of edge
            amin = amax = x0 * ax + y0 * ay
            for x, y in poly:
                proj = x * ax + y * ay
                if proj < amin:
                    amin = proj
                elif proj > amax:
                    amax = proj
            bmin = bmax = poly_b[0][0] * ax + poly_b[0][1] * ay if poly is poly_a else poly_a[0][0] * ax + poly_a[0][1] * ay
            other = poly_b if poly is poly_a else poly_a
            # recompute cleanly
            bx0, by0 = other[0]
            bmin = bmax = bx0 * ax + by0 * ay
            for x, y in other:
                proj = x * ax + y * ay
                if proj < bmin:
                    bmin = proj
                elif proj > bmax:
                    bmax = proj
            if amax < bmin or bmax < amin:
                return False
            x0, y0 = x1, y1
    return True


def _pose_clear_aabb(x, y, th, xu, yu, xl, yl, bounds, aabbs, limit) -> bool:
    """Allocation-free exact-clearance test of an oriented rectangle vs AABBs."""
    c = math.cos(th)
    s = math.sin(th)
    x0 = x + xu * c - yu * s
    y0 = y + xu * s + yu * c
    x1 = x + xu * c + yl * s
    y1 = y + xu * s - yl * c
    x2 = x - xl * c + yl * s
    y2 = y - xl * s - yl * c
    x3 = x - xl * c - yu * s
    y3 = y - xl * s + yu * c
    cx_min = x0 if x0 < x1 else x1
    if x2 < cx_min:
        cx_min = x2
    if x3 < cx_min:
        cx_min = x3
    cx_max = x0 if x0 > x1 else x1
    if x2 > cx_max:
        cx_max = x2
    if x3 > cx_max:
        cx_max = x3
    cy_min = y0 if y0 < y1 else y1
    if y2 < cy_min:
        cy_min = y2
    if y3 < cy_min:
        cy_min = y3
    cy_max = y0 if y0 > y1 else y1
    if y2 > cy_max:
        cy_max = y2
    if y3 > cy_max:
        cy_max = y3
    xmin, xmax, ymin, ymax = bounds
    if cx_min < xmin or cx_max > xmax or cy_min < ymin or cy_max > ymax:
        return False
    origin_x = x * c + y * s
    origin_y = -x * s + y * c
    corners = ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
    for ox0, ox1, oy0, oy1 in aabbs:
        dx = ox0 - cx_max if ox0 > cx_max else (cx_min - ox1 if cx_min > ox1 else 0.0)
        dy = oy0 - cy_max if oy0 > cy_max else (cy_min - oy1 if cy_min > oy1 else 0.0)
        if math.hypot(dx, dy) >= limit:
            continue
        # SAT: body x and body y (world axes already covered by the AABB test above
        # when the boxes overlap; if they overlap in world AABB we still need body axes).
        p0 = ox0 * c + oy0 * s
        p1 = ox1 * c + oy0 * s
        p2 = ox1 * c + oy1 * s
        p3 = ox0 * c + oy1 * s
        pmin = p0 if p0 < p1 else p1
        if p2 < pmin:
            pmin = p2
        if p3 < pmin:
            pmin = p3
        pmax = p0 if p0 > p1 else p1
        if p2 > pmax:
            pmax = p2
        if p3 > pmax:
            pmax = p3
        body_xmin, body_xmax = origin_x - xl, origin_x + xu
        q0 = -ox0 * s + oy0 * c
        q1 = -ox1 * s + oy0 * c
        q2 = -ox1 * s + oy1 * c
        q3 = -ox0 * s + oy1 * c
        qmin = q0 if q0 < q1 else q1
        if q2 < qmin:
            qmin = q2
        if q3 < qmin:
            qmin = q3
        qmax = q0 if q0 > q1 else q1
        if q2 > qmax:
            qmax = q2
        if q3 > qmax:
            qmax = q3
        body_ymin, body_ymax = origin_y - yl, origin_y + yu
        overlap = not (pmax < body_xmin or pmin > body_xmax or qmax < body_ymin or qmin > body_ymax)
        if overlap:
            return False
        dist = min(
            _point_aabb_distance(px, py, ox0, ox1, oy0, oy1) for px, py in corners
        )
        for bx, by in ((ox0, oy0), (ox0, oy1), (ox1, oy0), (ox1, oy1)):
            lx = c * (bx - x) + s * (by - y)
            ly = -s * (bx - x) + c * (by - y)
            d = _point_aabb_distance(lx, ly, -xl, xu, -yl, yu)
            if d < dist:
                dist = d
        if dist < limit:
            return False
    return True


def _pose_clear_convex(x, y, th, xu, yu, xl, yl, bounds, obstacles, aabbs, limit) -> bool:
    """Exact-clearance test vs general convex polygons. obstacles are open vertex tuples."""
    c = math.cos(th)
    s = math.sin(th)
    corners = (
        (x + xu * c - yu * s, y + xu * s + yu * c),
        (x + xu * c + yl * s, y + xu * s - yl * c),
        (x - xl * c + yl * s, y - xl * s - yl * c),
        (x - xl * c - yu * s, y - xl * s + yu * c),
    )
    cx_min = min(p[0] for p in corners)
    cx_max = max(p[0] for p in corners)
    cy_min = min(p[1] for p in corners)
    cy_max = max(p[1] for p in corners)
    xmin, xmax, ymin, ymax = bounds
    if cx_min < xmin or cx_max > xmax or cy_min < ymin or cy_max > ymax:
        return False
    for obs, box in zip(obstacles, aabbs):
        ox0, ox1, oy0, oy1 = box
        dx = ox0 - cx_max if ox0 > cx_max else (cx_min - ox1 if cx_min > ox1 else 0.0)
        dy = oy0 - cy_max if oy0 > cy_max else (cy_min - oy1 if cy_min > oy1 else 0.0)
        if math.hypot(dx, dy) >= limit:
            continue
        if _sat_overlap(corners, obs):
            return False
        if _convex_distance(corners, obs) < limit:
            return False
    return True


def rectangle_collision_free(points, scene, margin=0.0, min_clearance=0.0, sweep_step=None) -> bool:
    """矩形车身在每个位姿都不越界、不与障碍多边形重叠。

    margin > 0 时，把前后左右四个半长都外扩 margin，相当于矩形缓冲，
    不是半径为 margin 的圆形缓冲。这与 min_clearance 不是同一件事：
    min_clearance 要求原矩形到障碍的精确欧氏最短距离达到阈值。
    障碍若都是轴对齐矩形，走无分配的 SAT；否则退回通用凸多边形 SAT。
    贴边（间隙为 0）视为碰撞。
    """
    if min_clearance > 0.0:
        step = 0.05 if sweep_step is None else float(sweep_step)
        poses = np.asarray(points, dtype=float)
        if poses.ndim == 1:
            poses = poses[None, :]
        if sweep_step:
            poses = densify_poses(poses, scene.veh.ego, step=step)
        return segment_meets_clearance(
            poses, scene.obs, scene.veh.ego, float(min_clearance) - 1e-12,
            step=step, bounds=scene.XYbounds,
        )
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    ego = np.asarray(scene.veh.ego, dtype=float) + float(margin)
    xu, yu, xl, yl = (float(ego[0]), float(ego[1]), float(ego[2]), float(ego[3]))
    bounds = (float(scene.XYbounds[0]), float(scene.XYbounds[1]),
              float(scene.XYbounds[2]), float(scene.XYbounds[3]))
    open_obs = [_open_polygon(obs) for obs in scene.obs]
    if all(_is_axis_aligned(obs) for obs in open_obs):
        aabbs = tuple(
            (float(obs[:, 0].min()), float(obs[:, 0].max()),
             float(obs[:, 1].min()), float(obs[:, 1].max()))
            for obs in open_obs
        )
        for pose in points:
            if not _pose_free_aabb(float(pose[0]), float(pose[1]), float(pose[2]),
                                   xu, yu, xl, yl, bounds, aabbs):
                return False
        return True
    for pose in points:
        corners = vehicle_vertices(pose[:3], ego)
        if not _in_bounds(corners, bounds):
            return False
        car_min = corners.min(axis=0)
        car_max = corners.max(axis=0)
        for obs in open_obs:
            omin, omax = obs.min(axis=0), obs.max(axis=0)
            if car_max[0] < omin[0] or car_min[0] > omax[0] or car_max[1] < omin[1] or car_min[1] > omax[1]:
                continue
            if polygons_overlap(corners, obs, tol=0.0):
                return False
    return True


def make_rectangle_checker(scene, margin=0.0, min_clearance=0.0, sweep_step=None, budget=None):
    """预先整理障碍包围盒，避免搜索热路径里反复做多边形预处理。"""
    if min_clearance > 0.0:
        limit = float(min_clearance) - 1e-12
        ego0 = np.asarray(scene.veh.ego, dtype=float)
        xu, yu, xl, yl = (float(ego0[0]), float(ego0[1]), float(ego0[2]), float(ego0[3]))
        bounds = (float(scene.XYbounds[0]), float(scene.XYbounds[1]),
                  float(scene.XYbounds[2]), float(scene.XYbounds[3]))
        open_obs = [_open_polygon(obs) for obs in scene.obs]
        aabbs = tuple(
            (float(obs[:, 0].min()), float(obs[:, 0].max()),
             float(obs[:, 1].min()), float(obs[:, 1].max()))
            for obs in open_obs
        )
        use_aabb = all(_is_axis_aligned(obs) for obs in open_obs)
        obs_pts = None if use_aabb else tuple(
            tuple((float(pt[0]), float(pt[1])) for pt in obs) for obs in open_obs
        )

        def _pose_ok(pose):
            x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
            if use_aabb:
                return _pose_clear_aabb(x, y, th, xu, yu, xl, yl, bounds, aabbs, limit)
            return _pose_clear_convex(x, y, th, xu, yu, xl, yl, bounds, obs_pts, aabbs, limit)

        def check_clear(points):
            if budget is not None:
                budget.check()
            poses = np.asarray(points, dtype=float)
            if poses.ndim == 1:
                poses = poses[None, :]
            prev = None
            for pose in poses:
                if prev is not None:
                    mid_th = float(prev[2]) + 0.5 * math.atan2(
                        math.sin(float(pose[2]) - float(prev[2])),
                        math.cos(float(pose[2]) - float(prev[2])),
                    )
                    mid = (0.5 * (float(prev[0]) + float(pose[0])),
                           0.5 * (float(prev[1]) + float(pose[1])),
                           mid_th)
                    if not _pose_ok(mid):
                        return False
                if not _pose_ok(pose):
                    return False
                prev = pose
            return True

        return check_clear

    ego = np.asarray(scene.veh.ego, dtype=float) + float(margin)
    xu, yu, xl, yl = (float(ego[0]), float(ego[1]), float(ego[2]), float(ego[3]))
    bounds = (float(scene.XYbounds[0]), float(scene.XYbounds[1]),
              float(scene.XYbounds[2]), float(scene.XYbounds[3]))
    open_obs = [_open_polygon(obs) for obs in scene.obs]
    if not all(_is_axis_aligned(obs) for obs in open_obs):
        def check(points):
            return rectangle_collision_free(points, scene, margin)
        return check
    aabbs = tuple(
        (float(obs[:, 0].min()), float(obs[:, 0].max()),
         float(obs[:, 1].min()), float(obs[:, 1].max()))
        for obs in open_obs
    )

    def check(points):
        if budget is not None:
            budget.check()
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        for pose in points:
            if not _pose_free_aabb(
                float(pose[0]), float(pose[1]), float(pose[2]),
                xu, yu, xl, yl, bounds, aabbs,
            ):
                return False
        return True

    return check


def path_collision_free(points, scene, settings: PlannerSettings, occupancy=None) -> bool:
    model = settings.collision_model
    if model == "rectangle":
        return rectangle_collision_free(
            points, scene, settings.collision_margin,
            min_clearance=settings.min_clearance, sweep_step=settings.sweep_step,
        )
    if model != "inflated_point":
        raise ValueError(f"unknown collision_model: {model}")
    if occupancy is None:
        raise ValueError("inflated_point collision requires an occupancy grid")
    return collision_free(points, scene, occupancy)


def _distance_map(scene, occupancy):
    distances = np.full(occupancy.shape, np.inf)
    r, c = (int(a[0]) for a in grid_indices(scene.goal[None, :], scene))
    distances[r, c] = 0
    queue = [(0., r, c)]
    moves = [(dr, dc, math.hypot(dr, dc)/scene.resolution) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while queue:
        value, r, c = heappop(queue)
        if value != distances[r, c]:
            continue
        for dr, dc, length in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < occupancy.shape[0] and 0 <= nc < occupancy.shape[1] and not occupancy[nr, nc]:
                new = value + length
                if new < distances[nr, nc]:
                    distances[nr, nc] = new
                    heappush(queue, (new, nr, nc))
    return distances



def goal_component_escapes(scene: Scenario, settings: PlannerSettings,
                           escape_distance: float = 2.0,
                           max_expansions: int = 20000,
                           max_time: float = 0.5) -> tuple[bool, str]:
    """Backward flood from the goal with the planner's kinematic limits.

    Discrete backward flood used as a *diagnostic* heuristic only.

    Returns (True, reason) if a pose at least escape_distance from the goal is
    reached, or if the set is not exhausted in the budget (inconclusive).
    Returns (False, reason) when this finite grid is exhausted inside the
    escape radius. That is **not** a continuous kinematic infeasibility proof:
    bins, a discrete steer set, mid+end-only collision, first-arrival pruning,
    and a short time/expansion cap can all miss a feasible path. Callers must
    not skip formal Hybrid A* search or label the case unsolvable from this
    result alone.
    """
    limit_clearance = float(settings.min_clearance)
    is_free = make_rectangle_checker(
        scene, settings.collision_margin,
        min_clearance=limit_clearance, sweep_step=settings.sweep_step, budget=None,
    ) if settings.collision_model == "rectangle" else None
    if is_free is None:
        return True, "pre-check skipped for non-rectangle collision model"
    goal = np.asarray(scene.goal, dtype=float)
    primitive_length = min(0.1, float(settings.primitive_length))
    xy_bin = 0.05
    th_bin = math.pi / 90.0  # 2 deg
    def key(pose):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        return (int(round(x / xy_bin)), int(round(y / xy_bin)),
                int(round(((th + math.pi) % (2 * math.pi)) / th_bin)))
    from heapq import heappop, heappush
    from itertools import count
    nodes = [goal.copy()]
    best = {key(goal): 0.0}
    serial = count()
    queue = [(0.0, next(serial), 0)]
    expanded = 0
    max_dist = 0.0
    t0 = time.perf_counter()
    curvatures = np.linspace(-1.0 / settings.radius, 1.0 / settings.radius, settings.primitives)
    nstep = max(2, int(math.ceil(primitive_length / max(settings.validation_distance, 1e-3))) + 1)
    while queue and expanded < max_expansions and (time.perf_counter() - t0) < max_time:
        cost, _, index = heappop(queue)
        pose = nodes[index]
        expanded += 1
        dist = float(math.hypot(pose[0] - goal[0], pose[1] - goal[1]))
        if dist > max_dist:
            max_dist = dist
        if dist >= escape_distance:
            return True, f"escaped to {dist:.3f} m after {expanded} expansions"
        for direction in (1, -1):
            ds = direction * np.linspace(0.0, primitive_length, nstep)[1:]
            th0 = float(pose[2])
            x0, y0 = float(pose[0]), float(pose[1])
            for curvature in curvatures:
                theta = th0 + curvature * ds
                if abs(curvature) < 1e-12:
                    xs = x0 + ds * math.cos(th0)
                    ys = y0 + ds * math.sin(th0)
                else:
                    xs = x0 + (np.sin(theta) - math.sin(th0)) / curvature
                    ys = y0 - (np.cos(theta) - math.cos(th0)) / curvature
                end = np.array([float(xs[-1]), float(ys[-1]), float(theta[-1])])
                mid = np.array([float(xs[len(ds)//2]), float(ys[len(ds)//2]), float(theta[len(ds)//2])])
                if not is_free(np.vstack([mid, end])):
                    continue
                new_cost = cost + primitive_length
                sk = key(end)
                if new_cost >= best.get(sk, np.inf) - 1e-12:
                    continue
                best[sk] = new_cost
                nodes.append(end)
                heappush(queue, (new_cost, next(serial), len(nodes) - 1))
    if not queue:
        return False, (
            f"heuristic pre-check exhausted under Rmin={settings.radius} after "
            f"{expanded} expansions, max displacement {max_dist:.3f} m < {escape_distance} m "
            f"(discrete flood only; not a proof of infeasibility; formal search must still run)"
        )
    return True, f"inconclusive after {expanded} expansions, max displacement {max_dist:.3f} m"


def plan_hybrid_astar(scene: Scenario, settings: PlannerSettings) -> PlanResult:
    occ = inflate_occupancy(scene, settings.inflation)
    # 启发式距离场仍用膨胀栅格；是否碰撞由 collision_model 单独决定。
    budget = SearchBudget(settings.max_time)
    if settings.collision_model == "rectangle":
        is_free = make_rectangle_checker(
            scene, settings.collision_margin,
            min_clearance=settings.min_clearance,
            sweep_step=settings.sweep_step,
            budget=budget,
        )
    elif settings.collision_model == "inflated_point":
        def is_free(points, _occ=occ, _scene=scene):
            return collision_free(points, _scene, _occ)
    else:
        raise ValueError(f"unknown collision_model: {settings.collision_model}")
    # Check endpoints separately. Do not densify the line from start to goal:
    # that segment is not a planned path and typically goes through obstacles.
    if not is_free(np.asarray(scene.start, dtype=float).reshape(1, -1)) or not is_free(np.asarray(scene.goal, dtype=float).reshape(1, -1)):
        raise ValueError('Start or goal is in collision under the active collision model')
    # Independent stage: no goal-component flood or heuristic hard rejection.
    distances = _distance_map(scene, occ)

    def key(pose):
        row, col = grid_indices(pose[None, :], scene)
        heading = int(np.floor((pose[2] % (2*np.pi)) / (np.pi/36))) % 72
        return int(row[0]), int(col[0]), heading

    def heading_error(pose):
        return abs(math.atan2(math.sin(pose[2] - scene.goal[2]), math.cos(pose[2] - scene.goal[2])))

    def heuristic(pose):
        r, c, _ = key(pose)
        h = max(float(distances[r, c]), float(np.linalg.norm(pose[:2] - scene.goal[:2])))
        if settings.heading_heuristic_gain > 0.0:
            h += settings.heading_heuristic_gain * settings.radius * heading_error(pose) / math.pi
        return h

    serial = count()
    # Each immutable record stores its parent, incoming dense segment and g cost.
    nodes = [(-1, scene.start[None, :].copy(), 0.)]
    best = {key(scene.start): 0.}
    queue = [(heuristic(scene.start), next(serial), 0)]
    expanded = 0
    curvature_values = np.linspace(-1/settings.radius, 1/settings.radius, settings.primitives)
    while queue and expanded < settings.max_expansions:
        try:
            budget.check()
        except RuntimeError:
            raise RuntimeError(f'Hybrid A* stopped at the resource time cap after {expanded} expansions')
        _, _, index = heappop(queue)
        parent, segment, cost = nodes[index]
        pose = segment[-1]
        if cost > best.get(key(pose), np.inf) + 1e-12:
            continue
        expanded += 1
        budget.expanded = expanded
        xy_goal = float(np.linalg.norm(pose[:2] - scene.goal[:2]))
        # Accept a node already inside the evaluation terminal box (0.05 m / 0.05 rad)
        # and append the exact goal if the short closing segment is free.
        if (abs(float(pose[0]) - scene.goal[0]) <= 0.05
                and abs(float(pose[1]) - scene.goal[1]) <= 0.05
                and heading_error(pose) <= 0.05):
            connection = np.vstack([pose, scene.goal])
            if is_free(connection):
                pieces = [connection]
                node = index
                while node >= 0:
                    previous, part, _ = nodes[node]
                    pieces.append(part)
                    node = previous
                dense = np.vstack(pieces[::-1])
                length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1))]
                keep = np.r_[True, np.diff(length) > 1e-10]
                dense, length = dense[keep], length[keep]
                if length[-1] <= 1e-12:
                    states = np.vstack([scene.start, scene.goal])
                else:
                    sample = np.r_[np.arange(0, length[-1], settings.interpolation), length[-1]]
                    states = np.column_stack([np.interp(sample, length, dense[:, i] if i < 2 else np.unwrap(dense[:, i])) for i in range(3)])
                states[0] = scene.start
                states[-1] = scene.goal
                return PlanResult(states, expanded, cost, occ)
        near_goal = settings.rs_goal_radius > 0.0 and xy_goal <= settings.rs_goal_radius
        heading_ok = settings.rs_heading_limit is None or heading_error(pose) <= settings.rs_heading_limit
        try_rs = heading_ok or near_goal
        # Near the goal, try RS on every expansion; farther away, keep the 1-in-5 cadence.
        if try_rs and (near_goal or expanded % 5 == 1):
            budget.check()
            paths = calc_paths(*pose, *scene.goal, 1/settings.radius, settings.validation_distance)
            paths.sort(key=lambda p: sum(abs(d) * (settings.forward_cost if d >= 0 else settings.reverse_cost) for d in p.lengths))
            for path in paths:
                connection = np.column_stack((path.x, path.y, path.yaw))
                if is_free(connection):
                    pieces = [connection]
                    node = index
                    while node >= 0:
                        previous, part, _ = nodes[node]
                        pieces.append(part)
                        node = previous
                    dense = np.vstack(pieces[::-1])
                    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1))]
                    keep = np.r_[True, np.diff(length) > 1e-10]
                    dense, length = dense[keep], length[keep]
                    sample = np.r_[np.arange(0, length[-1], settings.interpolation), length[-1]]
                    states = np.column_stack([np.interp(sample, length, dense[:, i] if i < 2 else np.unwrap(dense[:, i])) for i in range(3)])
                    states[0] = scene.start
                    states[-1] = scene.goal
                    connection_cost = sum(abs(d) * (settings.forward_cost if d >= 0 else settings.reverse_cost) for d in path.lengths)
                    return PlanResult(states, expanded, cost + connection_cost, occ)
        for direction in (1, -1):
            ds = direction * np.linspace(0, settings.primitive_length, int(np.ceil(settings.primitive_length/settings.validation_distance)) + 1)[1:]
            for curvature in curvature_values:
                theta = pose[2] + curvature * ds
                if abs(curvature) < 1e-12:
                    x, y = pose[0] + ds*np.cos(pose[2]), pose[1] + ds*np.sin(pose[2])
                else:
                    x = pose[0] + (np.sin(theta)-np.sin(pose[2]))/curvature
                    y = pose[1] - (np.cos(theta)-np.cos(pose[2]))/curvature
                part = np.column_stack((x, y, theta))
                if not is_free(part):
                    continue
                new_cost = cost + settings.primitive_length * (settings.forward_cost if direction == 1 else settings.reverse_cost)
                state_key = key(part[-1])
                if new_cost >= best.get(state_key, np.inf) - 1e-12:
                    continue
                best[state_key] = new_cost
                nodes.append((index, part, new_cost))
                heappush(queue, (new_cost + heuristic(part[-1]), next(serial), len(nodes)-1))
    raise SearchStopped('EXPANSION_LIMIT' if queue else 'QUEUE_EXHAUSTED',expanded)

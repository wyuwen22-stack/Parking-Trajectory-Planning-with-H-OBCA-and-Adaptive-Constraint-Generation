"""Author-designed diagonal and irregular parking scenes.

These are not TPCAP cases. The original 60 perpendicular/parallel
geometries are not modified. Dimensions that are reused (vehicle, aisle,
slot widths) are taken from eval_parking.scenarios.
"""

from __future__ import annotations

import math

import numpy as np

from eval_parking.scenarios import (
    AISLE_Y0,
    BOUNDS,
    PERP_DEPTH,
    PERP_WIDTHS,
    TIGHTNESS_LEVELS,
    VEHICLE,
    WALL_Y0,
    WALL_Y1,
    _closed_rect,
    _wall,
)
from eval_parking.tpcap_cases import assemble_general
from traj_py.geometry import close_vertices, convex_decompose

# Slot headings, measured from +x (parallel is 0, perpendicular is 90).
DIAGONAL_ANGLES = (30, 45, 60)
# Four starts covering left/right and two aisle depths. Not the full 10-point grid,
# so the added set stays finishable in one night while still covering each bin.
DIAGONAL_STARTS = (
    np.array([-8.0, 8.5, 0.0]),
    np.array([-4.0, 7.5, 0.0]),
    np.array([4.0, 7.5, 0.0]),
    np.array([8.0, 8.5, 0.0]),
)
IRREGULAR_STARTS = (
    np.array([-8.0, 8.5, 0.0]),
    np.array([0.0, 7.5, 0.0]),
    np.array([8.0, 8.5, 0.0]),
)
MOUTH = np.array([0.0, AISLE_Y0], dtype=float)



def _clip_halfplane(points: np.ndarray, a, b, c, origin_inside) -> np.ndarray:
    """Sutherland-Hodgman clip: keep ax + by <= c, using origin_inside to set the side."""
    pts = np.asarray(points, dtype=float)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    # ensure inequality so origin_inside is feasible
    if a * origin_inside[0] + b * origin_inside[1] > c:
        a, b, c = -a, -b, -c

    def inside(p):
        return a * p[0] + b * p[1] <= c + 1e-12

    def intersect(p, q):
        dp = a * p[0] + b * p[1] - c
        dq = a * q[0] + b * q[1] - c
        t = dp / (dp - dq)
        return p + t * (q - p)

    out = []
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        pin, qin = inside(p), inside(q)
        if pin and qin:
            out.append(q)
        elif pin and not qin:
            out.append(intersect(p, q))
        elif (not pin) and qin:
            out.append(intersect(p, q))
            out.append(q)
    if len(out) < 3:
        return close_vertices(np.asarray(points, dtype=float))
    return close_vertices(np.asarray(out, dtype=float))


def _rotate(points: np.ndarray, angle: float, origin: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rel = np.asarray(points, dtype=float) - origin
    rot = np.column_stack((c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]))
    return rot + origin


def diagonal_obstacles(width: float, angle_deg: float) -> tuple[np.ndarray, ...]:
    """Rotate the tight/medium/wide perpendicular bay about the aisle mouth."""
    from eval_parking.scenarios import perpendicular_obstacles

    angle = math.radians(float(angle_deg))
    delta = angle - math.pi / 2.0
    left, right, wall = perpendicular_obstacles(width)
    if abs(delta) < 1e-15:
        return left, right, wall
    # Keep the aisle (y > AISLE_Y0) free of the rotated bay, so the standard
    # aisle starts remain valid. The wall is unchanged.
    left_r = close_vertices(_rotate(left, delta, MOUTH))
    right_r = close_vertices(_rotate(right, delta, MOUTH))
    aisle_origin = np.array([0.0, AISLE_Y0 - 1.0])
    left_r = _clip_halfplane(left_r, 0.0, 1.0, AISLE_Y0, aisle_origin)
    right_r = _clip_halfplane(right_r, 0.0, 1.0, AISLE_Y0, aisle_origin)
    return left_r, right_r, wall


def diagonal_goal(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    heading = np.array([math.cos(angle), math.sin(angle)])
    xu = float(VEHICLE.ego[0])
    goal_xy = MOUTH - xu * heading
    return np.array([goal_xy[0], goal_xy[1], angle], dtype=float)


def build_diagonal_scene(tightness: str, angle_deg: float, start: np.ndarray):
    if tightness not in TIGHTNESS_LEVELS:
        raise ValueError(f"unknown tightness: {tightness}")
    if int(angle_deg) not in DIAGONAL_ANGLES:
        raise ValueError(f"unsupported diagonal angle: {angle_deg}")
    obstacles = diagonal_obstacles(PERP_WIDTHS[tightness], angle_deg)
    pieces = []
    for obs in obstacles:
        pieces.extend(convex_decompose(obs))
    return assemble_general(VEHICLE, BOUNDS, start, diagonal_goal(angle_deg), pieces)


def irregular_catalog() -> dict:
    """Named author-designed irregular scenes (convex and non-convex)."""
    from eval_parking.scenarios import perpendicular_obstacles

    left, right, wall = perpendicular_obstacles(PERP_WIDTHS["medium"])
    # 1) Convex hexagonal pillar in the aisle, clear of the three chosen starts
    #    (vehicle at (0, 7.5, 0) occupies roughly x in [-1, 3.7], y in [6.5, 8.5]).
    hexagon = close_vertices(np.array([
        [5.3, 6.4], [6.4, 6.9], [6.4, 8.1], [5.3, 8.5], [4.3, 8.1], [4.3, 6.9],
    ], dtype=float))
    # 2) Convex parallelogram bay sides. Mouth width 2.8 m, same as the medium
    #    perpendicular slot, so the unchanged medium goal stays 0.40 m from each side.
    para_left = close_vertices(np.array([
        [-12.0, 0.0], [-2.2, 0.0], [-1.4, 5.0], [-12.0, 5.0],
    ], dtype=float))
    para_right = close_vertices(np.array([
        [2.2, 0.0], [12.0, 0.0], [12.0, 5.0], [1.4, 5.0],
    ], dtype=float))
    # 3) Concave L-wall kept left of x=-4 so starts at x=0 and x=-8,y=8.5 stay free.
    #    The vertical tongue ends at x=-4.2, y in [5.0, 7.0].
    l_wall = close_vertices(np.array([
        [-12.0, 5.0], [-5.5, 5.0], [-5.5, 6.8], [-4.2, 6.8],
        [-4.2, 7.0], [-12.0, 7.0],
    ], dtype=float))
    # 4) Concave notched island on the right side of the aisle.
    notch = close_vertices(np.array([
        [4.3, 6.2], [6.6, 6.2], [6.6, 9.4], [4.3, 9.4],
        [4.3, 8.6], [5.8, 8.6], [5.8, 7.0], [4.3, 7.0],
    ], dtype=float))
    medium_goal = np.array([0.0, 1.3, np.pi / 2], dtype=float)
    return {
        "convex_hexagon": {
            "obstacles": (left, right, wall, hexagon),
            "goal": medium_goal,
            "convexity": "convex_added_obstacle",
            "note": "Author-designed convex hexagonal pillar in the aisle of a medium perpendicular bay.",
        },
        "convex_parallelogram": {
            "obstacles": (para_left, para_right, wall),
            "goal": medium_goal,
            "convexity": "convex",
            "note": "Author-designed convex parallelogram slot sides, medium-scale opening.",
        },
        "concave_L": {
            "obstacles": (left, right, wall, l_wall),
            "goal": medium_goal,
            "convexity": "nonconvex_decomposed",
            "note": "Author-designed concave L-wall on the left approach; decomposed into convex pieces for OBCA.",
        },
        "concave_notch": {
            "obstacles": (left, right, wall, notch),
            "goal": medium_goal,
            "convexity": "nonconvex_decomposed",
            "note": "Author-designed concave notched island in the aisle; decomposed into convex pieces for OBCA.",
        },
    }


def build_irregular_scene(name: str, start: np.ndarray):
    catalog = irregular_catalog()
    if name not in catalog:
        raise ValueError(f"unknown irregular scene: {name}")
    item = catalog[name]
    pieces = []
    for obs in item["obstacles"]:
        pieces.extend(convex_decompose(obs))
    return assemble_general(VEHICLE, BOUNDS, start, item["goal"], pieces)

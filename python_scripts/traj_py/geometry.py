"""Exact polygon distance, overlap, convex decomposition, and pose clearance.

The 0.05 m safety rule uses the Euclidean shortest distance between the
original 4.7 m x 2.0 m vehicle rectangle and each obstacle polygon. It does
not enlarge the vehicle into a buffered rectangle (that would be a
different, conservative approximation).
"""

from __future__ import annotations

import math

import numpy as np

from traj_py.analysis import vehicle_vertices


def open_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) >= 2 and np.allclose(vertices[0], vertices[-1], atol=1e-12):
        return vertices[:-1]
    return vertices


def close_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = open_vertices(vertices)
    if len(vertices) == 0:
        return vertices.reshape(0, 2)
    return np.vstack((vertices, vertices[0]))


def signed_area(vertices: np.ndarray) -> float:
    pts = open_vertices(vertices)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    pts = open_vertices(vertices)
    if signed_area(pts) < 0:
        pts = pts[::-1]
    return pts


def _unique_axes(polygons) -> list[np.ndarray]:
    axes = []
    for poly in polygons:
        pts = open_vertices(poly)
        for i, p1 in enumerate(pts):
            p2 = pts[(i + 1) % len(pts)]
            edge = p2 - p1
            axis = np.array([edge[1], -edge[0]], dtype=float)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            if any(abs(abs(axis @ old) - 1.0) < 1e-9 for old in axes):
                continue
            axes.append(axis)
    return axes


def _projection_gap(poly_a: np.ndarray, poly_b: np.ndarray, axis: np.ndarray) -> float:
    a = open_vertices(poly_a) @ axis
    b = open_vertices(poly_b) @ axis
    return float(max(a.min() - b.max(), b.min() - a.max()))


def _convex_pieces(poly: np.ndarray) -> list[np.ndarray]:
    """Return convex pieces; already-convex input is kept as a single piece."""
    if is_convex_polygon(poly):
        return [poly]
    return convex_decompose(poly)


def _convex_polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray, tol: float = 0.0) -> bool:
    """Convex SAT: overlap if no axis separates the polygons by more than `tol`."""
    axes = _unique_axes((poly_a, poly_b))
    if not axes:
        return False
    return all(_projection_gap(poly_a, poly_b, axis) <= tol for axis in axes)


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray, tol: float = 0.0) -> bool:
    """Exact overlap. SAT is only valid for convex polygons, so a non-convex
    argument is decomposed first and the test is the union of its pieces.
    """
    pieces_a = _convex_pieces(poly_a)
    pieces_b = _convex_pieces(poly_b)
    return any(
        _convex_polygons_overlap(a, b, tol) for a in pieces_a for b in pieces_b
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    edge = b - a
    denom = float(edge @ edge)
    t = 0.0 if denom < 1e-18 else float(np.clip((point - a) @ edge / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * edge)))


def _convex_polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Minimum vertex-to-edge distance between two convex polygons (both directions)."""
    pa = open_vertices(poly_a)
    pb = open_vertices(poly_b)
    distance = np.inf
    for src, dst in ((pa, pb), (pb, pa)):
        for point in src:
            for i, a in enumerate(dst):
                distance = min(distance, _point_segment_distance(point, a, dst[(i + 1) % len(dst)]))
    return float(distance)


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Minimum Euclidean distance between two polygons (0 if they overlap).

    Non-convex arguments are decomposed first. Distance is the min over convex
    pieces; overlap of any piece pair is reported as 0.
    """
    pieces_a = _convex_pieces(poly_a)
    pieces_b = _convex_pieces(poly_b)
    if any(_convex_polygons_overlap(a, b, 0.0) for a in pieces_a for b in pieces_b):
        return 0.0
    return min(_convex_polygon_distance(a, b) for a in pieces_a for b in pieces_b)


def aabb_separation(a, b) -> float:
    """Lower bound on distance between two axis-aligned boxes (0 if they overlap)."""
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    if dx > 0.0 and dy > 0.0:
        return math.hypot(dx, dy)
    return max(dx, dy)


def is_convex_polygon(vertices: np.ndarray, tol: float = 1e-12) -> bool:
    pts = open_vertices(vertices)
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) <= tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _is_ear(pts: np.ndarray, i: int) -> bool:
    n = len(pts)
    a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
    if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) <= 1e-14:
        return False
    triangle = np.vstack((a, b, c))
    for j, p in enumerate(pts):
        if j in ((i - 1) % n, i, (i + 1) % n):
            continue
        if _point_in_triangle(p, a, b, c):
            return False
    return True


def _point_in_triangle(p, a, b, c, tol: float = 1e-12) -> bool:
    v0, v1, v2 = c - a, b - a, p - a
    dot00, dot01, dot02 = v0 @ v0, v0 @ v1, v0 @ v2
    dot11, dot12 = v1 @ v1, v1 @ v2
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return u >= -tol and v >= -tol and u + v <= 1.0 + tol


def convex_decompose(vertices: np.ndarray) -> list[np.ndarray]:
    """Decompose a simple polygon into convex closed rings (ear clipping).

    Already-convex input is returned as a single piece. Non-convex input must
    be decomposed before OBCA halfspace constraints are built. Triangles are
    merged into larger convex pieces when the union stays convex.
    """
    pts = ensure_ccw(vertices)
    pts = _drop_near_duplicates(pts)
    if len(pts) < 3:
        return []
    if is_convex_polygon(pts):
        return [close_vertices(pts)]
    remaining = pts.copy()
    triangles = []
    guard = 0
    while len(remaining) > 3 and guard < 10000:
        guard += 1
        ear = None
        for i in range(len(remaining)):
            if _is_ear(remaining, i):
                ear = i
                break
        if ear is None:
            # Fall back: keep the remaining polygon if convex, else skip a vertex.
            if is_convex_polygon(remaining):
                triangles.append(remaining.copy())
                remaining = remaining[:0]
                break
            remaining = np.vstack((remaining[1:], remaining[:1]))
            continue
        i = ear
        a, b, c = remaining[(i - 1) % len(remaining)], remaining[i], remaining[(i + 1) % len(remaining)]
        triangles.append(np.vstack((a, b, c)))
        remaining = np.delete(remaining, i, axis=0)
    if len(remaining) >= 3:
        triangles.append(remaining.copy())
    return _merge_convex_pieces([close_vertices(t) for t in triangles if len(open_vertices(t)) >= 3])


def _drop_near_duplicates(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) > tol:
            kept.append(p)
    if len(kept) > 1 and np.linalg.norm(kept[0] - kept[-1]) <= tol:
        kept = kept[:-1]
    return np.asarray(kept, dtype=float)


def _merge_convex_pieces(pieces: list[np.ndarray]) -> list[np.ndarray]:
    """Greedy merge of triangles that share an edge and stay convex."""
    work = [open_vertices(p) for p in pieces]
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(work)
        for i, a in enumerate(work):
            if used[i]:
                continue
            acc = a
            used[i] = True
            grew = True
            while grew:
                grew = False
                for j, b in enumerate(work):
                    if used[j]:
                        continue
                    union = _try_merge_convex(acc, b)
                    if union is not None:
                        acc = union
                        used[j] = True
                        grew = True
                        changed = True
            merged.append(acc)
        work = merged
    return [close_vertices(p) for p in work]


def _edge_set(pts: np.ndarray):
    n = len(pts)
    edges = {}
    for i in range(n):
        a = tuple(np.round(pts[i], 9))
        b = tuple(np.round(pts[(i + 1) % n], 9))
        edges[(a, b)] = i
    return edges


def _try_merge_convex(a: np.ndarray, b: np.ndarray):
    ea, eb = _edge_set(a), _edge_set(b)
    shared = None
    for (p, q) in ea:
        if (q, p) in eb:
            shared = (p, q)
            break
    if shared is None:
        return None
    # Walk the two chains excluding the shared edge and concatenate.
    def walk(pts, start, end):
        idx = {tuple(np.round(p, 9)): i for i, p in enumerate(pts)}
        i = idx[start]
        out = [pts[i]]
        n = len(pts)
        for _ in range(n):
            i = (i + 1) % n
            out.append(pts[i])
            if tuple(np.round(pts[i], 9)) == end:
                break
        return out

    p, q = shared  # edge a: p->q, edge b: q->p
    chain = walk(a, q, p) + walk(b, p, q)[1:-1]
    union = _drop_near_duplicates(np.asarray(chain, dtype=float))
    if len(union) >= 3 and is_convex_polygon(union):
        return ensure_ccw(union)
    return None


def pose_clearance(pose, obstacles, ego, bounds=None) -> tuple[float, bool]:
    """Exact rectangle-to-polygon clearance at one pose.

    Returns (min_distance, overlapped). Overlap forces distance to 0.
    `obstacles` must already be convex (decompose first).
    """
    car = vehicle_vertices(np.asarray(pose[:3], dtype=float), ego)
    if bounds is not None:
        if np.any(car[:, 0] < bounds[0] - 1e-12) or np.any(car[:, 0] > bounds[1] + 1e-12):
            return 0.0, True
        if np.any(car[:, 1] < bounds[2] - 1e-12) or np.any(car[:, 1] > bounds[3] + 1e-12):
            return 0.0, True
    clearance = np.inf
    overlapped = False
    for obstacle in obstacles:
        if polygons_overlap(car, obstacle, tol=0.0):
            overlapped = True
            clearance = 0.0
            break
        clearance = min(clearance, polygon_distance(car, obstacle))
    return float(clearance), overlapped


def poses_meet_clearance(poses, obstacles, ego, min_clearance: float, bounds=None) -> bool:
    for pose in np.asarray(poses, dtype=float).reshape(-1, 3):
        clearance, overlapped = pose_clearance(pose, obstacles, ego, bounds=bounds)
        if overlapped or clearance < min_clearance:
            return False
    return True


def densify_poses(poses: np.ndarray, ego, step: float = 0.05) -> np.ndarray:
    """Linearly interpolate adjacent poses so max corner travel is at most `step`.

    This is a sampled sweep, not a continuous swept-volume certificate.
    """
    poses = np.asarray(poses, dtype=float)
    if len(poses) == 0:
        return poses.reshape(0, 3)

    def wrap(delta):
        return math.atan2(math.sin(delta), math.cos(delta))

    samples = [poses[0, :3].copy()]
    for i in range(len(poses) - 1):
        p0 = poses[i, :3]
        p1 = poses[i + 1, :3]
        dpsi = wrap(float(p1[2] - p0[2]))
        v0 = vehicle_vertices(p0, ego)
        v1 = vehicle_vertices(np.array([p1[0], p1[1], p0[2] + dpsi]), ego)
        travel = max(float(np.linalg.norm(p1[:2] - p0[:2])), float(np.linalg.norm(v1 - v0, axis=1).max()))
        nseg = max(1, int(np.ceil(travel / step - 1e-12)))
        for k in range(1, nseg + 1):
            t = k / nseg
            samples.append(np.array([
                (1 - t) * p0[0] + t * p1[0],
                (1 - t) * p0[1] + t * p1[1],
                p0[2] + t * dpsi,
            ]))
    return np.asarray(samples)


def segment_meets_clearance(poses, obstacles, ego, min_clearance: float, step: float, bounds=None) -> bool:
    dense = densify_poses(np.asarray(poses, dtype=float), ego, step=step)
    return poses_meet_clearance(dense, obstacles, ego, min_clearance, bounds=bounds)

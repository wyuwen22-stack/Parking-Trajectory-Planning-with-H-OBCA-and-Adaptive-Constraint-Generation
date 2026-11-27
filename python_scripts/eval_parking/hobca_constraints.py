"""H-OBCA delivery constraint policy (eval-aligned interpolation + buffer).

Evaluation remains exact Euclidean clearance >= 0.05 m with machine-eps only.
These knobs only tighten the optimizer constraints; they do not loosen acceptance.
"""

from __future__ import annotations

import math

import numpy as np

from traj_py.analysis import vehicle_vertices

# Constraint buffer above the 0.05 m acceptance floor. Evaluation does not use this.
DMIN_BUFFER = 0.001
MIDPOINT_KIND = "linear"
SWEEP_STEP = 0.05
MID_CAP = 12


def adaptive_linear_midpoints(poses: np.ndarray, ego, step: float = SWEEP_STEP, cap: int = MID_CAP) -> int:
    """Number of interior linear samples so max corner travel per piece is ~step.

    Matches geometry.densify_poses. Capped to keep the NLP size finite.
    This remains a sampled constraint, not a continuous swept-volume certificate.
    """
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2:
        return 1
    needed = 0
    for i in range(len(poses) - 1):
        p0 = poses[i, :3]
        p1 = poses[i + 1, :3]
        dpsi = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
        v0 = vehicle_vertices(p0, ego)
        v1 = vehicle_vertices(np.array([p1[0], p1[1], p0[2] + dpsi]), ego)
        travel = max(
            float(np.linalg.norm(p1[:2] - p0[:2])),
            float(np.linalg.norm(v1 - v0, axis=1).max()),
        )
        nseg = max(1, int(np.ceil(travel / step - 1e-12)))
        needed = max(needed, nseg - 1)
    return int(min(cap, max(1, needed)))


def apply_delivery_constraints(problem, ego) -> int:
    """Mutate an ObcaInput in place. Returns the midpoint count used."""
    n_mid = adaptive_linear_midpoints(np.asarray(problem.x_initial[:3].T), ego)
    problem.collision_midpoints = n_mid
    problem.collision_midpoint_kind = MIDPOINT_KIND
    problem.dmin_buffer = float(DMIN_BUFFER)
    return n_mid

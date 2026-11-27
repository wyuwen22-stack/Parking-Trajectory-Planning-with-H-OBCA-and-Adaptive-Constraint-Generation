"""Unit tests for the parking success-rate evaluation package."""

import numpy as np
import pytest

from eval_parking.metrics import (
    parking_cost,
    polygon_distance,
    polygons_overlap,
    pose_clearance,
    densify_poses,
)
from eval_parking.scenarios import (
    BODY_LENGTH,
    BODY_WIDTH,
    PARA_BAY_Y0,
    PARA_BAY_Y1,
    PARA_GOAL_X,
    PARA_GOAL_Y,
    PARA_LENGTHS,
    PERP_DEPTH,
    PERP_WIDTHS,
    VEHICLE,
    build_scene,
    parallel_goal,
    perpendicular_goal,
    start_grid,
)
from traj_py.analysis import vehicle_vertices
from traj_py.obca import ObcaInput, solve_obca
from traj_py.scenario import parking_scenario


def test_parking_scenario_untouched():
    scene = parking_scenario()
    assert scene.veh.L == 2.7
    np.testing.assert_allclose(scene.veh.ego, [3.7, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(scene.start, [-9.0, 7.0, np.pi])
    np.testing.assert_allclose(scene.goal, [0.0, 1.3, np.pi / 2])
    left, right, wall = scene.obs
    np.testing.assert_allclose(left[:, 0].max(), -1.3)
    np.testing.assert_allclose(right[:, 0].min(), 1.3)
    np.testing.assert_allclose(left[:, 1].max(), 5.0)
    np.testing.assert_allclose(wall[0, 1], 11.0)


def test_perpendicular_slot_geometry():
    expected = {"tight": 2.60, "medium": 2.80, "wide": 3.20}
    for tightness, width in expected.items():
        assert PERP_WIDTHS[tightness] == pytest.approx(width)
        scene = build_scene("perpendicular", tightness, np.array([0.0, 7.5, 0.0]))
        left, right, wall = scene.obs
        half = width / 2.0
        np.testing.assert_allclose(left[:, 0].max(), -half)
        np.testing.assert_allclose(right[:, 0].min(), half)
        np.testing.assert_allclose([left[:, 1].min(), left[:, 1].max()], [0.0, PERP_DEPTH])
        np.testing.assert_allclose(wall[:, 1].min(), 11.0)
        np.testing.assert_allclose(scene.goal, perpendicular_goal())
        np.testing.assert_allclose(scene.goal, [0.0, 1.3, np.pi / 2])
        # Front bumper of the parked vehicle sits at the opening y = 5.0.
        corners = vehicle_vertices(scene.goal, scene.veh.ego)
        np.testing.assert_allclose(corners[:, 1].max(), 5.0, atol=1e-12)


def test_parallel_slot_geometry_and_body_centering():
    for tightness, length in PARA_LENGTHS.items():
        scene = build_scene("parallel", tightness, np.array([-4.0, 7.5, 0.0]))
        curb, left, right, wall = scene.obs
        half = length / 2.0
        np.testing.assert_allclose(curb[:, 1].max(), PARA_BAY_Y0)
        np.testing.assert_allclose(left[:, 0].max(), -half)
        np.testing.assert_allclose(right[:, 0].min(), half)
        np.testing.assert_allclose([left[:, 1].min(), left[:, 1].max()], [PARA_BAY_Y0, PARA_BAY_Y1])
        np.testing.assert_allclose(scene.goal, parallel_goal())
        np.testing.assert_allclose(scene.goal[0], PARA_GOAL_X)
        np.testing.assert_allclose(scene.goal[1], PARA_GOAL_Y)
        np.testing.assert_allclose(scene.goal[2], 0.0)
        corners = vehicle_vertices(scene.goal, scene.veh.ego)
        # 4.7 m body is centered in the slot along x, 2.0 m body centered in the 2.5 m bay along y.
        np.testing.assert_allclose(0.5 * (corners[:, 0].min() + corners[:, 0].max()), 0.0, atol=1e-12)
        np.testing.assert_allclose(0.5 * (corners[:, 1].min() + corners[:, 1].max()), PARA_GOAL_Y, atol=1e-12)
        assert corners[:, 0].min() - left[:, 0].max() == pytest.approx((length - BODY_LENGTH) / 2.0)
        assert right[:, 0].min() - corners[:, 0].max() == pytest.approx((length - BODY_LENGTH) / 2.0)
        assert corners[:, 1].min() - PARA_BAY_Y0 == pytest.approx((PARA_BAY_Y1 - PARA_BAY_Y0 - BODY_WIDTH) / 2.0)
        assert PARA_BAY_Y1 - corners[:, 1].max() == pytest.approx((PARA_BAY_Y1 - PARA_BAY_Y0 - BODY_WIDTH) / 2.0)
        clearance, overlapped = pose_clearance(scene.goal, scene)
        assert not overlapped
        assert clearance >= 0.05 - 1e-12


def test_start_grid_is_in_the_aisle():
    starts = start_grid()
    assert len(starts) == 10
    xs = sorted({round(p[0], 6) for p in starts})
    ys = sorted({round(p[1], 6) for p in starts})
    assert xs == [-8.0, -4.0, 0.0, 4.0, 8.0]
    assert ys == [7.5, 8.5]
    for maneuver in ("perpendicular", "parallel"):
        for tightness in ("tight", "medium", "wide"):
            for start in starts:
                scene = build_scene(maneuver, tightness, start)
                clearance, overlapped = pose_clearance(start, scene)
                assert not overlapped, (maneuver, tightness, start)
                assert clearance >= 0.5, (maneuver, tightness, start, clearance)
                assert np.allclose(start[2], 0.0)


def test_sat_detects_overlap_that_corner_distance_can_miss():
    # Two unit squares that cross in a plus-sign: no vertex of either lies inside
    # the AABB of the other in a naive way? Actually they do. Use a shallow
    # edge-to-edge overlap of two diamonds / rectangles where corners stay outside.
    a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    # Horizontal bar overlapping the middle of a vertically offset rectangle
    # such that all corners of `b` are outside `a`, but edges cross.
    b = np.array([[1.5, -0.5], [3.5, -0.5], [3.5, 1.5], [1.5, 1.5]])
    assert polygons_overlap(a, b)
    far = np.array([[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]])
    assert not polygons_overlap(a, far)
    assert polygon_distance(a, far) == pytest.approx(2.0)


def test_densify_respects_step():
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dense = densify_poses(poses, VEHICLE.ego, step=0.05)
    steps = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
    assert steps.max() <= 0.05 + 1e-12
    assert np.allclose(dense[0], poses[0])
    assert np.allclose(dense[-1], poses[-1])


def _tiny_parking_problem(objective: str) -> ObcaInput:
    start = np.array([0.0, 0.0, 0.0])
    # n=3, Ts=0.4, a_max=1, rest-to-rest reach is only ~0.32 m.
    goal = np.array([0.30, 0.0, 0.0])
    ref = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.20, 0.0, 0.0], [0.30, 0.0, 0.0]])
    X = np.zeros((4, 4))
    X[:3] = ref.T
    X[3] = [0.0, 0.4, 0.4, 0.0]
    X_ws = X.copy()
    X_ws[1] = 0.25  # offset so the position-tracking term is nonzero at a typical solution
    U = np.zeros((2, 3))
    U[1] = [1.0, 0.0, -1.0]
    return ObcaInput(
        ref=ref,
        x_initial=X_ws,
        u_initial=U,
        lambda_initial=(),
        mu_initial=(),
        Aob=(),
        bob=(),
        G=np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
        g=np.array([3.7, 1.0, 1.0, 1.0]),
        L=2.7,
        start=start,
        goal=goal,
        Ts=0.4,
        v_max=3.0,
        a_max=1.0,
        delta_max=0.5,
        ddelta_max=0.6,
        dmin=0.05,
        solver_options={"ipopt": {"print_level": 0, "max_iter": 2000, "tol": 1e-8}, "print_time": False},
        parking=True,
        collision_constraints=False,
        bounds=np.array([-10.0, 10.0, -10.0, 10.0]),
        parking_objective=objective,
    )


def test_once_objective_is_not_multiplied_by_n():
    once = solve_obca(_tiny_parking_problem("once"))
    assert once.status == "Solve_Succeeded"
    expected_once = parking_cost(once.X, once.U, _tiny_parking_problem("once").x_initial, mode="once")
    expected_repeated = parking_cost(once.X, once.U, _tiny_parking_problem("once").x_initial, mode="repeated")
    n = once.U.shape[1]
    assert n == 3
    assert abs(once.objective - expected_once) <= 1e-8 * max(1.0, abs(expected_once))
    # Position / Δv terms are inside the cost; once and repeated must differ.
    position = 0.3 * np.sum((once.X[:2] - _tiny_parking_problem("once").x_initial[:2]) ** 2)
    delta_v = np.sum(np.diff(once.X[3, :-1]) ** 2)
    assert position + delta_v > 1e-8
    assert expected_repeated - expected_once == pytest.approx((n - 1) * (position + delta_v))
    assert abs(once.objective - expected_repeated) > 1e-6

    repeated = solve_obca(_tiny_parking_problem("repeated"))
    assert repeated.status == "Solve_Succeeded"
    expected = parking_cost(repeated.X, repeated.U, _tiny_parking_problem("repeated").x_initial, mode="repeated")
    assert abs(repeated.objective - expected) <= 1e-8 * max(1.0, abs(expected))

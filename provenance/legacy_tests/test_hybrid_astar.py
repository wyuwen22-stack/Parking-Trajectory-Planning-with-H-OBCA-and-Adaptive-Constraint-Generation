from pathlib import Path

import numpy as np
from scipy.io import loadmat

from traj_py.hybrid_astar import (
    PlannerSettings,
    collision_free,
    inflate_occupancy,
    plan_hybrid_astar,
    rectangle_collision_free,
)
from traj_py.scenario import simple_scenario, parking_scenario
from eval_parking.scenarios import (
    PARKING_PLANNER_SETTINGS,
    RECTANGLE_PLANNER_SETTINGS,
    build_scene,
    start_grid,
)


def test_hybrid_astar_inflation_and_paths():
    reference = loadmat(Path(__file__).parent/'reference'/'hybrid_astar_reference.mat', simplify_cells=True)
    for scene, settings in [(simple_scenario(), PlannerSettings()),
                            (parking_scenario(), PlannerSettings(primitive_length=1., reverse_cost=1.5, interpolation=.3, validation_distance=.05, inflation=.2))]:
        result = plan_hybrid_astar(scene, settings)
        np.testing.assert_allclose(result.states[0], scene.start, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.states[-1], scene.goal, rtol=0, atol=1e-12)
        assert np.isfinite(result.states).all()
        assert collision_free(result.states, scene, result.inflated_occupancy)
        assert np.max(np.linalg.norm(np.diff(result.states[:, :2], axis=0), axis=1)) <= settings.interpolation+1e-12
        if settings.inflation == 1.:
            mismatch = np.count_nonzero(result.inflated_occupancy != reference['inflated_occupancy'])
            print(f'Inflation differing cells={mismatch}; Python path={result.states.shape}, MATLAB={reference["states"].shape} (independent search)')
            assert mismatch == 0



def test_default_collision_model_remains_inflated_point():
    assert PlannerSettings().collision_model == "inflated_point"
    assert PARKING_PLANNER_SETTINGS.collision_model == "inflated_point"
    assert RECTANGLE_PLANNER_SETTINGS.collision_model == "rectangle"
    assert RECTANGLE_PLANNER_SETTINGS.collision_margin == 0.0
    # 矩形模式对齐官方 H-OBCA 搜索参数；基线 PARKING_PLANNER_SETTINGS 保持不变。
    assert PARKING_PLANNER_SETTINGS.primitives == 5
    assert PARKING_PLANNER_SETTINGS.primitive_length == 1.0
    assert PARKING_PLANNER_SETTINGS.reverse_cost == 1.5
    assert PARKING_PLANNER_SETTINGS.heading_heuristic_gain == 0.0
    assert RECTANGLE_PLANNER_SETTINGS.primitives == 11
    assert RECTANGLE_PLANNER_SETTINGS.primitive_length == 0.3
    assert RECTANGLE_PLANNER_SETTINGS.reverse_cost == 1.0
    assert RECTANGLE_PLANNER_SETTINGS.heading_heuristic_gain == 2.0
    assert RECTANGLE_PLANNER_SETTINGS.rs_heading_limit == 0.6
    assert RECTANGLE_PLANNER_SETTINGS.inflation == PARKING_PLANNER_SETTINGS.inflation
    assert RECTANGLE_PLANNER_SETTINGS.validation_distance == PARKING_PLANNER_SETTINGS.validation_distance


def test_rectangle_rejects_body_overlap_that_inflated_point_allows():
    # 参考点在通道里，0.2 m 膨胀仍通过；4.7×2 车身已经切入紧垂直车位右侧障碍。
    scene = build_scene("perpendicular", "tight", np.array([0.0, 7.5, 0.0]))
    pose = np.array([[0.0, 5.5, 0.0]])
    occ = inflate_occupancy(scene, 0.2)
    assert collision_free(pose, scene, occ)
    assert not rectangle_collision_free(pose, scene, margin=0.0)


def test_parking_starts_and_goals_are_rectangle_free():
    for maneuver in ("perpendicular", "parallel"):
        for tightness in ("tight", "medium", "wide"):
            scene = build_scene(maneuver, tightness, np.array([0.0, 7.5, 0.0]))
            assert rectangle_collision_free(scene.goal[None, :], scene, margin=0.0)
            for start in start_grid():
                assert rectangle_collision_free(np.asarray(start)[None, :], scene, margin=0.0)


def test_inflated_point_parking_plan_still_runs():
    scene = parking_scenario()
    result = plan_hybrid_astar(scene, PARKING_PLANNER_SETTINGS)
    np.testing.assert_allclose(result.states[0], scene.start, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.states[-1], scene.goal, rtol=0, atol=1e-12)
    assert collision_free(result.states, scene, result.inflated_occupancy)


def test_aabb_agrees_with_generic_sat():
    # 加速后的轴对齐 SAT 必须与通用多边形 SAT 在随机位姿上一致。
    from traj_py.analysis import vehicle_vertices as verts
    from traj_py.hybrid_astar import _in_bounds, _open_polygon, polygons_overlap

    scene = build_scene("parallel", "tight", np.array([-4.0, 7.5, 0.0]))
    rng = np.random.default_rng(0)
    poses = np.column_stack((
        rng.uniform(-12, 12, 400),
        rng.uniform(0.5, 11.5, 400),
        rng.uniform(-np.pi, np.pi, 400),
    ))
    for pose in poses:
        fast = rectangle_collision_free(pose[None, :], scene, margin=0.0)
        corners = verts(pose, scene.veh.ego)
        slow = _in_bounds(corners, scene.XYbounds)
        if slow:
            for obs in scene.obs:
                if polygons_overlap(corners, _open_polygon(obs), tol=0.0):
                    slow = False
                    break
        assert fast is slow, pose


def test_rectangle_finds_tight_perpendicular():
    # 论文紧垂直回归起点：矩形碰撞 + 官方对齐参数应能找到路径。
    scene = build_scene("perpendicular", "tight", np.array([-9.0, 7.0, np.pi]))
    result = plan_hybrid_astar(scene, RECTANGLE_PLANNER_SETTINGS)
    np.testing.assert_allclose(result.states[0], scene.start, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.states[-1], scene.goal, rtol=0, atol=1e-12)
    assert rectangle_collision_free(result.states, scene, margin=0.0)


def test_tight_parallel_goal_is_kinematic_trap():
    """6.0 m tight parallel + Rmin=5: the *discrete* backward flood may
    exhaust inside the bay. That is a heuristic pre-check, not a proof,
    and must not skip formal search. Medium (6.5 m) must still escape.
    """
    from dataclasses import replace
    from traj_py.hybrid_astar import goal_component_escapes, plan_hybrid_astar
    tight = build_scene("parallel", "tight", np.array([-4.0, 7.5, 0.0]))
    ok, reason = goal_component_escapes(tight, RECTANGLE_PLANNER_SETTINGS)
    assert ok is False
    assert "heuristic pre-check exhausted" in reason
    medium = build_scene("parallel", "medium", np.array([-4.0, 7.5, 0.0]))
    ok_m, reason_m = goal_component_escapes(medium, RECTANGLE_PLANNER_SETTINGS)
    assert ok_m is True, reason_m
    # Formal search must start (time cap is evidence it was not hard-skipped).
    short = replace(RECTANGLE_PLANNER_SETTINGS, max_time=0.3)
    try:
        plan_hybrid_astar(tight, short)
    except ValueError as exc:
        text = str(exc)
        assert "kinematic trap" not in text
        assert "heuristic pre-check" not in text
        assert "Start or goal" in text or "in collision" in text
    except RuntimeError as exc:
        text = str(exc)
        assert ("time cap" in text) or ("no path" in text) or ("SearchBudget" in text)


def test_reeds_shepp_near_goal_does_not_crash():
    """A node a few centimetres from the goal used to raise ZeroDivisionError
    or have its short connection dropped as 'too short'."""
    from traj_py.reeds_shepp import calc_paths
    paths = calc_paths(0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.2, 0.05)
    assert paths, "short goal-connecting RS path must be kept"
    assert min(p.L for p in paths) < 0.1
    # exact coincidence must return a trivial path, not throw
    paths0 = calc_paths(1.0, 2.0, 0.3, 1.0, 2.0, 0.3, 0.2, 0.05)
    assert paths0

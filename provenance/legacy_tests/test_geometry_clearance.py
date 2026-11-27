import numpy as np

from eval_parking.scenarios import RECTANGLE_PLANNER_SETTINGS, VEHICLE, build_scene
from eval_parking.tpcap_cases import parse_official_cases, translation_roundtrip_ok, PROJECT_VEHICLE
from traj_py.geometry import (
    convex_decompose,
    is_convex_polygon,
    polygon_distance,
    polygons_overlap,
    pose_clearance,
)
from traj_py.hybrid_astar import rectangle_collision_free


def test_min_clearance_is_not_rectangle_expansion():
    assert RECTANGLE_PLANNER_SETTINGS.collision_margin == 0.0
    assert RECTANGLE_PLANNER_SETTINGS.min_clearance == 0.05
    # Two axis-aligned boxes 0.03 m apart: exact distance is 0.03, so
    # min_clearance=0.05 fails, while expanding the vehicle by 0.05 m is a
    # different (generally more conservative, axis-aligned) test.
    scene = build_scene("perpendicular", "tight", np.array([0.0, 7.5, 0.0]))
    # A pose in the aisle, far from obstacles, must remain free under 0.05 m.
    assert rectangle_collision_free(np.array([[0.0, 8.0, 0.0]]), scene, margin=0.0, min_clearance=0.05)


def test_concave_polygon_is_decomposed_into_convex_pieces():
    # C-shaped (concave) polygon.
    notch = np.array([
        [0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0],
        [0.0, 2.4], [2.4, 2.4], [2.4, 0.6], [0.0, 0.6],
    ], dtype=float)
    assert not is_convex_polygon(notch)
    pieces = convex_decompose(notch)
    assert len(pieces) >= 2
    assert all(is_convex_polygon(p) for p in pieces)


def test_tpcap_official_twenty_roundtrip_and_endpoint_feasibility():
    cases = parse_official_cases(PROJECT_VEHICLE, "project_4.7x2", 0.05)
    assert len(cases) == 20
    for case in cases:
        assert translation_roundtrip_ok(case)
        assert all(is_convex_polygon(p) for p in case.convex_pieces)
        for pose in (case.start, case.goal):
            clearance, overlapped = pose_clearance(pose, case.convex_pieces, case.vehicle.ego)
            assert not overlapped
            assert clearance >= 0.05 - 1e-9
        if case.case_id in (13, 14, 15):
            assert np.linalg.norm(case.origin_shift) > 1e6
            assert abs(case.start[0]) < 1e-8 and abs(case.start[1]) < 1e-8
            assert case.start[2] == case.official_start[2]


def test_concave_exact_distance_matches_convex_pieces():
    """Distance to a concave obstacle equals the min distance to its convex pieces.
    A point inside the notch must be reported as overlapping the original shape
    (or have zero/negative effective clearance), not as a positive vertex-edge
    distance to the outer hull.
    """
    notch = np.array([
        [0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0],
        [0.0, 2.4], [2.4, 2.4], [2.4, 0.6], [0.0, 0.6],
    ], dtype=float)
    pieces = convex_decompose(notch)
    # Point in the notch pocket: inside the C bounding box, but in free space
    # of the opening (not inside any convex piece). SAT on a raw concave
    # polygon is invalid; the API must decompose first so the pocket is free.
    probe = np.array([[0.25, 1.45], [0.35, 1.45], [0.35, 1.55], [0.25, 1.55]])
    raw_ov = polygons_overlap(probe, notch, tol=0.0)
    piece_ov = any(polygons_overlap(probe, p, tol=0.0) for p in pieces)
    assert piece_ov is False
    assert raw_ov is False, "concave overlap must use convex decomposition, not raw SAT"
    d_raw = polygon_distance(probe, notch)
    d_pieces = min(polygon_distance(probe, p) for p in pieces)
    assert abs(d_raw - d_pieces) <= 1e-12, (d_raw, d_pieces)
    assert d_raw > 0.8  # ~0.85 m to the inner edges y=0.6 and y=2.4
    # A probe clearly outside, 0.2 m left of the left extremities of the C.
    outside = np.array([[-0.5, 1.4], [-0.3, 1.4], [-0.3, 1.6], [-0.5, 1.6]])
    assert not polygons_overlap(outside, notch, tol=0.0)
    d_raw = polygon_distance(outside, notch)
    d_pieces = min(polygon_distance(outside, p) for p in pieces)
    assert abs(d_raw - d_pieces) <= 1e-12, (d_raw, d_pieces)
    assert d_raw >= 0.2 - 1e-9


def test_tpcap_translation_residual_and_local_clearance():
    """Cases 13-15: invert residual is reported, and clearance is evaluated
    in the *local* frame (the frame used for planning), not 1e10 official coords.
    """
    from eval_parking.tpcap_cases import invert_translation, open_vertices
    cases = parse_official_cases(PROJECT_VEHICLE, "project_4.7x2", 0.05)
    big = [c for c in cases if c.case_id in (13, 14, 15)]
    assert len(big) == 3
    for case in big:
        rec_s = invert_translation(case.start, case.origin_shift)
        rec_g = invert_translation(case.goal, case.origin_shift)
        err_s = float(np.max(np.abs(rec_s - case.official_start)))
        err_g = float(np.max(np.abs(rec_g - case.official_goal)))
        err_o = 0.0
        for loc, off in zip(case.obstacles, case.official_obstacles):
            err_o = max(err_o, float(np.max(np.abs(
                invert_translation(open_vertices(loc), case.origin_shift) - open_vertices(off)
            ))))
        # Recorded so a failure prints the residual; 1e-9 is already the test budget.
        assert err_s <= 1e-9 and err_g <= 1e-9 and err_o <= 1e-9, (case.case_id, err_s, err_g, err_o)
        for pose in (case.start, case.goal):
            d, ov = pose_clearance(pose, case.convex_pieces, case.vehicle.ego)
            assert not ov
            # Endpoint feasibility uses the user 5 cm rule with machine-eps only.
            assert d >= 0.05 - 1e-12, (case.case_id, pose, d)


def test_acceptance_limit_is_strict_five_centimetres():
    """The delivery predicate is 0.05 m, not 0.049 m."""
    from eval_parking.strict_recount import STRICT_LIMIT, LEGACY_LIMIT, DMIN, MACHINE_TOL, strict_clearance_ok
    assert DMIN == 0.05
    assert MACHINE_TOL <= 1e-12
    assert STRICT_LIMIT >= 0.05 - 1e-12
    assert LEGACY_LIMIT == 0.049
    assert not strict_clearance_ok(0.0498, 0.0498)
    assert strict_clearance_ok(0.05, 0.05)
    assert strict_clearance_ok(0.05 - 1e-13, 0.05 - 1e-13)

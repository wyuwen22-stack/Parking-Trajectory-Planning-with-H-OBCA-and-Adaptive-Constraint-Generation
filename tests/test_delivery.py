import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from safe_parking.stage import ROOT, config, load
from safe_parking.suite import specs
from safe_parking.view_data import load_case


def test_all128_frontend_backend_radius_and_five_cell_inflation():
    from eval_parking.scenarios import scene_for_spec, RECTANGLE_PLANNER_SETTINGS
    from traj_py.warm_start import make_warm_start
    settings = RECTANGLE_PLANNER_SETTINGS
    expected = math.atan(2.7 / 3.95)
    for spec in specs():
        scene = scene_for_spec(spec)
        assert scene.veh.Rmin == settings.radius == config()['vehicle']['Rmin'] == 3.95
        assert math.ceil(settings.inflation * scene.resolution) == 5
        path = np.linspace(scene.start, scene.goal, 10)
        p = make_warm_start(path, scene, parking=True, horizon=8).problem
        assert p.delta_max == pytest.approx(expected, abs=1e-12)


def test_legacy_archive_is_not_silently_relabelled():
    _, scene = load('h_obca_parallel_wide_x-4_y7.5.npz')
    assert scene.veh.Rmin == 5.
    from safe_parking.stage import worker
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(ValueError, match='differs from contract'):
            worker('h_obca_parallel_wide_x-4_y7.5.npz', folder, 'node')


def test_validator_enforces_new_steering_bound():
    from safe_parking.geometry import Geometry
    from safe_parking.validator import validate
    from traj_py.scenario import Scenario, Vehicle
    scene = Scenario(Vehicle(2.7, np.array([3.7, 1, 1, 1]), 3.95),
                     np.array([-10, 10, -10, 10]), np.zeros(3), np.zeros(3),
                     (), (), (), np.zeros((1, 1), bool), 5.)
    X = np.zeros((4, 3))
    bound = math.atan(2.7 / 3.95)
    for angle, accepted in [(bound, True), (bound + 2e-5, False)]:
        U = np.array([[angle, angle], [0., 0.]])
        result = validate(X, U, .4, scene, Geometry((), scene.veh.ego), config())
        assert result['dynamics_pass'] == accepted


@pytest.fixture
def saved_case(tmp_path):
    folder = tmp_path / 'example'
    folder.mkdir()
    scene = dict(start=np.array([-1., 0., 0.]), goal=np.array([1., 0., 0.]),
                 ego=np.array([.2, .2, .2, .2]), bounds=np.array([-3., 3., -3., 3.]),
                 L=2.7, Rmin=3.95, obs_0=np.array([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]]))
    np.savez(folder / 'scene.npz', **scene)
    target = folder / 'adaptive'
    target.mkdir()
    X = np.array([[-1., 1.], [0., 0.], [0., 0.], [0., 4.]])
    for turn in (1, 2):
        samples = [[0, .5]] if turn == 1 else [[0, .25], [0, .5]]
        added = [1] if turn == 1 else [2, 1]
        np.savez(target / f'round{turn:02}.npz', **scene, X=X, U=np.array([[0.], [4.]]),
                 Ts=1., constraint_samples=samples, constraint_added_round=added)
        (target / f'round{turn:02}.json').write_text(json.dumps({
            'status': 'Solve_Succeeded', 'validation': {'accepted': False}}))
    (target / 'result.json').write_text(json.dumps({'status': 'COLLISION'}))
    return tmp_path


def test_viewer_interior_collision_and_insertion_rounds(saved_case):
    case = load_case(saved_case, 'example')
    trace = case.traces['adaptive']
    assert trace.clearance.min() < 0  # nodes are clear but the intervening arc hits the obstacle
    assert trace.clearance[0] > 0 and trace.clearance[-1] > 0
    assert len(trace.additions) == 1
    added = trace.additions[0]
    assert (added['segment'], added['fraction'], added['round'], added['time']) == (0, .25, 2, .25)
    assert not trace.accepted
    assert case.statuses['node'] == 'NOT_RUN'
    assert load_case(saved_case, 'example', adaptive_round=1).traces['adaptive'].additions == []


def test_viewer_rejects_mixed_vehicle_metadata(saved_case):
    path = saved_case / 'example' / 'adaptive' / 'round02.npz'
    with np.load(path) as z:
        fields = dict(z)
    fields['Rmin'] = 5.
    np.savez(path, **fields)
    with pytest.raises(ValueError, match='Rmin'):
        load_case(saved_case, 'example')


def test_export_from_other_directory_without_planning_or_casadi(saved_case, tmp_path):
    # Block imports as well as execution. Missing node/dense results must remain absent.
    code = """
import sys, runpy
for module in ['casadi', 'safe_parking.stage', 'safe_parking.suite',
               'safe_parking.solver', 'traj_py.hybrid_astar', 'traj_py.warm_start']:
    sys.modules[module] = None
sys.argv = [sys.argv[1], '--results', sys.argv[2], '--case', 'example', '--export', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    before = {p.relative_to(saved_case): p.read_bytes() for p in saved_case.rglob('*') if p.is_file()}
    image = tmp_path.parent / (tmp_path.name + '_figure.png')
    proc = subprocess.run([sys.executable, '-B', '-c', code, str(ROOT / 'view_results.py'),
                           str(saved_case), str(image)], cwd=tmp_path,
                          env=dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1'),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert image.stat().st_size > 10000
    after = {p.relative_to(saved_case): p.read_bytes() for p in saved_case.rglob('*') if p.is_file()}
    assert before == after

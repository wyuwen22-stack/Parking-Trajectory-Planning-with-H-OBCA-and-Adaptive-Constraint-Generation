"""Each public entry must run from an empty directory without saved stage data."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import matplotlib.image as mpimg
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'python_scripts'
SCRIPTS = sorted(path.stem for path in (ROOT / 'matlab_ref').glob('*.m'))


@pytest.mark.parametrize('stem', ['scenario_simple', 'two_dimension_map', 'build_map_parking',
                                 'run_hybrid_astar', 'run_hybrid_astar_parking'])
def test_scene_and_path_without_casadi(stem, tmp_path):
    shutil.copytree(SOURCE / 'traj_py', tmp_path / 'traj_py', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(SOURCE / f'{stem}.py', tmp_path / f'{stem}.py')
    # A fresh process prevents already imported solver modules from hiding the bug.
    code = ("import sys, runpy; sys.modules['casadi'] = None; "
            "runpy.run_path(sys.argv[1], run_name='__main__')")
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code, f'{stem}.py'],
                            cwd=tmp_path, env=dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mplconfig')),
                            capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'results' / f'{stem}.png').is_file()


@pytest.mark.parametrize('stem', SCRIPTS)
def test_standalone_script(stem, tmp_path):
    shutil.copytree(SOURCE / 'traj_py', tmp_path / 'traj_py', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(SOURCE / f'{stem}.py', tmp_path / f'{stem}.py')
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path / 'mplconfig'))
    # IPOPT's Windows output requires a real file handle rather than a pipe.
    log = tmp_path / 'run.log'
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run([sys.executable, '-X', 'utf8', str(tmp_path / f'{stem}.py')],
                                cwd=tmp_path, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=240)
    assert result.returncode == 0, log.read_text(encoding='utf-8')
    figures = list((tmp_path / 'results').glob(f'{stem}*.png'))
    assert len(figures) == (3 if stem.startswith('analyze_') else 2 if 'mpc' in stem or stem == 'run_parking_animation' else 1)
    for figure in figures:
        pixels = mpimg.imread(figure)
        assert pixels.shape[0] >= 400 and pixels.shape[1] >= 400
        assert np.isfinite(pixels).all() and pixels.std() > .01
    if 'animation' in stem:
        from PIL import Image  # Matplotlib's existing PillowWriter dependency.
        with Image.open(tmp_path / 'results' / f'{stem}.gif') as gif:
            assert gif.n_frames == (51 if 'parking' in stem else 41)
            assert gif.info['duration'] == 120
    assert not list(tmp_path.rglob('*.npz'))

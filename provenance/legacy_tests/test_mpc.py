from pathlib import Path

import numpy as np
from scipy.io import loadmat

from traj_py.mpc import MpcInput, track_mpc


def test_mpc_matches_matlab_reference():
    folder = Path(__file__).parent / 'reference'
    r = loadmat(folder / 'mpc_tracking_zero_reference.mat', simplify_cells=True)
    s = loadmat(folder / 'scenario_simple_reference.mat', simplify_cells=True)
    fields = {name: r[name] for name in MpcInput.__dataclass_fields__ if name in r}
    result = track_mpc(MpcInput(**fields, L=s['veh']['L']))
    x_error = np.max(np.abs(result.X_actual - r['X_actual']))
    u_error = np.max(np.abs(result.U_applied - r['U_applied']))
    print(f'MPC X_max={x_error:.16g}, U_max={u_error:.16g}, statuses={set(result.statuses)}')
    assert all(status == 'Solve_Succeeded' for status in result.statuses)
    assert x_error <= 1e-4
    assert u_error <= 1e-4

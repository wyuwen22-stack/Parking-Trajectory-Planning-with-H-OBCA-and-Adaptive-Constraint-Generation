from pathlib import Path

import numpy as np
from scipy.io import loadmat

from traj_py.obca import ObcaInput, solve_obca


def test_obca_matches_matlab_reference():
    folder = Path(__file__).parent / 'reference'
    r = loadmat(folder / 'obca_reference.mat', simplify_cells=True)
    s = loadmat(folder / 'scenario_simple_reference.mat', simplify_cells=True)
    fields = {name: r[name] for name in ObcaInput.__dataclass_fields__ if name in r}
    p = ObcaInput(**fields, L=s['veh']['L'], start=s['map']['start'], goal=s['map']['goal'])
    result = solve_obca(p)
    objective_error = abs(result.objective - r['objective']) / abs(r['objective'])
    x_error = np.max(np.abs(result.X - r['x_solution']))
    u_error = np.max(np.abs(result.U - r['u_solution']))
    print(f'OBCA status={result.status}, iterations={result.iterations} (MATLAB {r["iterations"]}), objective_relative={objective_error:.16g}, X_max={x_error:.16g}, U_max={u_error:.16g}')
    assert result.status == 'Solve_Succeeded'
    assert result.iterations == r['iterations']
    assert objective_error <= 1e-6
    assert x_error <= 1e-4
    assert u_error <= 1e-4

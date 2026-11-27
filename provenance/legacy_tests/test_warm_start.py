from pathlib import Path

import numpy as np
from scipy.io import loadmat

from traj_py.scenario import simple_scenario
from traj_py.warm_start import make_warm_start


def test_warm_start_matches_matlab_reference():
    folder = Path(__file__).parent/'reference'
    path = loadmat(folder/'hybrid_astar_reference.mat', simplify_cells=True)['states']
    reference = loadmat(folder/'obca_reference.mat', simplify_cells=True)
    result = make_warm_start(path, simple_scenario())
    np.testing.assert_array_equal(result.indices, reference['indices'])
    for name in ('ref', 'x_initial', 'u_initial', 'lambda_initial', 'mu_initial', 'Aob', 'bob', 'G', 'g'):
        if name in ('lambda_initial', 'mu_initial', 'Aob', 'bob'):
            actual, expected = np.stack(getattr(result.problem, name)), np.stack(reference[name])
        else:
            actual, expected = np.asarray(getattr(result.problem, name)), np.asarray(reference[name])
        error = np.max(np.abs(actual-expected))
        print(f'Warm start {name}: max_absolute={error:.16g}')
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


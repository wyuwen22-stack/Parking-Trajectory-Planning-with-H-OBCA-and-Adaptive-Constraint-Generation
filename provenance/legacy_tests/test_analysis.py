from dataclasses import replace

import numpy as np

from traj_py.analysis import path_curvature, vehicle_vertices, approximate_clearance
from traj_py.scenario import simple_scenario


def test_analysis_known_geometry():
    # No MATLAB metrics export exists: analytic geometry checks, not equivalence claims.
    angles = np.linspace(0, np.pi/2, 20)
    circle = np.column_stack((5*np.cos(angles), 5*np.sin(angles)))
    np.testing.assert_allclose(path_curvature(circle), .2, rtol=0, atol=1e-12)
    scene = simple_scenario()
    np.testing.assert_allclose(vehicle_vertices([0, 0, np.pi/2], scene.veh.ego),
                               [[-1, 3.7], [1, 3.7], [1, -1], [-1, -1]], rtol=0, atol=1e-12)
    wall = np.array([[5., -10], [6, -10], [6, 10], [5, 10], [5, -10]])
    scene = replace(scene, obs=(wall,))
    np.testing.assert_allclose(approximate_clearance(np.array([[0., 0, 0], [2, 0, 0]]), scene),
                               [1.3, -.3], rtol=0, atol=1e-12)


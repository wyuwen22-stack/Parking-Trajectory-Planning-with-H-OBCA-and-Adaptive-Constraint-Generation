import numpy as np

from traj_py.hybrid_astar import PlannerSettings, plan_hybrid_astar
from traj_py.obca import solve_obca
from traj_py.scenario import parking_scenario
from traj_py.warm_start import make_warm_start


def test_parking_solution_satisfies_source_constraints():
    # Parking has no MATLAB export: check feasibility and the original repeated cost.
    scene = parking_scenario()
    path = plan_hybrid_astar(scene, PlannerSettings(primitive_length=1., reverse_cost=1.5,
                            interpolation=.3, validation_distance=.05, inflation=.2))
    p = make_warm_start(path.states, scene, parking=True).problem
    result = solve_obca(p)
    x, u = result.X, result.U
    dynamics = np.vstack((x[3, :-1]*np.cos(x[2, :-1]), x[3, :-1]*np.sin(x[2, :-1]),
                          x[3, :-1]/p.L*np.tan(u[0]), u[1]))
    error = np.max(np.abs(np.diff(x, axis=1)-p.Ts*dynamics))
    assert result.status == 'Solve_Succeeded'
    assert error <= 1e-4
    np.testing.assert_allclose(x[:, 0], np.r_[scene.start, 0], rtol=0, atol=1e-4)
    assert np.max(np.abs(x[:3, -1]-scene.goal)) <= .05+1e-4
    assert abs(x[3, -1]) <= .1+1e-4
    assert np.max(np.abs(x[3])) <= p.v_max+1e-4
    assert np.max(np.abs(u[0])) <= p.delta_max+1e-4
    assert np.max(np.abs(u[1])) <= p.a_max+1e-4
    assert np.max(np.abs(np.diff(u[0]))) <= p.ddelta_max*p.Ts+1e-4
    assert x[0].min() >= scene.XYbounds[0]+1-1e-4 and x[0].max() <= scene.XYbounds[1]-1+1e-4
    assert x[1].min() >= scene.XYbounds[2]-1e-4 and x[1].max() <= scene.XYbounds[3]-1+1e-4
    # Independent separating-axis check at the optimized poses (not between steps).
    from traj_py.analysis import vehicle_vertices
    for pose in x[:3].T:
        car = vehicle_vertices(pose, scene.veh.ego)
        for obstacle in scene.obs:
            axes = np.vstack((np.eye(2), [[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]]))
            gaps = []
            for axis in axes:
                a, b = car@axis, obstacle[:-1]@axis
                gaps.append(max(a.min()-b.max(), b.min()-a.max()))
            assert max(gaps) > 0, 'Vehicle overlaps an obstacle at an optimized pose'
    n = u.shape[1]
    objective = (.5*np.sum(u[0]**2)+.1*np.sum(u[1]**2)+.2*np.sum(np.diff(u[0])**2)
                 +2*np.sum(np.diff(u[1])**2)+n*(.3*np.sum((x[:2]-p.x_initial[:2])**2)
                 +np.sum(np.diff(x[3, :-1])**2)))
    assert abs(objective-result.objective)/abs(result.objective) <= 1e-12
    print(f'Parking feasibility: dynamics_max={error:.16g}, objective={result.objective}, iterations={result.iterations}; no MATLAB baseline')

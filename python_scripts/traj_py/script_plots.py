"""Plotting and in-memory prerequisites for the MATLAB-named entry scripts."""
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .analysis import approximate_clearance, path_curvature, vehicle_vertices
from .hybrid_astar import PlannerSettings, plan_hybrid_astar
from .scenario import simple_scenario, parking_scenario

OUTPUT = Path(__file__).resolve().parents[1] / 'results'


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f'{stem}.png'
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(path)


def draw_scene(ax, scene):
    for obstacle in scene.obs:
        ax.fill(obstacle[:, 0], obstacle[:, 1], color='.2')
    ax.plot(*scene.start[:2], 'go', label='Start')
    ax.plot(*scene.goal[:2], 'ro', label='Goal')
    ax.set(xlim=scene.XYbounds[:2], ylim=scene.XYbounds[2:], xlabel='X [m]', ylabel='Y [m]')
    ax.set_aspect('equal')


def draw_vehicle(ax, pose, scene, **kwargs):
    vertices = vehicle_vertices(pose, scene.veh.ego)
    vertices = np.vstack((vertices, vertices[0]))
    return ax.plot(vertices[:, 0], vertices[:, 1], **kwargs)[0]


def scene_for(parking):
    return parking_scenario() if parking else simple_scenario()


def coarse_path(scene, parking):
    settings = (PlannerSettings(primitive_length=1., reverse_cost=1.5,
                interpolation=.3, validation_distance=.05, inflation=.2)
                if parking else PlannerSettings())
    return plan_hybrid_astar(scene, settings).states


def optimized(scene, parking, collision=True):
    from .obca import solve_obca
    from .warm_start import make_warm_start

    path = coarse_path(scene, parking)
    problem = make_warm_start(path, scene, parking).problem
    if not collision:
        problem = replace(problem, collision_constraints=False,
            solver_options={'ipopt': {'print_level': 5, 'max_iter': 3000}, 'print_time': True})
    result = solve_obca(problem)
    if result.status != 'Solve_Succeeded':
        raise RuntimeError(f'OBCA: {result.status}')
    print(f'OBCA objective={result.objective:.12g}, iterations={result.iterations}')
    return path, problem, result


def plot_map(stem, parking=False):
    scene = scene_for(parking)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    draw_scene(axes[0], scene)
    draw_vehicle(axes[0], scene.start, scene, color='green')
    draw_vehicle(axes[0], scene.goal, scene, color='red')
    axes[0].set_title('Vehicle and obstacles')
    axes[1].imshow(scene.occupancy, origin='upper', extent=scene.XYbounds, cmap='Greys')
    axes[1].set(title='Occupancy grid (5 cells/m)', xlabel='X [m]', ylabel='Y [m]')
    save(fig, stem)


def plot_hybrid(stem, parking=False):
    scene = scene_for(parking)
    path = coarse_path(scene, parking)
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    draw_scene(ax, scene)
    ax.plot(path[:, 0], path[:, 1], '.-', color='royalblue', label='Python Hybrid A*')
    for pose in path[::max(1, len(path)//12)]:
        draw_vehicle(ax, pose, scene, color='seagreen', linewidth=.8)
    ax.legend(); ax.set_title('Hybrid A* coarse path')
    save(fig, stem)


def plot_obca(stem, parking=False, collision=True):
    scene = scene_for(parking)
    path, problem, result = optimized(scene, parking, collision)
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    draw_scene(ax, scene)
    ax.plot(problem.ref[:, 0], problem.ref[:, 1], '.', color='royalblue', label='Hybrid A* warm start')
    ax.plot(result.X[0], result.X[1], color='firebrick', label='OBCA')
    for pose in result.X[:3, ::4].T:
        draw_vehicle(ax, pose, scene, color='seagreen', alpha=.65, linewidth=.8)
    ax.legend()
    ax.set_title('H-OBCA' if collision else 'Model check: collision constraints disabled')
    save(fig, stem)


def plot_analysis(stem, parking=False):
    scene = scene_for(parking)
    path, problem, result = optimized(scene, parking)
    X, U = result.X, result.U
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for points, label in [(path, 'Hybrid A*'), (X[:3].T, 'OBCA')]:
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]
        ax.plot(arc, path_curvature(points), label=label)
    ax.set(xlabel='Path length [m]', ylabel='Absolute curvature [1/m]'); ax.legend()
    save(fig, stem + '_curvature')
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.plot(approximate_clearance(X[:3].T, scene), 'o-', markersize=3)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set(xlabel='Time step k', ylabel='Approximate clearance [m]',
           title='MATLAB corner-to-edge metric (sampled poses)')
    save(fig, stem + '_clearance')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
    for ax, values, label, limit, first in zip(axes.flat,
            [X[3], U[0], U[1], np.diff(U[0])],
            ['Speed [m/s]', 'Steering [rad]', 'Acceleration [m/s²]', 'Steering increment [rad/step]'],
            [problem.v_max, problem.delta_max, problem.a_max, None], [0, 1, 1, 2]):
        ax.plot(np.arange(len(values))+first, values)
        if limit is not None:
            ax.axhline(limit, color='gray', linestyle='--'); ax.axhline(-limit, color='gray', linestyle='--')
        ax.set(xlabel='Time step k', ylabel=label)
    save(fig, stem + '_controls')


def plot_mpc(stem, parking=False, perturbation=(0., 0., 0., 0.), animation=False):
    from .mpc import MpcInput, track_mpc

    scene = scene_for(parking)
    path, problem, reference = optimized(scene, parking)
    p = MpcInput(reference.X, reference.U, reference.X[:, 0]+np.asarray(perturbation),
        np.diag([15, 15, 8, 1] if parking else [10, 10, 5, 1]), np.eye(2), problem.Ts, 10,
        scene.veh.L, np.arctan(scene.veh.L/scene.veh.Rmin), 1. if parking else 1.5,
        {'ipopt': {'print_level': 0}, 'print_time': False})
    result = track_mpc(p)
    if any(status != 'Solve_Succeeded' for status in result.statuses):
        raise RuntimeError(f'MPC: {result.statuses}')
    actual = result.X_actual
    err_pos = np.linalg.norm(actual[:2]-reference.X[:2], axis=0)
    yaw = actual[2]-reference.X[2]
    err_yaw = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw)) if parking else yaw)
    print(f'MPC position RMS={np.sqrt(np.mean(err_pos**2)):.6g} m')
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    draw_scene(ax, scene)
    ax.plot(reference.X[0], reference.X[1], '--', color='royalblue', label='OBCA reference')
    ax.plot(actual[0], actual[1], color='firebrick', label='MPC actual')
    ax.legend(); ax.set_title('MPC trajectory tracking')
    save(fig, stem)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), layout='constrained')
    for ax, values, label in zip(axes, [err_pos, err_yaw], ['Position error [m]', 'Heading error [rad]']):
        ax.plot(values); ax.set(xlabel='Time step k', ylabel=label)
    save(fig, stem + '_errors')
    if animation:
        save_animation(stem, scene, reference.X, actual, problem.Ts)


def save_animation(stem, scene, reference, actual, Ts):
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    draw_scene(ax, scene)
    ax.set_xlim(scene.XYbounds[0]-1, scene.XYbounds[1]+1)
    ax.plot(reference[0], reference[1], '--', color='royalblue', label='OBCA reference')
    trail, = ax.plot([], [], color='firebrick', label='MPC actual')
    car, = ax.plot([], [], color='seagreen', linewidth=2, label='Vehicle')
    fig.legend(loc='outside upper center', ncol=5, fontsize=8)

    def update(k):
        vertices = vehicle_vertices(actual[:3, k], scene.veh.ego)
        vertices = np.vstack((vertices, vertices[0]))
        car.set_data(vertices[:, 0], vertices[:, 1])
        trail.set_data(actual[0, :k+1], actual[1, :k+1])
        ax.set_title(f't={k*Ts:.1f}s | v={actual[3,k]:.2f} m/s')
        return car, trail

    animation = FuncAnimation(fig, update, frames=actual.shape[1], interval=120, blit=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f'{stem}.gif'
    animation.save(path, writer=PillowWriter(fps=8.33), dpi=100)
    plt.close(fig)
    print(path)

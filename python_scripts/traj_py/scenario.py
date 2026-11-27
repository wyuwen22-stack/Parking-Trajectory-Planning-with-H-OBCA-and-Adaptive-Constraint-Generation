from dataclasses import dataclass

import numpy as np


@dataclass
class Vehicle:
    L: float
    ego: np.ndarray
    Rmin: float


@dataclass
class Scenario:
    veh: Vehicle
    XYbounds: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    obs: tuple[np.ndarray, ...]
    Aineq: tuple[np.ndarray, ...]
    bineq: tuple[np.ndarray, ...]
    occupancy: np.ndarray
    resolution: float


def _poly2hrep(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = vertices[:-1]
    center = vertices.mean(axis=0)
    A = np.zeros((len(vertices), 2))
    b = np.zeros(len(vertices))
    for k, p1 in enumerate(vertices):
        p2 = vertices[(k + 1) % len(vertices)]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        if normal @ (center - p1) > 0:
            normal = -normal
        A[k] = normal
        b[k] = normal @ p1
    return A, b


def simple_scenario() -> Scenario:
    veh = Vehicle(2.7, np.array([3.7, 1.0, 1.0, 1.0]), 5.0)
    bounds = np.array([0.0, 50.0, 0.0, 8.0])
    start = np.array([3.0, 2.0, 0.0])
    goal = np.array([47.0, 2.0, 0.0])
    obstacles = (
        np.array([[16, 0], [20.7, 0], [20.7, 2.2], [16, 2.2], [16, 0]]),
        np.array([[30, 3], [34, 3], [34, 8], [30, 8], [30, 3]]),
    )
    return _assemble(veh, bounds, start, goal, obstacles)


def parking_scenario() -> Scenario:
    veh = Vehicle(2.7, np.array([3.7, 1.0, 1.0, 1.0]), 5.0)
    bounds = np.array([-15.0, 15.0, 0.0, 12.0])
    obstacles = (
        np.array([[-12, 0], [-1.3, 0], [-1.3, 5], [-12, 5], [-12, 0]]),
        np.array([[1.3, 0], [12, 0], [12, 5], [1.3, 5], [1.3, 0]]),
        np.array([[-12., 11], [12, 11], [12, 12], [-12, 12], [-12, 11]]),
    )
    return _assemble(veh, bounds, np.array([-9., 7., np.pi]), np.array([0., 1.3, np.pi/2]), obstacles)


def _assemble(veh, bounds, start, goal, obstacles):
    halfspaces = tuple(_poly2hrep(vertices) for vertices in obstacles)

    resolution = 5.0
    rows = int((bounds[3] - bounds[2]) * resolution)
    cols = int((bounds[1] - bounds[0]) * resolution)
    xg, yg = np.meshgrid(
        np.linspace(bounds[0], bounds[1], cols), np.linspace(bounds[2], bounds[3], rows)
    )
    inside = np.zeros((rows, cols), dtype=bool)
    for vertices in obstacles:
        # All source polygons in these two scenarios are axis-aligned rectangles.
        inside |= (
            (vertices[:, 0].min() <= xg)
            & (xg <= vertices[:, 0].max())
            & (vertices[:, 1].min() <= yg)
            & (yg <= vertices[:, 1].max())
        )
    occupancy = np.zeros((rows, cols), dtype=bool)
    # MATLAB grid column = floor(x*resolution)+1, row = rows-floor(y*resolution).
    grid_cols = np.clip(np.floor((xg[inside] - bounds[0]) * resolution).astype(int), 0, cols - 1)
    grid_rows = np.clip(rows - 1 - np.floor((yg[inside] - bounds[2]) * resolution).astype(int), 0, rows - 1)
    occupancy[grid_rows, grid_cols] = True

    return Scenario(
        veh, bounds, start, goal, obstacles,
        tuple(A for A, _ in halfspaces), tuple(b for _, b in halfspaces),
        occupancy, resolution,
    )

"""Numerical metrics from analyze_results.m / analyze_parking.m."""
import numpy as np


def vehicle_vertices(pose, ego):
    xu, yu, xl, yl = ego
    corners = np.array([[xu, yu], [xu, -yl], [-xl, -yl], [-xl, yu]])
    theta = pose[2]
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return corners @ rotation.T + pose[:2]


def path_curvature(path):
    values = np.zeros(len(path))
    for i in range(1, len(path)-1):
        d1, d2 = path[i, :2]-path[i-1, :2], path[i+1, :2]-path[i, :2]
        a, b, c = np.linalg.norm(d1), np.linalg.norm(d2), np.linalg.norm(path[i+1, :2]-path[i-1, :2])
        area = .5*abs(d1[0]*d2[1]-d2[0]*d1[1])
        if a*b*c > 1e-9:
            values[i] = 4*area/(a*b*c)
    values[0], values[-1] = values[1], values[-2]
    return values


def approximate_clearance(poses, scene):
    distances = []
    for pose in poses:
        vertices = vehicle_vertices(pose, scene.veh.ego)
        distance = np.inf
        for obstacle in scene.obs:
            ob = obstacle[:-1]
            for point in vertices:
                for a, b in zip(ob, np.roll(ob, -1, axis=0)):
                    edge = b-a
                    t = np.clip((point-a) @ edge/max(edge @ edge, 1e-9), 0, 1)
                    distance = min(distance, np.linalg.norm(point-(a+t*edge)))
            inside = ((vertices >= ob.min(axis=0)) & (vertices <= ob.max(axis=0))).all(axis=1)
            if inside.any():
                distance = -abs(distance)
        distances.append(distance)
    return np.array(distances)

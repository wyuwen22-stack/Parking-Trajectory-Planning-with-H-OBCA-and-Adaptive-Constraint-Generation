# Adapted from PythonRobotics (MIT); see THIRD_PARTY_NOTICES.md.
import math
import numpy as np


class Path:
    """
    Path data container
    """

    def __init__(self):
        # course segment length  (negative value is backward segment)
        self.lengths = []
        # course segment type char ("S": straight, "L": left, "R": right)
        self.ctypes = []
        self.L = 0.0  # Total lengths of the path
        self.x = []  # x positions
        self.y = []  # y positions
        self.yaw = []  # orientations [rad]
        self.directions = []

def pi_2_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

def mod2pi(x):
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v

def set_path(paths, lengths, ctypes, step_size):
    path = Path()
    path.ctypes = ctypes
    path.lengths = lengths
    path.L = sum(np.abs(lengths))

    # check same path exist
    for i_path in paths:
        type_is_same = (i_path.ctypes == path.ctypes)
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        if type_is_same and length_is_close:
            return paths  # same path found, so do not insert path

    # Keep short-but-nonzero paths. Dropping L <= step_size used to discard
    # valid goal connections shorter than the validation step, so a node a
    # few centimetres from the goal could never close.
    if path.L <= 1e-12:
        return paths

    paths.append(path)
    return paths

def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta

def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']

    return False, [], []

def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ['L', 'S', 'R']

    return False, [], []

def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']

    return False, [], []

def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2*A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']

    return False, [], []

def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if 1e-12 < u1 <= 4.0:
        arg = 1 - u1**2 * 0.125
        if arg < -1.0 or arg > 1.0:
            return False, [], []
        u = math.acos(arg)
        s = 2 * math.sin(u) / u1
        if s < -1.0 or s > 1.0:
            return False, [], []
        A = math.asin(s)
        t = mod2pi(-A + theta + math.pi/2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']

    return False, [], []

def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi/2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2*u)
        if ((t >= 0) and (u >= 0) and (v >= 0)):
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']

    return False, [], []

def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if (0 <= u2 <= 1) and u1 > 1e-12:
        u = math.acos(u2)
        s = 2 * math.sin(u) / u1
        if s < -1.0 or s > 1.0:
            return False, [], []
        A = math.asin(s)
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']

    return False, [], []

def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi + math.pi/2)
        if (t >= 0) and (v >= 0):
           return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'L']

    return False, [], []

def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi/2)
        v = mod2pi(t - phi - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'R', 'L']

    return False, [], []

def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi/2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'R']

    return False, [], []

def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'L', 'R']

    return False, [], []

def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -math.pi/2, v], ['L', 'R', 'S', 'L', 'R']

    return False, [], []

def timeflip(travel_distances):
    return [-x for x in travel_distances]

def reflect(steering_directions):
    def switch_dir(dirn):
        if dirn == 'L':
            return 'R'
        elif dirn == 'R':
            return 'L'
        else:
            return 'S'
    return[switch_dir(dirn) for dirn in steering_directions]

def generate_path(q0, q1, max_curvature, step_size):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]
    c = math.cos(q0[2])
    s = math.sin(q0[2])
    x = (c * dx + s * dy) * max_curvature
    y = (-s * dx + c * dy) * max_curvature
    step_size *= max_curvature

    paths = []
    path_functions = [left_straight_left, left_straight_right,                          # CSC
                      left_x_right_x_left, left_x_right_left, left_right_x_left,        # CCC
                      left_right_x_left_right, left_x_right_left_x_right,               # CCCC
                      left_x_right90_straight_left, left_x_right90_straight_right,      # CCSC
                      left_straight_right90_x_left, left_straight_left90_x_right,       # CSCC
                      left_x_right90_straight_left90_x_right]                           # CCSCC

    def _safe(path_func, x, y, dth):
        try:
            return path_func(x, y, dth)
        except (ValueError, ZeroDivisionError, OverflowError):
            return False, [], []

    for path_func in path_functions:
        flag, travel_distances, steering_dirns = _safe(path_func, x, y, dth)
        if flag:
            total = sum(abs(d) for d in travel_distances)
            # A short *total* path is a valid goal connection; do not discard it.
            # A tiny *segment* of a long path is a known degenerate family — skip
            # that family only, do not abort the other Reeds-Shepp families.
            if total > step_size:
                tiny = any(0.1 * total < abs(d) < step_size for d in travel_distances)
                if tiny:
                    continue
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = _safe(path_func, -x, y, -dth)
        if flag:
            total = sum(abs(d) for d in travel_distances)
            # A short *total* path is a valid goal connection; do not discard it.
            # A tiny *segment* of a long path is a known degenerate family — skip
            # that family only, do not abort the other Reeds-Shepp families.
            if total > step_size:
                tiny = any(0.1 * total < abs(d) < step_size for d in travel_distances)
                if tiny:
                    continue
            travel_distances = timeflip(travel_distances)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = _safe(path_func, x, -y, -dth)
        if flag:
            total = sum(abs(d) for d in travel_distances)
            # A short *total* path is a valid goal connection; do not discard it.
            # A tiny *segment* of a long path is a known degenerate family — skip
            # that family only, do not abort the other Reeds-Shepp families.
            if total > step_size:
                tiny = any(0.1 * total < abs(d) < step_size for d in travel_distances)
                if tiny:
                    continue
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        flag, travel_distances, steering_dirns = _safe(path_func, -x, -y, dth)
        if flag:
            total = sum(abs(d) for d in travel_distances)
            # A short *total* path is a valid goal connection; do not discard it.
            # A tiny *segment* of a long path is a known degenerate family — skip
            # that family only, do not abort the other Reeds-Shepp families.
            if total > step_size:
                tiny = any(0.1 * total < abs(d) < step_size for d in travel_distances)
                if tiny:
                    continue
            travel_distances = timeflip(travel_distances)
            steering_dirns = reflect(steering_dirns)
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths

def calc_interpolate_dists_list(lengths, step_size):
    interpolate_dists_list = []
    for length in lengths:
        d_dist = step_size if length >= 0.0 else -step_size
        interp_dists = np.arange(0.0, length, d_dist)
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list

def generate_local_course(lengths, modes, max_curvature, step_size):
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0

    xs, ys, yaws, directions = [], [], [], []
    for (interp_dists, mode, length) in zip(interpolate_dists_list, modes,
                                            lengths):

        for dist in interp_dists:
            x, y, yaw, direction = interpolate(dist, length, mode,
                                               max_curvature, origin_x,
                                               origin_y, origin_yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        origin_x = xs[-1]
        origin_y = ys[-1]
        origin_yaw = yaws[-1]

    return xs, ys, yaws, directions

def interpolate(dist, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw):
    if mode == "S":
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw
    else:  # curve
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(dist)) / max_curvature
            yaw = origin_yaw + dist
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(dist)) / -max_curvature
            yaw = origin_yaw - dist
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        x = origin_x + gdx
        y = origin_y + gdy

    return x, y, yaw, 1 if length > 0.0 else -1

def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    q0 = [sx, sy, syaw]
    q1 = [gx, gy, gyaw]
    dpsi = abs((syaw - gyaw + math.pi) % (2 * math.pi) - math.pi)
    if math.hypot(gx - sx, gy - sy) <= 1e-9 and dpsi <= 1e-9:
        path = Path()
        path.lengths = [0.0]
        path.ctypes = ['S']
        path.L = 0.0
        path.x = [sx, gx]
        path.y = [sy, gy]
        path.yaw = [syaw, gyaw]
        path.directions = [1, 1]
        return [path]

    paths = generate_path(q0, q1, maxc, step_size)
    for path in paths:
        xs, ys, yaws, directions = generate_local_course(path.lengths,
                                                         path.ctypes, maxc,
                                                         step_size)

        # convert global coordinate
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for
                  (ix, iy) in zip(xs, ys)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for
                  (ix, iy) in zip(xs, ys)]
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        path.directions = directions
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc

    return paths

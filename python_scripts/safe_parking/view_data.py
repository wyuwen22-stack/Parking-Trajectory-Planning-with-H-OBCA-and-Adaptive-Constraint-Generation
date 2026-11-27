"""Read saved artifacts and reconstruct curves; never imports a planner or solver."""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .geometry import Geometry
from .motion import arc, rollout, speed_roots

VARIANTS = ('node', 'dense', 'adaptive')
LABELS = {'node': 'Node constraints', 'dense': 'Fixed 3 points / segment',
          'adaptive': 'Adaptive constraints'}
COLORS = {'node': '#d55e00', 'dense': '#0072b2', 'adaptive': '#009e73'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def case_ids(root):
    return sorted(p.name for p in Path(root).iterdir()
                  if p.is_dir() and (p / 'scene.npz').exists())


@dataclass
class Trace:
    variant: str
    round: int
    data: dict
    record: dict
    status: str
    accepted: bool
    times: np.ndarray
    poses: np.ndarray
    clearance: np.ndarray
    nodes: np.ndarray
    additions: list
    mode: str

    @property
    def duration(self):
        return self.data['U'].shape[1] * float(self.data['Ts'])

    def at(self, time):
        T = float(self.data['Ts'])
        n = self.data['U'].shape[1]
        time = np.clip(time, 0., n * T)
        k = min(int(time / T), n - 1)
        return arc(self.nodes[:, k], self.data['U'][:, k], time - k * T,
                   float(self.data['L']))


@dataclass
class Case:
    id: str
    scene: dict
    traces: dict = field(default_factory=dict)
    statuses: dict = field(default_factory=dict)
    available_rounds: list = field(default_factory=list)
    ha: dict = field(default_factory=dict)

    @property
    def obstacles(self):
        return [v for k, v in self.scene.items() if k.startswith('obs_')]


def build_trace(variant, round_no, data, record, mode='rollout'):
    X, U, T, L = data['X'], data['U'], float(data['Ts']), float(data['L'])
    if mode not in ('rollout', 'saved'):
        raise ValueError('Unknown reconstruction mode')
    if X.shape != (4, U.shape[1] + 1) or U.shape[0] != 2:
        raise ValueError('Invalid saved trajectory shape')
    if not (np.isfinite(X).all() and np.isfinite(U).all() and T > 0 and L > 0):
        raise ValueError('Non-finite trajectory or invalid time/vehicle metadata')
    nodes = rollout(np.r_[data['start'], 0.], U, T, L) if mode == 'rollout' else X
    g = Geometry([v for k, v in data.items() if k.startswith('obs_')], data['ego'])
    samples = np.asarray(data.get('constraint_samples', []), dtype=float).reshape(-1, 2)
    rounds = np.asarray(data.get('constraint_added_round', []), dtype=int)
    if len(samples) != len(rounds):
        raise ValueError('Constraint samples and introduction rounds do not match')
    fractions = [set(np.linspace(0., 1., 13)) for _ in range(U.shape[1])]
    for k, f in samples:
        if not (int(k) == k and 0 <= k < len(fractions) and 0 < f < 1):
            raise ValueError('Invalid constraint location')
        fractions[int(k)].add(float(f))
    # Include the independent validator's collision witnesses, not just a uniform grid.
    for k, f, *_ in record.get('validation', {}).get('curve', {}).get('witnesses', []):
        fractions[int(k)].add(float(f))
    times, poses = [], []
    for k, fs in enumerate(fractions):
        fs.update(t / T for t in speed_roots(nodes[3, k], U[1, k], T))
        for f in sorted(fs):
            times.append((k + f) * T)
            poses.append(arc(nodes[:, k], U[:, k], f * T, L))
    poses = np.asarray(poses)
    distances = np.asarray([g.clearance(q) for q in poses])
    additions = []
    for (k, f), round_added in zip(samples, rounds):
        if round_added <= 1:  # round 1 is the initial midpoint set, not a new insertion
            continue
        q = arc(nodes[:, int(k)], U[:, int(k)], f * T, L)
        additions.append(dict(segment=int(k), fraction=float(f), round=int(round_added),
                              time=float((k + f) * T), pose=q,
                              clearance=g.clearance(q)))
    status = record.get('status', 'UNFINISHED')
    accepted = bool(record.get('validation', {}).get('accepted', False)
                    and status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'))
    return Trace(variant, round_no, data, record, status, accepted,
                 np.asarray(times), poses, distances, nodes, additions, mode)


def load_case(root, identifier, adaptive_round=None, mode='rollout'):
    root = Path(root).resolve()
    folder = (root / identifier).resolve()
    if folder.parent != root:
        raise ValueError('Case ID must name a direct child of the results directory')
    scene = read_npz(folder / 'scene.npz')
    case = Case(identifier, scene)
    if (folder / 'ha.json').exists():
        case.ha = read_json(folder / 'ha.json')
    for variant in VARIANTS:
        target = folder / variant
        result = read_json(target / 'result.json') if (target / 'result.json').exists() else {}
        case.statuses[variant] = result.get('status', 'NOT_RUN')
        if result.get('stop_reason') and not result.get('accepted'):
            case.statuses[variant] += ':' + result['stop_reason']
        available = sorted(int(p.stem[5:]) for p in target.glob('round*.npz')
                           if p.with_suffix('.json').exists())
        if variant == 'adaptive':
            case.available_rounds = available
        if not available:
            continue  # Missing results stay missing. There is no planning fallback.
        selected = adaptive_round if variant == 'adaptive' and adaptive_round is not None else available[-1]
        if selected not in available:
            raise ValueError(f'Adaptive round {selected} is not available')
        data = read_npz(target / f'round{selected:02}.npz')
        for name in ('L', 'Rmin', 'ego', 'start', 'goal', 'bounds'):
            if name not in data or not np.allclose(data[name], scene[name], atol=1e-10, rtol=0):
                raise ValueError(f'{variant}: inconsistent or missing {name} metadata')
        record = read_json(target / f'round{selected:02}.json')
        trace = build_trace(variant, selected, data, record, mode)
        if selected == available[-1]:
            trace.accepted = trace.accepted and bool(result.get('accepted', False))
        case.traces[variant] = trace
    return case

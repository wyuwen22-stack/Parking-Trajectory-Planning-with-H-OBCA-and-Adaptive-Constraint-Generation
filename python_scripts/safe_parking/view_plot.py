"""Matplotlib comparison figure. Only consumes loaded results."""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from .geometry import body, vertices
from .view_data import COLORS, LABELS, VARIANTS

ROUND_COLORS = {2: '#882255', 3: '#cc6677', 4: '#aa4499', 5: '#ddaa33', 6: '#332288'}


def contact_center(car, obstacles):
    """Closest boundary pair, for placing a useful local zoom (not a safety test)."""
    best = (float('inf'), car.mean(axis=0))
    for obstacle in obstacles:
        for a, b in ((car, vertices(obstacle)), (vertices(obstacle), car)):
            edges = np.roll(b, -1, axis=0) - b
            for p in a:
                f = np.clip(np.sum((p - b) * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
                closest = b + f[:, None] * edges
                ds = np.linalg.norm(closest - p, axis=1)
                i = int(np.argmin(ds))
                if ds[i] < best[0]:
                    best = (ds[i], (p + closest[i]) / 2)
    return best[1]


class ComparisonPlot:
    def __init__(self):
        self.figure = Figure(figsize=(14, 8.5), dpi=100, facecolor='white')
        self.figure.subplots_adjust(left=.065, right=.975, top=.90, bottom=.22,
                                    wspace=.17, hspace=.38)
        grid = self.figure.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1.4, 1])
        self.map_ax = self.figure.add_subplot(grid[0, 0])
        self.zoom_ax = self.figure.add_subplot(grid[0, 1])
        self.curve_ax = self.figure.add_subplot(grid[1, :])
        self.title = self.figure.suptitle('', x=.065, y=.985, ha='left', fontsize=14, weight='bold')
        self.subtitle = self.figure.text(.065, .945, '', fontsize=10, color='#444444')
        self.status = self.figure.text(.065, .105, '', fontsize=9, linespacing=1.4)
        self.note = self.figure.text(.065, .018, '', fontsize=8, color='#555555')
        self.artists = []
        self.case = None
        self.cursor = None
        self.current_time = 0.

    def _scene(self, ax):
        for obstacle in self.case.obstacles:
            ax.add_patch(Polygon(obstacle, facecolor='#c5c9cf', edgecolor='#727983', linewidth=.7))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(alpha=.15)

    def render(self, case, footprints=True, risk_variant='auto', risk_time=None):
        self.case = case
        self.artists = []
        self.cursor = None
        for ax in (self.map_ax, self.zoom_ax, self.curve_ax):
            ax.clear()
        self._scene(self.map_ax)
        self._scene(self.zoom_ax)
        b = case.scene['bounds']
        self.map_ax.set_xlim(b[0] - .5, b[1] + .5)
        self.map_ax.set_ylim(b[2] - .5, b[3] + .5)
        self.map_ax.set_title('Rear-axle paths and body outlines', loc='left', fontsize=11)
        traces = case.traces
        radius = float(case.scene['Rmin'])
        self.title.set_text(case.id)
        self.subtitle.set_text(f'Rmin = {radius:g} m  |  max steering = '
                               f'{np.degrees(np.arctan(float(case.scene["L"]) / radius)):.2f} deg'
                               '  |  saved results only')
        for point, label, color in ((case.scene['start'], 'Start', '#444444'),
                                    (case.scene['goal'], 'Goal', '#222222')):
            self.map_ax.add_patch(Polygon(body(point, case.scene['ego']), fill=False,
                                          edgecolor=color, linewidth=1.3, linestyle='--'))
            self.map_ax.annotate(label, point[:2], xytext=(4, 8), textcoords='offset points', fontsize=8)
        if traces:
            pool = [traces[risk_variant]] if risk_variant in traces else list(traces.values())
            risk_trace = min(pool, key=lambda tr: float(np.min(tr.clearance)))
            index = int(np.argmin(risk_trace.clearance))
            time = float(risk_trace.times[index]) if risk_time is None else float(risk_time)
            q = risk_trace.at(time)
            center = contact_center(body(q, case.scene['ego']), case.obstacles)
            half = 1.15
            self.zoom_ax.set_xlim(center[0] - half, center[0] + half)
            self.zoom_ax.set_ylim(center[1] - half, center[1] + half)
            self.map_ax.add_patch(Rectangle(center - half, 2 * half, 2 * half,
                                            fill=False, edgecolor='#555555', linestyle=':'))
            self.zoom_ax.set_title(f'Local risk: {risk_trace.variant}, t = {time:.2f} s',
                                   loc='left', fontsize=11)
            max_time = max(tr.duration for tr in traces.values())
            for variant in VARIANTS:
                if variant not in traces:
                    continue
                tr = traces[variant]
                color = COLORS[variant]
                label = f'{LABELS[variant]} (R{tr.round})'
                self.map_ax.plot(tr.poses[:, 0], tr.poses[:, 1], color=color, lw=1.7, label=label)
                self.map_ax.plot(tr.nodes[0], tr.nodes[1], '.', color=color, ms=2, alpha=.6)
                self.zoom_ax.plot(tr.poses[:, 0], tr.poses[:, 1], color=color, lw=1.6)
                if footprints:
                    for t in np.linspace(0., tr.duration, 7)[1:-1]:
                        self.map_ax.add_patch(Polygon(body(tr.at(t), case.scene['ego']), fill=False,
                                                      edgecolor=color, alpha=.28, linewidth=.7))
                self.zoom_ax.add_patch(Polygon(body(tr.at(time), case.scene['ego']), fill=False,
                                               edgecolor=color, linewidth=1.8))
                vehicle = Polygon(body(tr.at(0), case.scene['ego']), facecolor=color,
                                  alpha=.2, edgecolor=color, linewidth=1.8)
                self.map_ax.add_patch(vehicle)
                self.artists.append((tr, vehicle))
                self.curve_ax.plot(tr.times, tr.clearance, color=color, lw=1.4, label=label)
                # Node dots make the inter-node dip visible on the clearance plot.
                indices = np.flatnonzero(np.isclose(tr.times / float(tr.data['Ts']),
                                                   np.round(tr.times / float(tr.data['Ts']))))
                self.curve_ax.plot(tr.times[indices], tr.clearance[indices], '.', color=color, ms=2.5)
            adaptive = traces.get('adaptive')
            if adaptive:
                for r in sorted({a['round'] for a in adaptive.additions}):
                    added = [a for a in adaptive.additions if a['round'] == r]
                    color = ROUND_COLORS.get(r, '#882255')
                    pts = np.array([a['pose'] for a in added])
                    for ax in (self.map_ax, self.zoom_ax):
                        ax.scatter(pts[:, 0], pts[:, 1], marker='^', s=30, color=color,
                                   edgecolor='white', linewidth=.4, zorder=6)
                    self.curve_ax.scatter([a['time'] for a in added], [a['clearance'] for a in added],
                                          marker='^', s=32, color=color, edgecolor='white',
                                          linewidth=.4, zorder=6, label=f'Added R{r} ({len(added)})')
            self.map_ax.legend(loc='upper left', fontsize=8, framealpha=.9)
            self.curve_ax.set_xlim(0, max_time)
            self.curve_ax.axvline(time, color='#777777', ls=':', lw=1)
            self.cursor = self.curve_ax.axvline(0., color='#222222', alpha=.7, lw=1)
            self.curve_ax.legend(loc='upper right', ncol=4, fontsize=8, framealpha=.9)
            details = []
            for variant in VARIANTS:
                if variant in traces:
                    tr = traces[variant]
                    drift = tr.record.get('validation', {}).get('global_position_drift_m', float('nan'))
                    verdict = 'PASS' if tr.accepted else 'NOT ACCEPTED'
                    final_status = case.statuses[variant].replace('PASS_CONDITIONAL_NUMERICAL', 'PASS')
                    details.append(f'{variant} R{tr.round}: {verdict}, run={final_status}, solver={tr.status}, '
                                   f'plotted min={np.min(tr.clearance):.4f} m, drift={drift:.2g} m')
                else:
                    details.append(f'{variant}: {case.statuses[variant]} (no saved trajectory)')
            self.status.set_text('\n'.join(details))
            mode = next(iter(traces.values())).mode
            self.note.set_text(('Global control rollout (validator model).' if mode == 'rollout'
                                else 'Local arcs reset at saved nodes; not the global validator model.')
                               + '  Curves are finite samples, not certificates.  '
                               'Triangles: constraint times mapped to the displayed trajectory; R1 seeds excluded.')
        else:
            self.zoom_ax.set_title('No saved optimized trajectory', loc='left', fontsize=11)
            self.zoom_ax.set_xlim(b[:2]); self.zoom_ax.set_ylim(b[2:])
            self.status.set_text('HA: ' + case.ha.get('status', 'NOT_RUN') + '\n'
                                 + '  |  '.join(f'{v}: {case.statuses[v]}' for v in VARIANTS))
            self.note.set_text('No planner was run. Use run_planning.py explicitly to generate new results.')
        self.curve_ax.axhline(.05, color='#777777', ls='--', lw=1)
        self.curve_ax.axhline(0., color='#bb2222', lw=1)
        self.curve_ax.set_title('Body-obstacle clearance  |  dashed: optimization margin 0.05 m; red: contact 0 m',
                                loc='left', fontsize=10, pad=10)
        self.curve_ax.set_xlabel('Time [s]')
        self.curve_ax.set_ylabel('Signed clearance [m]')
        self.curve_ax.grid(alpha=.2)
        self.update_time(0.)

    def update_time(self, time):
        self.current_time = float(time)
        for tr, patch in self.artists:
            patch.set_xy(body(tr.at(time), self.case.scene['ego']))
        if self.cursor is not None:
            self.cursor.set_xdata([time, time])

    def save(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filename, dpi=180, facecolor='white')
        return filename

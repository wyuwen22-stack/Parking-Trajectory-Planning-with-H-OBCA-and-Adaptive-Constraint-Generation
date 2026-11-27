"""Desktop viewer: case/round selection, playback, and explicit figure export."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .view_data import case_ids, load_case, read_json
from .view_plot import ComparisonPlot


def default_case(root, identifiers):
    # Prefer a saved example that actually contains adaptive insertions.
    for identifier in identifiers:
        p = Path(root) / identifier / 'adaptive' / 'result.json'
        if p.exists():
            r = read_json(p)
            if r.get('accepted') and len(r.get('history', [])) > 1:
                return identifier
    return identifiers[0]


class Viewer:
    def __init__(self, root, results, identifier=None, adaptive_round=None, mode='rollout'):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.root = root
        self.results = Path(results).resolve()
        self.ids = case_ids(self.results)
        if not self.ids:
            raise ValueError('No scene.npz in the results directory. The viewer does not start planning. Run run_planning.py explicitly.')
        self.playing = False
        self.closed = False
        self.timer = None
        self.root.title('Parking Safe 128 | Saved-results viewer')
        self.root.geometry('1400x920')
        self.root.minsize(1050, 750)
        top = ttk.Frame(root, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text='Case').grid(row=0, column=0, padx=4)
        self.case_var = tk.StringVar(value=identifier or default_case(self.results, self.ids))
        self.case_box = ttk.Combobox(top, textvariable=self.case_var, values=self.ids, width=47)
        self.case_box.grid(row=0, column=1, sticky='ew')
        self.case_box.bind('<<ComboboxSelected>>', self.select_case)
        self.case_box.bind('<Return>', self.select_case)
        ttk.Button(top, text='Previous', command=lambda: self.step_case(-1)).grid(row=0, column=2, padx=3)
        ttk.Button(top, text='Next', command=lambda: self.step_case(1)).grid(row=0, column=3, padx=3)
        ttk.Label(top, text='Adaptive round').grid(row=0, column=4, padx=6)
        self.round_var = tk.StringVar(value=str(adaptive_round) if adaptive_round else 'Final')
        self.round_box = ttk.Combobox(top, textvariable=self.round_var, values=['Final'], width=7, state='readonly')
        self.round_box.grid(row=0, column=5)
        self.round_box.bind('<<ComboboxSelected>>', self.refresh)
        self.footprints = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='Body outlines', variable=self.footprints, command=self.refresh).grid(row=0, column=6, padx=7)
        ttk.Button(top, text='Save figure', command=self.save_image).grid(row=0, column=7, padx=4)
        ttk.Label(top, text='Curve model').grid(row=1, column=0, pady=7)
        self.mode_var = tk.StringVar(value='Global control rollout (acceptance model)' if mode == 'rollout' else 'Per-segment optimized arcs (node reset)')
        mode_box = ttk.Combobox(top, textvariable=self.mode_var, width=42, state='readonly',
                               values=['Global control rollout (acceptance model)', 'Per-segment optimized arcs (node reset)'])
        mode_box.grid(row=1, column=1, sticky='w')
        mode_box.bind('<<ComboboxSelected>>', self.refresh)
        ttk.Label(top, text='Risk focus').grid(row=1, column=2)
        self.risk_var = tk.StringVar(value='auto')
        risk_box = ttk.Combobox(top, textvariable=self.risk_var, width=10, state='readonly',
                               values=['auto', 'node', 'dense', 'adaptive'])
        risk_box.grid(row=1, column=3)
        risk_box.bind('<<ComboboxSelected>>', self.refresh)
        ttk.Label(top, text='Click the clearance curve to move the local zoom. Triangle color marks the insertion round.').grid(row=1, column=4, columnspan=4, padx=6)
        top.columnconfigure(1, weight=1)
        bottom = ttk.Frame(root, padding=(10, 4))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.message = tk.StringVar(value='Reads saved results only. Search and optimization are not started.')
        ttk.Label(bottom, textvariable=self.message, wraplength=1300).pack(side=tk.BOTTOM, fill=tk.X, pady=3)
        self.play_button = ttk.Button(bottom, text='Play', command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        self.time_var = tk.DoubleVar(value=0.)
        self.slider = ttk.Scale(bottom, from_=0., to=1., variable=self.time_var, command=self.seek)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.time_label = ttk.Label(bottom, text='0.00 s', width=10)
        self.time_label.pack(side=tk.LEFT, padx=8)
        self.plot = ComparisonPlot()
        self.canvas = FigureCanvasTkAgg(self.plot.figure, master=root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.pick_time)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()

    @lru_cache(maxsize=6)
    def loaded(self, identifier, round_no, mode):
        return load_case(self.results, identifier, round_no, mode)

    def select_case(self, event=None):
        self.round_var.set('Final')
        self.refresh()

    def step_case(self, offset):
        i = self.ids.index(self.case_var.get()) if self.case_var.get() in self.ids else 0
        self.case_var.set(self.ids[(i + offset) % len(self.ids)])
        self.select_case()

    def stop(self):
        self.playing = False
        self.play_button.configure(text='Play')
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def refresh(self, event=None):
        from tkinter import messagebox
        self.stop()
        identifier = self.case_var.get()
        if identifier not in self.ids:
            self.message.set('Unknown case ID. Choose one from the list, or type the full ID and press Enter.')
            return
        self.root.configure(cursor='watch')
        self.message.set('Reading saved results and sampled clearance. Planning is not restarted.')
        self.root.update_idletasks()
        try:
            round_no = None if self.round_var.get() == 'Final' else int(self.round_var.get())
            mode = 'rollout' if self.mode_var.get().startswith('Global') else 'saved'
            self.case = self.loaded(identifier, round_no, mode)
            self.round_box.configure(values=['Final'] + [str(r) for r in self.case.available_rounds])
            self.plot.render(self.case, self.footprints.get(), self.risk_var.get())
            self.duration = max((tr.duration for tr in self.case.traces.values()), default=0.)
            self.slider.configure(to=max(self.duration, .001))
            self.play_button.configure(state='normal' if self.case.traces else 'disabled')
            self.time_var.set(0.)
            self.seek(0.)
            self.toolbar.update()
            self.canvas.draw_idle()
            count = len(self.case.traces.get('adaptive').additions) if 'adaptive' in self.case.traces else 0
            self.message.set(f'Read {identifier}; {count} added constraint(s) after the round-1 midpoints. '
                             'NOT ACCEPTED means the saved run did not pass acceptance. The trajectory is diagnostic only.')
        except (OSError, ValueError, KeyError) as exc:
            self.message.set(f'Failed to read results: {exc}. Planning was not started.')
            messagebox.showerror('Failed to read results', str(exc), parent=self.root)
        finally:
            self.root.configure(cursor='')

    def seek(self, value):
        time = float(value)
        self.plot.update_time(time)
        self.time_label.configure(text=f'{time:.2f} s')
        self.canvas.draw_idle()

    def toggle_play(self):
        if self.playing:
            self.stop()
        elif getattr(self, 'duration', 0.) > 0:
            if self.time_var.get() >= self.duration:
                self.time_var.set(0.)
            self.playing = True
            self.play_button.configure(text='Pause')
            self.tick()

    def tick(self):
        self.timer = None
        if not self.playing or self.closed:
            return
        time = min(self.time_var.get() + .1, self.duration)
        self.time_var.set(time)
        self.seek(time)
        if time >= self.duration:
            self.stop()
        else:
            self.timer = self.root.after(100, self.tick)

    def pick_time(self, event):
        if event.inaxes is not self.plot.curve_ax or event.xdata is None or self.toolbar.mode:
            return
        self.stop()
        time = max(0., min(event.xdata, self.duration))
        self.plot.render(self.case, self.footprints.get(), self.risk_var.get(), time)
        self.time_var.set(time)
        self.seek(time)
        adaptive = self.case.traces.get('adaptive')
        if adaptive and adaptive.additions:
            point = min(adaptive.additions, key=lambda a: abs(a['time'] - time))
            self.message.set(f"Nearest added constraint: round {point['round']}, segment {point['segment']} (0-based), "
                             f"fraction {point['fraction']:.6f}, t={point['time']:.3f} s.")

    def save_image(self):
        from tkinter import filedialog
        self.stop()
        mode_tag = 'rollout' if self.mode_var.get().startswith('Global') else 'saved'
        name = f'{self.case.id}_{mode_tag}_{datetime.now():%Y%m%d_%H%M%S}.png'
        filename = filedialog.asksaveasfilename(parent=self.root, title='Save the current comparison figure',
                                               initialfile=name, defaultextension='.png',
                                               filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if filename:
            self.plot.save(filename)
            self.message.set(f'Saved: {filename}')

    def close(self):
        self.closed = True
        self.stop()
        self.root.destroy()


def launch(results, identifier=None, adaptive_round=None, mode='rollout'):
    import tkinter as tk
    root = tk.Tk()
    try:
        Viewer(root, results, identifier, adaptive_round, mode)
        root.mainloop()
    except Exception:
        if root.winfo_exists():
            root.destroy()
        raise

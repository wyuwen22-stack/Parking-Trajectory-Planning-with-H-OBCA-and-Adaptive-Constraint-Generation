# Documentation

- [Project preview](../README.html) and [GitHub README](../README.md).
- [128-case atlas](CASE_ATLAS_EN.html): layouts, acceptance filters and three-method outcomes.
- [Results](../RESULTS.md): 84 node-only, 89 fixed-interior and 98 adaptive acceptances out of 128 cases.
- [Implementation notes](IMPLEMENTATION_NOTES.md): acceptance contract and known execution issues.
- [HA failure study](HA_FAILURE_REPORT_EN.html): separate front-end investigation; its 22/24 recoveries and historical 126/128 ledger do not replace the backend scores.
- `figures/parking_motion_three_panel.gif`: the full-motion viewer-style animation linked by the README.
- `figures/scene_overview.png`, `figures/adaptive_repair.png` and `figures/method_pipeline.png`: README illustrations.
- `FIGURE_PROVENANCE.json` and `figures/parking_motion_three_panel.json`: figure source records.

The HA study includes its report and tabulated evidence here. Its independent recovery runner and complete study trajectories are not included; `run_planning.py --source fresh` runs the base front end, not the study's recovery sequence. Older viewer-export figures are retained as historical assets.

# Frozen backend benchmark (v2)

All three methods use the same 112 historical HA seeds. The 16 cases without a seed remain in the 128-case denominator. Acceptance requires every check in the declared contract to pass. The later HA recovery study does not change this result snapshot.

| Family | Cases | Node-only | Fixed interior | Adaptive |
|---|---:|---:|---:|---:|
| Parallel and perpendicular | 60 | 34/60 (56.67%) | 38/60 (63.33%) | 43/60 (71.67%) |
| Angled | 36 | 33/36 (91.67%) | 36/36 (100.00%) | 36/36 (100.00%) |
| Irregular obstacles | 12 | 8/12 (66.67%) | 8/12 (66.67%) | 9/12 (75.00%) |
| Adapted TPCAP | 20 | 9/20 (45.00%) | 7/20 (35.00%) | 10/20 (50.00%) |
| Total | 128 | 84/128 (65.63%) | 89/128 (69.53%) | 98/128 (76.56%) |

## Final outcome categories

These categories are mutually exclusive. A solver stop takes precedence over a collision found in its returned iterate. “Solver-reported infeasibility” is not a proof that the continuous parking task has no solution.

| Outcome | Node-only | Fixed interior | Adaptive |
|---|---:|---:|---:|
| Accepted | 84 | 89 | 98 |
| No HA seed | 16 | 16 | 16 |
| Solver-reported infeasibility | 3 | 20 | 11 |
| Iteration limit reached | 1 | 0 | 1 |
| Worker / native-process failure | 0 | 1 | 0 |
| Collision after a successful solve | 24 | 2 | 2 |

## Recovery subsets

Twenty-four node-only cases meet the solver and node-diagnostic conditions but have a between-node collision. Fifteen are accepted by the adaptive workflow, including nine also accepted by fixed interior constraints. These are cross-method comparisons.

The 11 cases below instead start with a solver-successful, node-safe but colliding first round of the adaptive workflow itself, and subsequently pass all checks:

| Case | Accepted adaptive round |
|---|---:|
| `perpendicular_tight_x-8_y8.5` | 2 |
| `perpendicular_wide_x-8_y8.5` | 3 |
| `parallel_medium_x0_y7.5` | 2 |
| `parallel_medium_x8_y7.5` | 5 |
| `parallel_medium_x8_y8.5` | 5 |
| `parallel_wide_x-4_y7.5` | 5 |
| `parallel_wide_x8_y7.5` | 6 |
| `parallel_wide_x8_y8.5` | 2 |
| `diagonal_wide_a60_x-8_y8.5` | 2 |
| `irregular_convex_hexagon_x-8_y8.5` | 2 |
| `tpcap_case01_project_4.7x2` | 5 |

Adaptive has up to six solves; the two baselines have one. The comparison measures workflow outcomes and does not isolate constraint-point selection from repeated solving.

The [case atlas](docs/CASE_ATLAS_EN.html) contains all 128 cases and their three-method statuses. Raw source records are `results/seeded/results.json`, `results/seeded/results.csv` and the per-case directories. The [HA failure study](docs/HA_FAILURE_REPORT_EN.html) is a separate front-end experiment.

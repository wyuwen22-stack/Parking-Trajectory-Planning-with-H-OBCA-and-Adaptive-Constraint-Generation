# Implementation notes

## Acceptance and evidence

The frozen backend benchmark uses the `whole_body_arc_zero_terminal_v2` acceptance contract. Allowed solver states are `Solve_Succeeded` and `Solved_To_Acceptable_Level`. The complete control sequence is replayed from the true initial state. Dynamics residuals, replay drift, controls, initial and terminal conditions, the declared reference-point boundary check and interval obstacle clearance must all pass.

`COLLISION`, `UNKNOWN` and `NOT_RUN` are not accepted. A geometric `PASS` does not override another failed check. The 0.05 m node diagnostic is retained as a diagnostic only. Optimizer constraints at selected poses still use 0.05 m plus a 0.001 m buffer.

The adaptive loop starts with interval midpoints. Each refinement may add at most 24 new interior locations. It stops after acceptance, a solver stop, six total rounds or the absence of new eligible points. It does not increase the control-grid resolution. The solver, stop reason, verification status and failed checks are recorded separately.

## Recorded example

The README illustration uses only the saved adaptive rounds of `tpcap_case01_project_4.7x2`:

| Quantity | Round 1 | Round 5 |
|---|---:|---:|
| Solver status | Solve_Succeeded | Solve_Succeeded |
| Saved node minimum clearance | 0.051000004 m | 0.051000010 m |
| Interval state | COLLISION | PASS |
| Minimum evaluated clearance in the interval check | −0.029799856 m | 0.021944260 m |
| Certified whole-trajectory lower bound | Not available | 0.002041777 m |
| Overall acceptance | No | Yes |

Round 1 records an interior-overlap witness on zero-based interval 32 at fraction 0.25. With `Ts = 0.4 s`, its time is `(32 + 0.25) × 0.4 = 12.90 s`. Both body snapshots in the README use that time. The displayed clearance curves are new samples of the saved control rollout, not new planning results or replacement certificates.

The fixed-interior method in this case reported `Infeasible_Problem_Detected` and returned no saved trajectory. That status does not prove the continuous parking problem is infeasible.

## Known execution issues

The fixed-interior run for `tpcap_case05_project_4.7x2` ended with Windows exit code `3221225477` (`0xC0000005`, access violation). No trajectory was returned. The `WORKER_ERROR / NOT_RUN` result remains a failure in the benchmark; its native-process root cause has not been identified.

If a positive HA time limit is restored, a `RuntimeError` handler in the search can lose the original `SearchStopped` type. The current contract has no project time cap. Some outer worker-error records do not contain `code_sha256`; the frozen experiment and case directory provide the associated provenance.

## Historical records

The backend benchmark reuses 112 HA paths from a prior 90 s search; 104 passed the historical sampled 5 cm check. Fourteen searches timed out and two exhausted the queue. These records are distinct from the later HA recovery study.

`readonly_inputs_sha256.json` and `inputs/counts_128.json` contain historical source-tree labels. Those labels are not runtime data paths. Their scientific content and fingerprints should be preserved when refreshing documentation.

The 128 cases comprise 39 obstacle layouts. The additional catalog regression case is outside the benchmark denominator. No independent held-out layout set is available, and the adaptive-versus-baseline comparison does not separate point selection from repeated solving.

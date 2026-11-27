# Frozen Hybrid A* seeds

This directory is the official v2 seed source for `--source seeded`.

The search used inflation 1.0 m (5 grid cells), a 3.95 m radius that matches the vehicle `Rmin`, a 2.7 m wheelbase, a 90 s limit per case, a 200,000 expansion cap, a rectangular body, and a 5 cm frontend clearance policy. The search was not an unlimited-time rerun.

Summary of the 128-case search:

- 112 cases returned a path (`PATH`)
- 14 ended with `SEARCH_TIMEOUT`
- 2 ended with `SEARCH_EXHAUSTED`
- 104 of the 112 returned paths passed the contemporaneous 5 cm sampled-body check
- All 112 returned paths are used as optimization seeds, including the 8 that failed that sample check

The 16 cases without a path remain in the official 128-case denominator. 9 of those 16 are in the tight parallel family, which has 10 cases.

`实验结果说明.txt` is the original Chinese experiment note for this search, including a comparison with an earlier configuration that changed both inflation and turning radius at once. That comparison cannot isolate the two parameter changes, and it is not a new official backend score. This English file is an index, not a replacement of that note.

`protocol.json`, `results.json`, `results.csv`, and `summary.json` are the machine-readable records. `returned_path_failures.json` lists returned paths that failed the sample check.

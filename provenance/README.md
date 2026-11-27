# Provenance notes

This directory keeps historical material that shipped with the v2 package.

- `README_original.md` is an earlier Chinese delivery note. Some scores, entry names, and environment statements in that file predate the official v2 experiment. Do not treat it as the current result record. Use the repository `README.md` and `RESULTS.md` instead. The file mentions a historical `/workspace/traj_py` path as a development label.
- `legacy_tests/` holds older unit tests retained for reference. They are not the v2 acceptance test list. Run `python -m pytest -q` from the repository root for the current tests.

The official Hybrid A* seeds and their own notes live in `inputs/ha_r395/`, not here.

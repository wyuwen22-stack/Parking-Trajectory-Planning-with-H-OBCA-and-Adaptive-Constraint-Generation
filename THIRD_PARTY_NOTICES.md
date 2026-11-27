# Third-party notices

This file records third-party code and data that appear in the repository, and method sources that the project follows. It does not grant a license beyond the terms of each original work.

The top-level `LICENSE` file is an MIT license for Yuwen Wang's own source code and documentation. That MIT license does not cover official TPCAP records, the 20 adapted TPCAP scenes, or other third-party material listed below.

## PythonRobotics Reeds-Shepp geometry

`python_scripts/traj_py/reeds_shepp.py` adapts the geometry functions from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ReedsSheppPath/reeds_shepp_path_planning.py), retrieved 2026-11-03. Authors: Atsushi Sakai and Videh Patel. Plotting and demo imports were removed. Angle normalization is local. Diagnostic prints were removed. No new external package is required.

The MIT License (MIT)

Copyright (c) 2016 - now Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## H-OBCA and OBCA

The planner follows the Optimization-Based Collision Avoidance (OBCA) separation constraints of Zhang, Liniger, and Borrelli, and the Hierarchical OBCA (H-OBCA) warm-start pattern of Zhang, Liniger, Sakai, and Borrelli. The official H-OBCA repository is https://github.com/XiaojingGeorgeZhang/H-OBCA. A relevant upstream commit is `72503586429f81a4fd397edc5391db2f0308b8ed`. This package is not a copy of that Julia codebase. Citation details are in the References section of `README.md`.

## TPCAP scenes (not under MIT)

Twenty cases are adapted from the Trajectory Planning Competition of Automated Parking (TPCAP) preliminary set. The competition description is in Li et al., IEEE Transactions on Intelligent Vehicles, 8(1):16-21, 2023, https://doi.org/10.1109/TIV.2022.3228963. The official records and scene source are also published at https://github.com/libai1943/TPCAP_benchmarks.

`inputs/tpcap_preliminary_20.official.json` stores official preliminary records used as a source for the adapted scenes. The 20 adapted cases live in the scene catalog and related input files under `inputs/`. Official TPCAP records and those 20 adapted scenes are not licensed under the MIT license in `LICENSE`. This notice does not grant permission to republish that material under MIT terms. Confirm redistribution rights with the TPCAP organizers or the source repository before a public upload. The address `competition_itsc@163.com` appears inside the official JSON as a source contact, not as a contact for this repository.

The adapted cases use this project's 4.7 m by 2.0 m vehicle. They are not official protocol reproductions, and they must not be compared with official TPCAP leaderboard scores.

## Runtime dependencies

Planning and tests use NumPy, SciPy, CasADi, IPOPT (via CasADi), Matplotlib, and pytest. Those packages are declared in `requirements.txt` and are not vendored here. Each dependency keeps its own license.

## Historical fingerprint labels

`readonly_inputs_sha256.json` uses keys from an older development tree (`/workspace/traj_py/...`). The keys are fingerprint labels used by tests. They are not paths that the program opens at runtime.

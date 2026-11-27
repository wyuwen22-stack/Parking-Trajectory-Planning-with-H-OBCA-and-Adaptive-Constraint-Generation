#!/usr/bin/env python3
"""Strict 0.05 m recount from cases.csv and optional path NPZ files.

Main success predicate for delivery:
  no overlap, min_node_clearance >= DMIN - MACHINE_TOL,
  min_swept_clearance >= DMIN - MACHINE_TOL,
  with DMIN = 0.05 and MACHINE_TOL = 1e-12 (float64 noise only).

The suite that produced the CSV used DMIN - 1e-3 = 0.049 as its live flag.
That flag is retained only as a contrast column and must not be reported
as meeting the 5 cm requirement.

Heuristic pre-check rejections (goal_component_escapes) are labeled
HEURISTIC_PRECHECK: formal Hybrid A* search was skipped. They stay in
the denominator and are not "proven infeasible".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DMIN = 0.05
MACHINE_TOL = 1e-12
STRICT_LIMIT = DMIN - MACHINE_TOL  # 0.05 minus machine-eps only
LEGACY_LIMIT = DMIN - 1e-3         # 0.049, historical suite flag only
TRAP_RE = re.compile(r"kinematic trap|goal kinematic trap", re.I)


def _f(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def is_precheck(row: dict) -> bool:
    err = row.get("error") or ""
    return bool(TRAP_RE.search(err))


def strict_clearance_ok(node: float, swept: float) -> bool:
    return (
        math.isfinite(node)
        and math.isfinite(swept)
        and node >= STRICT_LIMIT
        and swept >= STRICT_LIMIT
    )


def classify(row: dict) -> str:
    """Primary outcome label for the strict 0.05 delivery tables."""
    if is_precheck(row):
        return "HEURISTIC_PRECHECK"
    if strict_clearance_ok(_f(row.get("min_node_clearance")), _f(row.get("min_swept_clearance"))):
        # A path was produced and meets strict 0.05. Other fail codes (TERM/CONS)
        # still matter for "suite success", but clearance is the 5 cm question.
        fail = row.get("fail_code") or ""
        if fail in ("OPT", "CONS", "TERM"):
            return f"STRICT_CLEAR_BUT_{fail}"
        if fail in ("WS",):
            return "HEURISTIC_PRECHECK" if is_precheck(row) else "WS"
        return "STRICT_OK"
    fail = row.get("fail_code") or ""
    node, swept = _f(row.get("min_node_clearance")), _f(row.get("min_swept_clearance"))
    if fail == "WS" or (not math.isfinite(node) and not math.isfinite(swept) and fail in ("", "WS")):
        return "WS"
    if fail == "OPT":
        return "OPT"
    if fail == "CONS":
        return "CONS"
    if fail == "TERM":
        return "TERM"
    if math.isfinite(swept) or math.isfinite(node):
        # legacy 0.049 flag may have been OK; still COLL under strict 0.05
        return "COLL_STRICT"
    return fail or "NO_PATH"


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            raw = dict(raw)
            raw["_node"] = _f(raw.get("min_node_clearance"))
            raw["_swept"] = _f(raw.get("min_swept_clearance"))
            raw["_legacy_ok"] = raw.get("success", "") in ("1", "True", "true")
            raw["_strict_ok"] = strict_clearance_ok(raw["_node"], raw["_swept"]) and not is_precheck(raw)
            # If another fail code exists besides clearance, strict_ok for the
            # 5 cm table is clearance-only; pipeline success is stricter.
            raw["_label"] = classify(raw)
            raw["_precheck"] = is_precheck(raw)
            rows.append(raw)
    return rows


def group_table(rows: list[dict]) -> dict:
    out = {}
    families = []
    for r in rows:
        f = r.get("family") or "standard"
        if f not in families:
            families.append(f)
    for family in families + ["ALL"]:
        subset = rows if family == "ALL" else [r for r in rows if (r.get("family") or "standard") == family]
        out[family] = {}
        for method in ("hybrid_astar", "h_obca"):
            mrows = [r for r in subset if r["method"] == method]
            labels = Counter(r["_label"] for r in mrows)
            out[family][method] = {
                "n": len(mrows),
                "strict_ok": sum(1 for r in mrows if r["_label"] == "STRICT_OK"),
                "legacy_ok": sum(1 for r in mrows if r["_legacy_ok"]),
                "precheck": sum(1 for r in mrows if r["_precheck"]),
                "labels": dict(labels),
            }
    return out


def failure_rows(rows: list[dict]) -> list[dict]:
    fails = [r for r in rows if r["_label"] != "STRICT_OK"]
    fails.sort(key=lambda r: (r.get("family") or "", r["method"], r.get("maneuver") or "", r.get("tightness") or "", r.get("start_x") or ""))
    return fails


def write_outputs(rows: list[dict], table: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # annotated CSV
    if rows:
        fields = list(rows[0].keys())
        # keep helper fields
        with (dest / "cases_strict_005.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                out = dict(r)
                for k, v in list(out.items()):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        out[k] = ""
                    elif isinstance(v, bool):
                        out[k] = int(v)
                writer.writerow(out)
    (dest / "strict_summary.json").write_text(
        json.dumps(
            {
                "predicate": {
                    "DMIN_m": DMIN,
                    "machine_tol": MACHINE_TOL,
                    "strict_limit_m": STRICT_LIMIT,
                    "legacy_limit_m": LEGACY_LIMIT,
                    "legacy_note": "0.049 was the live suite flag; not a 5 cm pass.",
                    "precheck_note": (
                        "HEURISTIC_PRECHECK: goal_component_escapes skipped formal "
                        "Hybrid A* search. Not a proof of infeasibility. Stays in the denominator."
                    ),
                },
                "n_rows": len(rows),
                "table": table,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fails = failure_rows(rows)
    with (dest / "failures_strict_005.csv").open("w", newline="") as handle:
        fields = [
            "label", "method", "family", "scene_name", "maneuver", "tightness",
            "start_x", "start_y", "start_psi", "legacy_success", "fail_code",
            "min_node_clearance", "min_swept_clearance", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in fails:
            err = ((r.get("error") or "").splitlines() or [""])[0][:200]
            writer.writerow({
                "label": r["_label"],
                "method": r["method"],
                "family": r.get("family") or "",
                "scene_name": r.get("scene_name") or "",
                "maneuver": r.get("maneuver") or "",
                "tightness": r.get("tightness") or "",
                "start_x": r.get("start_x") or "",
                "start_y": r.get("start_y") or "",
                "start_psi": r.get("start_psi") or "",
                "legacy_success": int(r["_legacy_ok"]),
                "fail_code": r.get("fail_code") or "",
                "min_node_clearance": r.get("min_node_clearance") or "",
                "min_swept_clearance": r.get("min_swept_clearance") or "",
                "error": err,
            })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_rows(args.csv)
    table = group_table(rows)
    write_outputs(rows, table, args.out)
    print(f"rows={len(rows)}  wrote {args.out}")
    for family, block in table.items():
        for method, b in block.items():
            print(
                f"  {family:10} {method:13} n={b['n']:3d}  "
                f"strict {b['strict_ok']}/{b['n']}  "
                f"legacy_flag {b['legacy_ok']}/{b['n']}  "
                f"precheck {b['precheck']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit planning entry. Viewing results is a separate view_results.py entry."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true', help='Run/resume all 128 cases')
    selection.add_argument('--case', help='Run/resume one exact case ID')
    parser.add_argument('--source', choices=['seeded', 'fresh'], default='seeded',
                        help='seeded reuses bundled R=3.95 HA; fresh explicitly repeats search')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    os.environ['PARKING_SAFE_OUTPUT'] = str(args.output.resolve())
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    sys.path.insert(0, str(ROOT / 'python_scripts'))
    if args.source == 'seeded':
        from parking_config import read_config
        cfg = read_config()
        frozen = json.loads((ROOT / 'inputs' / 'ha_r395' / 'protocol.json').read_text(encoding='utf-8'))
        if (any(cfg['vehicle'][k] != frozen['vehicle'][k] for k in ('L', 'Rmin', 'ego'))
                or cfg['planner']['inflation_m'] != frozen['settings']['inflation']):
            parser.error('Current parameters differ from bundled HA. Use --source fresh and a new --output directory.')
    from safe_parking import suite
    if args.all:
        suite.run(args.source, args.workers)
    else:
        suite.get_spec(args.case)  # reject misspelled IDs before writing outputs
        suite.prepare()
        suite.case(args.case, args.source)
        suite.summarize(args.source)


if __name__ == '__main__':
    main()

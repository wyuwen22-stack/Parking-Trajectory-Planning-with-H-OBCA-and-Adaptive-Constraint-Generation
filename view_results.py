"""Browse saved planning results. This entry never starts planning."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results' / 'seeded')
    parser.add_argument('--case', help='Exact saved case ID (optional)')
    parser.add_argument('--round', type=int, dest='adaptive_round')
    parser.add_argument('--mode', choices=['rollout', 'saved'], default='rollout')
    parser.add_argument('--list', action='store_true', help='List saved cases without opening a window')
    parser.add_argument('--export', type=Path, help='Save one comparison figure without opening a window')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'python_scripts'))
    from safe_parking.view_data import case_ids, load_case
    if not args.results.is_dir():
        parser.error(f'No saved results directory: {args.results}. No planning was started.')
    identifiers = case_ids(args.results)
    if not identifiers:
        parser.error('No saved cases found. Generate results explicitly with run_planning.py.')
    if args.list:
        print('\n'.join(identifiers))
        return
    from safe_parking.view_app import default_case
    identifier = args.case or default_case(args.results, identifiers)
    if identifier not in identifiers:
        parser.error('Unknown saved case ID. Use --list to see available cases.')
    if args.export:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from safe_parking.view_plot import ComparisonPlot
        case = load_case(args.results, identifier, args.adaptive_round, args.mode)
        plot = ComparisonPlot()
        FigureCanvasAgg(plot.figure)
        plot.render(case)
        print(plot.save(args.export).resolve())
    else:
        from safe_parking.view_app import launch
        launch(args.results, identifier, args.adaptive_round, args.mode)


if __name__ == '__main__':
    main()

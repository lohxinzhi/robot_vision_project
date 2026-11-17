"""Task 7 hyperparameter search and final fitting using the configured single seed."""
import argparse
import json
from pathlib import Path

import torch

# These names remain importable for existing project scripts.
from training import evaluate, fit, predict, read_data, save_json
from search import Study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--data', type=Path, default=here/'dataset_2026')
    parser.add_argument('--output', '--results', type=Path, default=here/'results')
    parser.add_argument('--config', type=Path, default=here/'search_space.json')
    parser.add_argument('--phase', choices=('all','search','confirm','final'), default='all')
    parser.add_argument('--group', help='Run one named search group from the config file.')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    plan = json.loads(args.config.read_text())
    if args.group and args.group not in {g['name'] for g in plan['groups']}:
        parser.error('group must match a name in the search config')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    study = Study(args.data,args.output,plan,args.epochs)
    if args.phase in ('all','search'):
        study.search(args.group)
    if args.phase in ('all','confirm'):
        study.confirm()
    if args.phase in ('all','final'):
        study.finalize()
        from summarize_results import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()

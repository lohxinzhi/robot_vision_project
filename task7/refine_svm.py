"""Register a single SVM parameter trial in the shared experiment registry."""
import argparse
import json
from pathlib import Path
import torch
from search import Study


def run_refinement(data_root, results, name, regularization, cell_size, reason):
    plan = json.loads(Path(__file__).with_name('search_space.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    study = Study(data_root, results, plan)
    config = {**study.best('svm')['config'], 'C': regularization,
              'feature': 'hog', 'hog_cell_size': cell_size}
    record = study.run_trial('svm', config, plan['seed'])
    study.state.setdefault('notes', {})[record['id']] = {'name': name, 'reason': reason}
    study.save()
    study.export_tables()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--data', type=Path, default=here/'dataset_2026')
    parser.add_argument('--results', type=Path, default=here/'results')
    parser.add_argument('--name', required=True)
    parser.add_argument('--C', type=float, default=100)
    parser.add_argument('--cell-size', type=int, choices=(4, 8), default=4)
    parser.add_argument('--reason', default='Individual parameter comparison')
    args = parser.parse_args()
    if args.C <= 0:
        parser.error('C must be positive')
    run_refinement(args.data,args.results,args.name,args.C,args.cell_size,args.reason)

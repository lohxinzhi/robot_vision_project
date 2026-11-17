"""Export saved figures and generate report tables from recorded experiments."""
import base64
import json
from pathlib import Path
import shutil

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parent
FIGURES = REPORT / 'figures'
TABLES = REPORT / 'tables'
RESULTS = REPO / 'task7/results'
CELLS = {
    6: 'original', 15: 'contrast', 19: 'average', 23: 'square_median',
    25: 'cross_median', 29: 'combined_stretch', 30: 'combined_threshold',
    32: 'threshold_median', 36: 'crop', 37: 'task4_binary',
    38: 'task4_median', 40: 'task5_outline', 42: 'task6_binary',
    46: 'task6_boxes',
}


def write_table(name, header, rows, alignment):
    lines = [r'\begin{center}\begin{tabular}{' + alignment + '}', r'\toprule',
             ' & '.join(header) + r' \\\midrule']
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    lines.append(r'\bottomrule\end{tabular}\end{center}')
    (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def export_tables():
    state = json.loads((RESULTS / 'study.json').read_text())
    metrics = json.loads((RESULTS / 'classification_results.json').read_text())
    records = list(state['records'].values())

    def group_records(name):
        group = state['groups'][name]
        return [next(record for record in records
                     if record['method'] == group['method'] and record['seed'] == state['seed']
                     and record['config'] == candidate)
                for candidate in group['candidates']]

    def accuracy(record):
        value = record['validation_accuracy']
        return f'{100 * value:.2f}\\%' if value is not None else 'Failed'

    rows = []
    for record in group_records('cnn_structure'):
        config = record['config']
        rows.append([config['depth'], config['pool_size'], record['fc_input'],
                     f"{config['dropout']:g}", f"{config['weight_decay']:g}", accuracy(record)])
    write_table('cnn_structure', ['Depth', 'Pool', 'FC input', 'Dropout', 'Weight decay', 'Validation'], rows, 'rrrrrr')

    rows = []
    for record in group_records('cnn_width'):
        rows.append([record['config']['hidden_size'], record['parameters'], record['best_epoch'], accuracy(record)])
    write_table('cnn_width', ['Hidden width', 'Parameters', 'Best epoch', 'Validation'], rows, 'rrrr')
    rows = [[record['config']['mode'], record['best_epoch'], accuracy(record)]
            for record in group_records('cnn_preprocessing')]
    write_table('cnn_preprocessing', ['Mode', 'Best epoch', 'Validation'], rows, 'lrr')
    rows = []
    for record in group_records('cnn_regularization'):
        config = record['config']
        rows.append([f"{config['dropout']:g}", f"{config['weight_decay']:g}",
                     record['best_epoch'], accuracy(record)])
    write_table('cnn_regularization', ['Dropout', 'Weight decay', 'Best epoch', 'Validation'], rows, 'rrrr')

    for method in ('cnn', 'svm'):
        rows = []
        for record in group_records(f'{method}_optimizers'):
            config = record['config']
            duration = record['fit_seconds']
            rows.append([config['optimizer'].upper(), f"{config['lr']:g}",
                         record['best_epoch'] or '--', f'{duration:.3f}' if duration is not None else '--',
                         accuracy(record)])
        write_table(f'{method}_optimizers', ['Optimizer', 'Learning rate', 'Best epoch', 'Fit (s)', 'Validation'], rows, 'lrrrr')

    rows = []
    for record in group_records('svm_features'):
        config = record['config']
        rows.append([config['mode'], config['feature'].upper(), record['best_epoch'], accuracy(record)])
    write_table('svm_features', ['Mode', 'Features', 'Best epoch', 'Validation'], rows, 'llrr')
    rows = []
    for record in group_records('svm_hyperparameters'):
        config = record['config']
        if config['mode'] == 'stretch' and config['hog_cell_size'] == 4:
            rows.append([config['loss'].replace('_', r'\_'), f"{config['C']:g}",
                         record['best_epoch'], accuracy(record)])
    write_table('svm_loss', ['Margin loss', r'$C$', 'Best epoch', 'Validation'], rows, 'lrrr')

    rows = []
    for method, selected in state['selection']['models'].items():
        rows.append([method.upper(), state['seed'], f"{100 * selected['validation_accuracy']:.2f}\\%",
                     selected['refit_epochs']])
    write_table('selection', ['Model', 'Seed', 'Validation accuracy', 'Refit epochs'], rows, 'lrrr')

    rows = []
    for method, result in metrics.items():
        matrix = result['confusion_matrix']
        total = sum(map(sum, matrix))
        correct = sum(matrix[index][index] for index in range(len(matrix)))
        rows.append([method.upper(), f'{correct} / {total}', total - correct,
                     f"{100 * result['test_accuracy']:.2f}\\%", f"{result['macro_f1']:.5f}"])
    write_table('performance', ['Model', 'Correct / total', 'Errors', 'Accuracy', 'Macro F1'], rows, 'llrrr')
    rows = []
    for label in metrics['cnn']['per_class']:
        values = [f"{metrics[method]['per_class'][label][key]:.4f}"
                  for method in ('cnn', 'svm') for key in ('precision', 'recall')]
        rows.append([label, *values])
    write_table('per_class', ['Class', 'CNN precision', 'CNN recall', 'SVM precision', 'SVM recall'], rows, 'lrrrr')
    rows = [
        ['Learned parameters', *[metrics[method]['parameters'] for method in ('cnn', 'svm')]],
        ['Checkpoint bytes', *[metrics[method]['model_bytes'] for method in ('cnn', 'svm')]],
        ['Development refit (s)', *[f"{metrics[method]['refit_seconds']:.3f}" for method in ('cnn', 'svm')]],
        ['Model time (ms/image)', *[f"{metrics[method]['model_ms_per_image']:.5f}" for method in ('cnn', 'svm')]],
        ['HOG and model (ms/image)', '--', f"{metrics['svm']['feature_and_model_ms_per_image']:.5f}"],
    ]
    write_table('efficiency', ['Measurement', 'CNN', 'SVM'], rows, 'lrr')
    for source, target in [('experiments', 'all_experiments'), ('chip_preprocessing_experiments', 'chip_preprocessing')]:
        shutil.copy2(RESULTS / f'{source}.tex', TABLES / f'{target}.tex')


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    notebook = json.loads((REPO / 'task1-6/tasks.ipynb').read_text())
    for index, name in CELLS.items():
        outputs = notebook['cells'][index].get('outputs', [])
        images = [item['data']['image/png'] for item in outputs
                  if 'image/png' in item.get('data', {})]
        if len(images) != 1:
            raise ValueError(f'Expected one saved figure in notebook cell {index}')
        encoded = images[0]
        if isinstance(encoded, list):
            encoded = ''.join(encoded)
        (FIGURES / f'{name}.png').write_bytes(base64.b64decode(encoded))
    task7_figures = ['preprocessing_examples', 'cnn_learning_curves', 'cnn_confusion', 'svm_confusion',
                     'cnn_optimizer_comparison',
                     'svm_optimizer_comparison', 'chip_preprocessing_comparison']
    for name in task7_figures:
        shutil.copy2(RESULTS / f'{name}.png', FIGURES / f'{name}.png')
    export_tables()
    print(f'Exported {len(CELLS) + len(task7_figures)} figures and {len(list(TABLES.glob("*.tex")))} tables')


if __name__ == '__main__':
    main()

"""Generate the unified Task 7 result tables and comparison figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def label(record):
    c = record['config']
    if record['method'] == 'cnn':
        return f"{c['optimizer']}, lr={c['lr']:g}, depth={c['depth']}, hidden={c['hidden_size']}"
    return f"{c['optimizer']}, lr={c['lr']:g}, {c['loss']}, C={c['C']:g}"


def summarize(output):
    state = json.loads((output/'study.json').read_text())
    records = list(state['records'].values())
    seed = state.get('seed', 5411)
    lines = ['# Task 7 experiments', '',
             'One dataset manifest is shared by all trials. Inner training uses 4,270 images, validation 1,064, and testing 1,778.',
             f'Each trial is initialized independently using seed {seed}. Validation accuracy selects the configuration and epoch before the final development refit.', '',
             '| Method | Optimizer | Mode | Structure / feature | Regularization / loss | LR | Batch | Seed | Best epoch | Validation |',
             '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    fields = ['id','method','optimizer','mode','seed','status','best_epoch','validation_accuracy','parameters','fit_seconds','config']
    with (output/'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fields,lineterminator='\n')
        writer.writeheader()
        for r in records:
            c = r['config']
            structure = (f"depth {c['depth']}, pool {c['pool_size']}, FC {r['fc_input']}→{c['hidden_size']}→7"
                         if r['method']=='cnn' else f"{c['feature']}, cell {c['hog_cell_size']}")
            regularization = (f"dropout {c['dropout']:g}, WD {c['weight_decay']:g}"
                              if r['method']=='cnn' else f"{c['loss']}, C={c['C']:g}")
            score = f"{r['validation_accuracy']:.2%}" if r.get('validation_accuracy') is not None else 'Numerical failure'
            lines.append(f"| {r['method'].upper()} | {c['optimizer']} | {c['mode']} | {structure} | {regularization} | "
                         f"{c['lr']:g} | {c['batch_size'] or 'full'} | {r['seed']} | {r['best_epoch']} | {score} |")
            writer.writerow({**{key:r.get(key) for key in fields},'optimizer':c['optimizer'],
                             'mode':c['mode'],'config':json.dumps(c,sort_keys=True)})
    lines += ['', 'LBFGS uses full-batch updates. One recorded outer epoch allows up to ten quasi-Newton iterations with a strong-Wolfe line search. Adam, AdamW and SGD use the recorded minibatch size. Runtime accompanies validation results.', '']
    (output/'all_experiment_tables.md').write_text('\n'.join(lines).rstrip()+'\n')
    experiment_tex = [r'\section{Hyperparameter comparisons}',
                      r'\begin{longtable}{@{}p{0.16\linewidth}p{0.43\linewidth}rrr@{}}',
                      r'\toprule Model / optimizer & Configuration & Seed & Epoch & Validation \\\midrule',
                      r'\endhead']
    for record in records:
        config = record['config']
        if record['method'] == 'cnn':
            setting = (f"depth {config['depth']}, pool {config['pool_size']}, hidden {config['hidden_size']}, "
                       f"dropout {config['dropout']:g}, WD {config['weight_decay']:g}")
        else:
            setting = (f"{config['feature']}, cell {config['hog_cell_size']}, "
                       f"{config['loss']}, C={config['C']:g}")
        setting += f", {config['mode']}, LR {config['lr']:g}, batch {config['batch_size'] or 'full'}"
        setting = setting.replace('_', r'\_')
        score = f"{100*record['validation_accuracy']:.2f}\\%" if record.get('validation_accuracy') is not None else 'Failed'
        experiment_tex.append(f"{record['method'].upper()} / {config['optimizer']} & {setting} & "
                              f"{record['seed']} & {record['best_epoch'] or '--'} & {score}" + r' \\')
    experiment_tex += [r'\bottomrule\end{longtable}']
    (output/'experiments.tex').write_text('\n'.join(experiment_tex)+'\n')
    for method in ('cnn','svm'):
        candidates = [r for r in records if r['method']==method and r['seed']==seed
                      and r.get('validation_accuracy') is not None]
        optimizers = sorted({r['config']['optimizer'] for r in candidates})
        best = [max([r for r in candidates if r['config']['optimizer']==optimizer],
                    key=lambda r:r['validation_accuracy']) for optimizer in optimizers]
        fig,ax=plt.subplots(figsize=(9,4))
        ax.bar(optimizers,[r['validation_accuracy']*100 for r in best],color='#377c99')
        ax.set(ylabel='Best validation accuracy (%)',title=method.upper()+' optimizer comparison',ylim=(0,101))
        for i, record in enumerate(best):
            ax.text(i, record['validation_accuracy']*100 + .5,
                    f"{record['validation_accuracy']:.2%}", ha='center')
        fig.tight_layout()
        fig.savefig(output/f'{method}_optimizer_comparison.png', dpi=160)
        plt.close(fig)
    if 'selection' in state:
        selected_lines=['# Single-seed validation selection', '',
                        '| Method | Configuration | Seed | Validation | Refit epochs |',
                        '|---|---|---:|---:|---:|']
        for method, item in state['selection']['models'].items():
            selected_lines.append(f"| {method.upper()} | {label(state['records'][item['trial_id']])} | "
                                  f"{seed} | {item['validation_accuracy']:.2%} | {item['refit_epochs']} |")
        (output/'validation_selection.md').write_text('\n'.join(selected_lines)+'\n')
    if not (output/'selection.json').exists():
        return
    selection=json.loads((output/'selection.json').read_text())
    results=json.loads((output/'classification_results.json').read_text())
    chip=json.loads((output/'chip_results.json').read_text())
    parameters = {**state.get('runtime', {}), 'seed': selection['seed'],
                  'trial_records': len(records),
                  'selected_configurations': {method: result['config'] for method, result in results.items()}}
    (output/'training_parameters.json').write_text(json.dumps(parameters, indent=2)+'\n')
    summary=['# Task 7 selected models','',f"Selection uses validation accuracy with seed {selection['seed']}. Ties favor fewer parameters and then an earlier best epoch.",'',
             '| Method | Validation accuracy | Test accuracy | Macro F1 | Errors | Parameters | Refit seconds | Model ms/image |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    latex=[r'\section{Task 7 model comparison}',r'\begin{center}\begin{tabular}{lrrr}\toprule',r'Method & Validation accuracy & Test accuracy & Macro F1 \\\midrule']
    for method,r in results.items():
        matrix = r['confusion_matrix']
        errors = sum(map(sum, matrix)) - sum(matrix[i][i] for i in range(7))
        summary.append(f"| {method.upper()} | {r['validation_accuracy']:.2%} | {r['test_accuracy']:.2%} | {r['macro_f1']:.5f} | "
                       f"{errors} | {r['parameters']} | {r['refit_seconds']:.3f} | {r['model_ms_per_image']:.5f} |")
        latex.append(f"{method.upper()} & {100*r['validation_accuracy']:.2f}\\% & {100*r['test_accuracy']:.2f}\\% & {r['macro_f1']:.5f}"+r' \\')
    summary += ['',f"SVM feature-plus-model inference is {results['svm']['feature_and_model_ms_per_image']:.5f} ms/image. Model-only timings exclude file decoding, input normalization and segmentation. SVM model timing also excludes HOG extraction. Timings use CPU four threads, one warm-up and the median of five passes.",'',
                '## Selected parameters','']
    for method, result in results.items():
        summary += [f'### {method.upper()}', '', '```json',
                    json.dumps(result['config'], indent=2), '```', '']
    summary += ['## Chip recognition','','| Connectivity | Method | Prediction | Position accuracy |','|---|---|---|---:|']
    for connectivity,item in chip['connectivity'].items():
        for method,prediction in item['models'].items():
            score=prediction['position_accuracy']
            value=f'{score:.2%}' if score is not None else 'Unaligned'
            summary.append(f"| {connectivity} | {method.upper()} | `{prediction['prediction']}` | {value} |")
    if chip.get('image_processing'):
        summary += ['', 'Chip image-processing parameters were selected using the chip development image. '
                    'Read [the image-processing comparisons](chip_preprocessing_experiments.md) for the baseline, '
                    'bounded-refinement experiments and fixed-checkpoint evidence.', '', '```json',
                    json.dumps(chip['image_processing'], indent=2), '```']
    summary += ['', 'The shared test partition and chip image had already been inspected during project development. Parameter selection in this search uses validation data. The final scores describe this dataset split and this chip example.',
                '', 'All trials are in [all_experiment_tables.md](all_experiment_tables.md) and `experiments.csv`. Selected configurations are in [validation_selection.md](validation_selection.md).', '']
    latex += [r'\bottomrule\end{tabular}\end{center}',
              'Each selected configuration was refitted on the complete development partition. '
              f"All experiments use seed {selection['seed']}. "
              'The test partition and chip example were previously observed during project development.']
    latex += [r'\subsection{Chip development image}',
              r'\begin{center}\begin{tabular}{lllr}\toprule',
              r'Connectivity & Method & Prediction & Accuracy (\%) \\\midrule']
    for connectivity, item in chip['connectivity'].items():
        for method, prediction in item['models'].items():
            score = prediction['position_accuracy']
            value = f'{100 * score:.2f}' if score is not None else '--'
            latex.append(f"{connectivity} & {method.upper()} & {prediction['prediction']} & {value}" + r' \\')
    latex += [r'\bottomrule\end{tabular}\end{center}',
              'Chip filtering and extraction parameters were adjusted using this image. '
              'The reported chip accuracy covers its ten characters.']
    (output/'task7_results.md').write_text('\n'.join(summary))
    (output/'task7_results.tex').write_text('\n'.join(latex)+'\n')
    chosen=selection['models']['cnn']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    r=state['records'][chosen['trial_id']]
    axes[0].plot([h['epoch'] for h in r['history']],[h['loss'] for h in r['history']],label=str(r['seed']))
    axes[1].plot([h['epoch'] for h in r['history']],[h['validation_accuracy'] for h in r['history']],label=str(r['seed']))
    axes[0].set(xlabel='Epoch',ylabel='Training cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
    for ax in axes:
        ax.legend(title='Seed')
    fig.tight_layout()
    fig.savefig(output/'cnn_learning_curves.png', dpi=160)
    plt.close(fig)
    default_output = Path(__file__).with_name('results')
    if output.resolve() == default_output.resolve():
        readme = Path(__file__).with_name('README.md')
        if readme.exists():
            block = [f"The registry contains **{len(records)} trial records**, including "
                     f"{sum(r.get('validation_accuracy') is None for r in records)} numerically failed trial(s).",
                     '', '| Selected model | Validation accuracy | Test accuracy | Macro F1 | Parameters |',
                     '|---|---:|---:|---:|---:|']
            for method, result in results.items():
                block.append(f"| {method.upper()} | {result['validation_accuracy']:.2%} | "
                             f"{result['test_accuracy']:.2%} | {result['macro_f1']:.5f} | {result['parameters']:,} |")
            block += ['', '| Parameter | CNN | SVM |', '|---|---|---|']
            cnn, svm = results['cnn']['config'], results['svm']['config']
            channels = 32 if cnn['depth'] == 2 else 64
            input_dim = channels * cnn['pool_size'] ** 2
            block += [f"| Representation | {cnn['depth']} conv layers, FC {input_dim}→{cnn['hidden_size']}→7 | HOG cell {svm['hog_cell_size']}×{svm['hog_cell_size']} |",
                      f"| Optimizer | {cnn['optimizer']} | {svm['optimizer']} |",
                      f"| Learning rate | {cnn['lr']} | {svm['lr']} |",
                      f"| Preprocessing | {cnn['mode']} | {svm['mode']} |",
                      f"| Regularization / loss | Dropout {cnn['dropout']}, WD {cnn['weight_decay']} | {svm['loss']}, C={svm['C']:g} |",
                      f"| Refit epochs | {cnn['epochs']} | {svm['epochs']} |",
                      f"| Batch size | {cnn['batch_size']} | {svm['batch_size'] or 'full'} |", '',
                      '| Connectivity | CNN prediction | SVM prediction |', '|---|---|---|']
            for connectivity, item in chip['connectivity'].items():
                cells = []
                for method in ('cnn', 'svm'):
                    prediction = item['models'][method]
                    accuracy = prediction['position_accuracy']
                    score = f'{accuracy:.2%}' if accuracy is not None else 'Unaligned'
                    cells.append(f"`{prediction['prediction']}` ({score})")
                block.append(f"| {connectivity} | {cells[0]} | {cells[1]} |")
            block += ['', 'The chip reference is `HD44780A00`. Position scores describe this development image.', '',
                      'Read the [full experiment tables](results/all_experiment_tables.md), '
                      '[single-seed selection](results/validation_selection.md) and '
                      '[final measurements](results/task7_results.md).']
            if chip.get('image_processing'):
                block += ['', 'Image processing uses mild filtering to locate characters and bounded grayscale-median refinement '
                          'for recognition. Read the [image-processing experiments](results/chip_preprocessing_experiments.md). '
                          'Those choices used feedback from this chip image with the selected classifiers held fixed.']
            text = readme.read_text()
            before, remainder = text.split('<!-- results:start -->', 1)
            _, after = remainder.split('<!-- results:end -->', 1)
            readme.write_text(before + '<!-- results:start -->\n' + '\n'.join(block) +
                              '\n<!-- results:end -->' + after)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results'))
    summarize(parser.parse_args().output)

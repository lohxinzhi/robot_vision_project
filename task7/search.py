"""Single-seed experiment search, validation selection and final evaluation."""
import contextlib
import csv
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from models import hog_features
from prepare_data import prepare_data
from recognize_chip import recognize
from training import evaluate, fit, predict, read_data, save_json


def normalize_config(method, values, seed=5411, cnn_epochs=20):
    common = {'mode': 'pad', 'optimizer': 'adam', 'batch_size': 128, 'shuffle_seed': seed}
    defaults = ({'lr': .003, 'epochs': cnn_epochs, 'depth': 3, 'pool_size': 4,
                 'hidden_size': 64, 'dropout': .25, 'weight_decay': 1e-4}
                if method == 'cnn' else
                {'lr': .01, 'epochs': 35, 'C': 10., 'feature': 'hog',
                 'hog_cell_size': 8, 'loss': 'squared_hinge'})
    config = {**common, **defaults, **values, 'shuffle_seed': seed}
    if config['optimizer'] == 'sgd':
        config['momentum'] = config.get('momentum', .9)
    else:
        config.pop('momentum', None)
    if config['optimizer'] == 'lbfgs':
        config['batch_size'] = 0
    elif config['batch_size'] == 0:
        config['batch_size'] = 128
    for name in ('lr', 'dropout', 'weight_decay', 'C', 'momentum'):
        if name in config:
            config[name] = float(config[name])
    return config


def ranking(record):
    return record['validation_accuracy'], -record['parameters'], -record['best_epoch']


class Study:
    def __init__(self, data, output, plan, epochs=20):
        self.data, self.output, self.plan, self.epochs = data, output, plan, epochs
        output.mkdir(parents=True, exist_ok=True)
        if not (output / 'data_split.csv').exists():
            prepare_data(data, output, plan['seed'])
        self.path = output / 'study.json'
        self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                      {'records': {}, 'groups': {}})
        self.state['seed'] = plan['seed']
        self.state['runtime'] = {'torch': str(torch.__version__), 'device': 'cpu', 'threads': 4}
        self.state['search_plan'] = plan
        self.images, self.labels, self.indices, self.classes = read_data(
            data, output, modes=('resize', 'pad', 'stretch'))
        self.cache = {}
        self.experiments = output / 'experiments'
        self.experiments.mkdir(exist_ok=True)

    def save(self):
        save_json(self.path, self.state)

    def features(self, method, config):
        key = (method, config['mode'], config.get('feature'), config.get('hog_cell_size'))
        if key not in self.cache:
            images = self.images[config['mode']]
            if method == 'cnn':
                array = images[:, None]
            elif config['feature'] == 'pixels':
                array = images.reshape(len(images), -1)
            else:
                array = hog_features(images, config['hog_cell_size'])
            self.cache[key] = torch.from_numpy(array)
        return self.cache[key]

    def best(self, method):
        records = [r for r in self.state['records'].values()
                   if r['method'] == method and r['seed'] == self.plan['seed']
                   and r.get('validation_accuracy') is not None]
        if not records:
            raise ValueError(f'Run the fixed {method} baseline groups before a best-based group.')
        return max(records, key=ranking)

    def run_trial(self, method, values, seed):
        config = normalize_config(method, values, seed, self.epochs)
        for record in self.state['records'].values():
            if record['method'] == method and record['seed'] == seed and record['config'] == config:
                return record
        key = f"{method}_{len(self.state['records']) + 1:04d}"
        print('Trial', key, config, flush=True)
        try:
            _, record = fit(method, self.features(method, config), self.labels,
                            self.indices['train'], self.indices['validation'],
                            config, len(self.classes), seed)
            record['status'] = 'completed'
        except FloatingPointError as error:
            record = {'config': config, 'status': 'failed', 'reason': str(error),
                      'validation_accuracy': None, 'parameters': None, 'fc_input': None,
                      'best_epoch': None, 'fit_seconds': None, 'history': []}
            print('Numerical failure', key, str(error), flush=True)
        record.update(id=key, method=method, seed=seed)
        directory = self.experiments / key
        directory.mkdir(exist_ok=True)
        save_json(directory / 'record.json', record)
        self.state['records'][key] = record
        if seed == self.plan['seed'] and record.get('validation_accuracy') is not None:
            self.state.pop('selection', None)
        self.save()
        return record

    def search(self, group_name=None):
        for group in self.plan['groups']:
            if group_name and group['name'] != group_name:
                continue
            name, method = group['name'], group['method']
            specification = {'group': group, 'epochs': self.epochs, 'seed': self.plan['seed']}
            existing = self.state['groups'].get(name)
            if existing and existing.get('specification') != specification:
                self.state['groups'].pop(name)
            if name not in self.state['groups']:
                base = self.best(method)['config'] if group['base'] == 'best' else {}
                base = {**base, **group.get('fixed', {})}
                candidates = []
                for grid in group['grids']:
                    for values in itertools.product(*grid.values()):
                        config = {**base, **dict(zip(grid, values))}
                        candidates.append(normalize_config(method, config, self.plan['seed'], self.epochs))
                self.state['groups'][name] = {'method': method, 'candidates': candidates,
                                              'specification': specification, 'completed': False}
                self.save()
            progress = self.state['groups'][name]
            if progress['completed']:
                print('Completed group', name, flush=True)
                continue
            for config in progress['candidates']:
                self.run_trial(method, config, self.plan['seed'])
            progress['completed'] = True
            progress['selected_id'] = self.best(method)['id']
            self.state.pop('selection', None)
            self.save()
            print('Group result', name, self.best(method)['validation_accuracy'], self.best(method)['config'], flush=True)
            self.export_tables()

    def confirm(self):
        selected = {}
        for method in ('cnn', 'svm'):
            record = self.best(method)
            selected[method] = {'trial_id': record['id'], 'config': record['config'],
                                'validation_accuracy': record['validation_accuracy'],
                                'parameters': record['parameters'], 'refit_epochs': record['best_epoch']}
            print('Selected', method, selected[method], flush=True)
        self.state['selection'] = {
            'seed': self.plan['seed'],
            'criterion': 'validation accuracy, fewer parameters, earliest best epoch',
            'models': selected,
        }
        self.save()
        self.export_tables()

    def finalize(self):
        if 'selection' not in self.state:
            raise ValueError('Run --phase confirm to select configurations before the final fit.')
        selection = self.state['selection']
        selected = selection['models']
        if (self.output / 'selection.json').exists():
            previous = json.loads((self.output / 'selection.json').read_text())
            if previous == selection and (self.output / 'classification_results.json').exists():
                print('Final models already match the selected configurations.', flush=True)
                return
        save_json(self.output / 'selection_frozen.json', selection)
        pending = self.output / '.finalizing'
        pending.mkdir(exist_ok=True)
        development = np.concatenate([self.indices['train'], self.indices['validation']])
        truth = self.labels[self.indices['test']].numpy()
        results = {}
        for method, choice in selected.items():
            config = {**choice['config'], 'epochs': choice['refit_epochs']}
            features = self.features(method, config)
            model, refit = fit(method, features, self.labels, development, None,
                               config, len(self.classes), self.plan['seed'], refit=True)
            prediction = predict(model, features[self.indices['test']])
            result = evaluate(truth, prediction, self.classes, pending / f'{method}_confusion.png')
            model_path = pending / f'{method}_model.pt'
            torch.save({'state_dict': model.state_dict(), 'config': config, 'classes': self.classes,
                        'method': method}, model_path)
            test_features = features[self.indices['test']]
            predict(model, test_features)
            times = []
            for _ in range(5):
                start = time.perf_counter()
                predict(model, test_features)
                times.append(time.perf_counter() - start)
            result.update(config=config, validation_accuracy=choice['validation_accuracy'],
                          parameters=choice['parameters'],
                          refit_seconds=refit['fit_seconds'], model_bytes=model_path.stat().st_size,
                          model_ms_per_image=float(np.median(times)*1000/len(truth)))
            if method == 'svm':
                images = self.images[config['mode']][self.indices['test']]
                durations = []
                for repeat in range(6):
                    start = time.perf_counter()
                    array = (images.reshape(len(images), -1) if config['feature'] == 'pixels'
                             else hog_features(images, config['hog_cell_size']))
                    predict(model, torch.from_numpy(array))
                    if repeat:
                        durations.append(time.perf_counter()-start)
                result['feature_and_model_ms_per_image'] = float(np.median(durations)*1000/len(truth))
            results[method] = result
            np.savez(pending / f'{method}_predictions.npz', truth=truth, prediction=prediction, indices=self.indices['test'])
        with contextlib.redirect_stdout(io.StringIO()):
            recognize(self.data / 'Test_Image_2026.bmp', pending, (36,202,912,136))
        save_json(pending / 'classification_results.json', results)
        for path in pending.iterdir():
            path.replace(self.output / path.name)
        pending.rmdir()
        save_json(self.output / 'selection.json', selection)
        self.export_tables()
        print('Final test accuracy', {method: r['test_accuracy'] for method,r in results.items()}, flush=True)

    def export_tables(self):
        records = list(self.state['records'].values())
        save_json(self.output / 'experiments.json', records)
        fields = ['id','method','seed','validation_accuracy','best_epoch','parameters','fc_input','fit_seconds','config']
        with (self.output / 'experiments.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            for record in records:
                writer.writerow({**record,'config': json.dumps(record['config'],sort_keys=True)})

"""Shared data loading, model fitting, prediction and evaluation for Task 7."""
import copy
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from models import CharacterCNN, LinearSVM, hog_features, margin_loss
from prepare_data import preprocess_character


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_data(root, output, modes=("resize", "pad")):
    with (output / "data_split.csv").open() as stream:
        records = list(csv.DictReader(stream))
    classes = sorted({record["label"] for record in records})
    labels = torch.tensor([classes.index(record["label"]) for record in records])
    indices = {split: np.array([index for index, record in enumerate(records)
                               if record["split"] == split])
               for split in ("train", "validation", "test")}
    images = {mode: [] for mode in modes}
    for record in records:
        with Image.open(root / record["path"]) as source:
            gray = source.convert("L")
            for mode in images:
                images[mode].append(preprocess_character(gray, mode))
    return {mode: np.asarray(array) for mode, array in images.items()}, labels, indices, classes


def predict(model, features):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in features.split(256):
            predictions.extend(model(batch).argmax(1).tolist())
    return np.asarray(predictions)


def fit(kind, features, labels, train_indices, validation_indices, config,
        class_count, seed, refit=False):
    torch.manual_seed(seed)
    model = (CharacterCNN(class_count, depth=config.get("depth", 3),
                          pool_size=config.get("pool_size", 4), dropout=config.get("dropout", 0.25),
                          hidden_size=config.get("hidden_size", 64))
             if kind == "cnn" else LinearSVM(class_count, features.shape[1]))
    weight_decay = config.get("weight_decay", 1e-4) if kind == "cnn" else 0
    optimizer_name = config.get('optimizer', 'adam')
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                    momentum=config.get('momentum', .9), weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    elif optimizer_name == 'lbfgs':
        if kind != 'svm' or config.get('loss') != 'squared_hinge':
            raise ValueError('LBFGS trials require the squared-hinge SVM objective.')
        optimizer = torch.optim.LBFGS(model.parameters(), lr=config['lr'], max_iter=10,
                                      history_size=10, line_search_fn='strong_wolfe')
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}')
    train_features, train_labels = features[train_indices], labels[train_indices]
    shuffle_generator = None
    if "shuffle_seed" in config:
        shuffle_generator = torch.Generator().manual_seed(config["shuffle_seed"])
    best_accuracy = -1
    best_epoch = config["epochs"]
    best_state = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum = 0
        if optimizer_name == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                loss = margin_loss(model(train_features), train_labels, model, config['C'], squared=True)
                loss.backward()
                return loss
            optimizer.step(closure)
            loss_sum = float(closure().detach()) * len(train_indices)
        else:
            for batch in torch.randperm(len(train_indices), generator=shuffle_generator).split(config.get("batch_size", 128)):
                optimizer.zero_grad()
                scores = model(train_features[batch])
                if kind == "cnn":
                    loss = torch.nn.functional.cross_entropy(scores, train_labels[batch])
                else:
                    loss = margin_loss(scores, train_labels[batch], model, config["C"],
                                       squared=config.get("loss", "squared_hinge") == "squared_hinge")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch)
        if not np.isfinite(loss_sum):
            raise FloatingPointError(f'Non-finite training loss at epoch {epoch}')
        entry = {"epoch": epoch, "loss": loss_sum / len(train_indices)}
        if not refit:
            prediction = predict(model, features[validation_indices])
            accuracy = float(np.mean(prediction == labels[validation_indices].numpy()))
            entry["validation_accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(entry)
        if epoch == 1 or epoch % 5 == 0:
            print(kind, config, entry, flush=True)
    if not refit:
        model.load_state_dict(best_state)
    return model, {"config": config, "best_epoch": best_epoch,
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "fc_input": model.fc_input if kind == "cnn" else None,
                   "validation_accuracy": best_accuracy if not refit else None,
                   "fit_seconds": time.perf_counter() - start, "history": history}


def evaluate(truth, prediction, classes, path):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(0), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(classes)), classes)
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"test_accuracy": float(np.mean(truth == prediction)), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist(),
            "per_class": {label: {"precision": float(p), "recall": float(r), "f1": float(f)}
                          for label, p, r, f in zip(classes, precision, recall, f1)}}

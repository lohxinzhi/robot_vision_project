"""Learned models and hand-written HOG features for Task 7."""
import numpy as np
import torch
from torch import nn


class CharacterCNN(nn.Module):
    def __init__(self, classes, depth=3, pool_size=4, dropout=0.25, hidden_size=64):
        super().__init__()
        if depth not in (2, 3) or pool_size not in (4, 8):
            raise ValueError("depth must be 2 or 3 and pool_size must be 4 or 8")
        layers = [
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        ]
        channels = 32
        if depth == 3:
            layers.extend([nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()])
            channels = 64
        self.fc_input = channels * pool_size * pool_size
        layers.extend([
            nn.AdaptiveAvgPool2d((pool_size, pool_size)),
            nn.Flatten(), nn.Linear(self.fc_input, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, classes),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, images):
        return self.layers(images)


class LinearSVM(nn.Module):
    def __init__(self, classes, features=324):
        super().__init__()
        self.linear = nn.Linear(features, classes)

    def forward(self, features):
        return self.linear(features)


def squared_hinge_loss(scores, labels, model, regularization):
    """One-vs-rest linear SVM objective with unregularized intercepts."""
    return margin_loss(scores, labels, model, regularization, squared=True)


def margin_loss(scores, labels, model, regularization, squared=False):
    """Hand-written one-vs-rest hinge or squared-hinge objective."""
    targets = -torch.ones_like(scores)
    targets.scatter_(1, labels[:, None], 1)
    margins = torch.relu(1 - targets * scores)
    if squared:
        margins = margins.square()
    penalty = margins.sum(1).mean()
    return 0.5 * model.linear.weight.square().sum() + regularization * penalty


def hog_features(images, cell_size=8):
    """Unsigned HOG, nine bins and overlapping L2-Hys blocks."""
    if cell_size not in (4, 8):
        raise ValueError("cell_size must be 4 or 8")
    cells_per_side = 32 // cell_size
    features = []
    for image in images:
        dy, dx = np.gradient(image)
        magnitude = np.hypot(dx, dy)
        orientation = (np.arctan2(dy, dx) % np.pi) * 9 / np.pi
        lower = np.floor(orientation).astype(int) % 9
        fraction = orientation - np.floor(orientation)
        histogram = np.zeros((cells_per_side, cells_per_side, 9), dtype=np.float32)
        rows, columns = np.indices(image.shape)
        cells = (rows // cell_size, columns // cell_size)
        np.add.at(histogram, (*cells, lower), magnitude * (1 - fraction))
        np.add.at(histogram, (*cells, (lower + 1) % 9), magnitude * fraction)
        blocks = []
        for row in range(cells_per_side - 1):
            for column in range(cells_per_side - 1):
                block = histogram[row:row + 2, column:column + 2].ravel()
                block = block / np.sqrt(block @ block + 1e-6)
                block = np.minimum(block, 0.2)
                blocks.extend(block / np.sqrt(block @ block + 1e-6))
        features.append(blocks)
    return np.asarray(features, dtype=np.float32)

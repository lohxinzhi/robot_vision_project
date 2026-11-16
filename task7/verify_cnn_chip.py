"""Inference-only check for the saved CNN: original dataset not required.
Run: python task7/verify_cnn_chip.py (from repository root)
Requires torch, numpy, opencv-python.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REFERENCE = 'HD44780A00'

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*8*8, 64), nn.ReLU(), nn.Linear(64, 7)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def prep(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    gray = cv2.bitwise_not(gray)  # Task 6 PNGs: white glyph on black
    ys, xs = np.where(gray < 245)
    if not len(xs):
        raise ValueError(f'No foreground: {path}')
    gray = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    gray = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)/255.0

def main():
    checkpoint = ROOT/'results/task7/cnn_final_model.pth'
    # Only load trusted local model checkpoints.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    names = list(state['class_names'])
    model = SimpleCNN()
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    for name,subfolder in [('Pipeline A','task6_characters_cleaned'),('Pipeline B','task6_teammate_characters')]:
        files=[ROOT/'results'/subfolder/f'character_{i:02}.png' for i in range(1,11)]
        images=np.stack([prep(x) for x in files]).astype(np.float32)
        x=torch.from_numpy(images).unsqueeze(1)
        with torch.no_grad():
            idx=model(x).argmax(1).tolist()
        pred=''.join(names[i] for i in idx)
        correct=sum(a==b for a,b in zip(REFERENCE,pred))
        print(f'{name}: {pred}  {correct}/10 ({correct*10}%)')
    print('Reference:',REFERENCE)

if __name__=='__main__':main()

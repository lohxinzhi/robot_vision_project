"""Inference-only check for saved HOG+SVM: original dataset not required.
Run: python task7/verify_svm_chip.py (from repository root)
Requires scikit-learn, scikit-image, joblib, numpy, opencv-python.
WARNING: joblib is pickle-based. Load only this trusted project artifact.
"""
from pathlib import Path
import cv2
import joblib
import numpy as np
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent
REFERENCE = 'HD44780A00'

def prep(path):
    gray=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:raise FileNotFoundError(path)
    gray=cv2.bitwise_not(gray)
    ys,xs=np.where(gray<245)
    if not len(xs):raise ValueError(f'No foreground: {path}')
    gray=gray[ys.min():ys.max()+1,xs.min():xs.max()+1]
    gray=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)/255.0

def features(image):
    return hog(image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)

def main():
    # Only load the trusted local joblib model, not arbitrary downloads.
    saved=joblib.load(ROOT/'results/task7/svm_final_model.joblib')
    model=saved['model']
    for name,subfolder in [('Pipeline A','task6_characters_cleaned'),('Pipeline B','task6_teammate_characters')]:
        files=[ROOT/'results'/subfolder/f'character_{i:02}.png' for i in range(1,11)]
        X=np.stack([features(prep(path)) for path in files]).astype(np.float32)
        pred=''.join(model.predict(X))
        correct=sum(a==b for a,b in zip(REFERENCE,pred))
        print(f'{name}: {pred}  {correct}/10 ({correct*10}%)')
    print('Reference:',REFERENCE)

if __name__=='__main__':main()

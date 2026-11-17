"""Apply the selected Task 7 models to image-processed chip characters."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chip_processing import DEFAULT_CONFIG, extract_glyphs, load_preprocessing
from models import CharacterCNN, LinearSVM, hog_features
from prepare_data import preprocess_character
from training import predict


def recognize(image_path, output, crop, preprocessing=None, save=True):
    image_path, output = Path(image_path), Path(output)
    settings = load_preprocessing() if preprocessing is None else preprocessing
    torch.set_num_threads(4)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError("Crop must be inside the image")
    gray_crop = image[y:y + height, x:x + width]
    checkpoints = {kind: torch.load(output / f"{kind}_model.pt", map_location="cpu", weights_only=True)
                   for kind in ("cnn", "svm")}
    result = {"image": image_path.name,
              "crop_xywh": list(crop), "reference": "HD44780A00",
              "image_processing": settings,
              "connectivity": {}}
    for connectivity in (4, 8):
        glyphs, components, windows, binary, display_mask = extract_glyphs(gray_crop, settings, connectivity)
        mode_result = {"character_count": len(glyphs), "boxes": components,
                       "extraction_windows": windows, "models": {}}
        for kind, checkpoint in checkpoints.items():
            classes = checkpoint["classes"]
            mode = checkpoint["config"]["mode"]
            normalized = np.stack([preprocess_character(glyph, mode) for glyph in glyphs])
            config = checkpoint["config"]
            if kind == "cnn":
                features = normalized[:, None]
                model = CharacterCNN(len(classes), depth=config.get("depth", 3),
                                     pool_size=config.get("pool_size", 4), dropout=config.get("dropout", 0.25),
                                     hidden_size=config.get("hidden_size", 64))
            else:
                features = (normalized.reshape(len(normalized), -1) if config.get("feature", "hog") == "pixels"
                            else hog_features(normalized, cell_size=config.get("hog_cell_size", 8)))
                model = LinearSVM(len(classes), features.shape[1])
            model.load_state_dict(checkpoint["state_dict"])
            prediction = predict(model, torch.from_numpy(features))
            text = "".join(classes[index] for index in prediction)
            reference = result["reference"]
            mode_result["models"][kind] = {
                "prediction": text, "preprocessing": mode,
                "position_accuracy": sum(a == b for a, b in zip(text, reference)) / len(reference)
                if len(text) == len(reference) else None,
            }
        result["connectivity"][str(connectivity)] = mode_result
        if not save:
            continue
        figure, axes = plt.subplots(2, (len(glyphs) + 1) // 2, figsize=(12, 5), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for index, (axis, glyph) in enumerate(zip(axes.flat, glyphs)):
            axis.imshow(glyph, cmap="gray", vmin=0, vmax=255)
            cnn_text = mode_result["models"]["cnn"]["prediction"]
            svm_text = mode_result["models"]["svm"]["prediction"]
            axis.set_title(f"{index + 1}: CNN {cnn_text[index]}, SVM {svm_text[index]}")
        figure.tight_layout()
        figure.savefig(output / f"chip_predictions_{connectivity}.png", dpi=160)
        plt.close(figure)
        figure, axes = plt.subplots(3, 1, figsize=(14, 7))
        for axis, pixels, title in zip(axes, (gray_crop, binary, display_mask),
                                       ("Grayscale ROI", "Location mask", "Extracted recognition masks")):
            axis.imshow(pixels, cmap="gray", vmin=0, vmax=255)
            axis.set_title(title)
            axis.axis("off")
        for box in components:
            axes[1].add_patch(plt.Rectangle((box["x"], box["y"]), box["width"], box["height"],
                                            fill=False, edgecolor="tab:red", linewidth=1))
        figure.tight_layout()
        figure.savefig(output / f"chip_processing_{connectivity}.png", dpi=160)
        plt.close(figure)
    if save:
        (output / "chip_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path(__file__).with_name("dataset_2026") / "Test_Image_2026.bmp")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--crop", type=int, nargs=4, default=(36, 202, 912, 136),
                        metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_CONFIG,
                        help="JSON parameters for chip image processing")
    arguments = parser.parse_args()
    recognize(arguments.image, arguments.output, arguments.crop, load_preprocessing(arguments.preprocessing))
